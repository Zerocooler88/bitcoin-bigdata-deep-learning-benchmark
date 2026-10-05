from __future__ import annotations

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


BRONZE_DEFAULT = Path(
    "data/bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026.parquet"
)
GOLD_CANDIDATES = [
    Path("data/gold/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_2019_2026.parquet"),
    Path("data/1h/global/btcusdt_spot_1h_model_input_2019_2026.parquet"),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.constrained_layout.use": True,
        }
    )


def save(fig: Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(outdir / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def find_col(columns, *candidates: str) -> str:
    lower = {str(c).lower(): str(c) for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in columns:
        lc = str(c).lower()
        if any(k.lower() in lc for k in candidates):
            return str(c)
    raise KeyError(f"Cannot locate any of {candidates}. Available columns: {list(columns)}")


# TIMESTAMP_WINDOWS_MATPLOTLIB_FIX_V3
def to_datetime_utc(s: pd.Series) -> pd.Series:
    # Robust parser for mixed Unix units occasionally present in exported Spark data.
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="raise")
        absvals = np.abs(vals.to_numpy(dtype=np.float64))

        mask_ns = absvals >= 1e17
        mask_us = (absvals >= 1e14) & (absvals < 1e17)
        mask_ms = (absvals >= 1e11) & (absvals < 1e14)
        mask_s  = (absvals >= 1e8)  & (absvals < 1e11)
        known = mask_ns | mask_us | mask_ms | mask_s

        if not bool(np.all(known)):
            bad = vals.loc[~known].head(10).tolist()
            raise ValueError(
                "Hay timestamps numericos con magnitud no reconocida. "
                f"Ejemplos={bad}"
            )

        parsed = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
        groups = [
            ("ns", mask_ns),
            ("us", mask_us),
            ("ms", mask_ms),
            ("s",  mask_s),
        ]

        counts = {}
        for unit, mask in groups:
            count = int(np.sum(mask))
            counts[unit] = count
            if count == 0:
                continue
            idx = s.index[mask]
            converted = pd.to_datetime(
                vals.loc[idx],
                unit=unit,
                origin="unix",
                utc=True,
                errors="coerce",
            ).dt.tz_convert(None)
            parsed.loc[idx] = converted.to_numpy(dtype='datetime64[ns]')

        if parsed.isna().any():
            nbad = int(parsed.isna().sum())
            samples = vals.loc[parsed.isna()].head(10).tolist()
            raise ValueError(
                f"No se pudieron convertir {nbad} timestamps. Ejemplos={samples}"
            )

        print(
            "Timestamp unit audit:",
            f"s={counts['s']:,}",
            f"ms={counts['ms']:,}",
            f"us={counts['us']:,}",
            f"ns={counts['ns']:,}",
        )
    else:
        parsed = pd.to_datetime(s, utc=True, errors="raise").dt.tz_convert(None)

    years = pd.DatetimeIndex(parsed).year
    ymin = int(np.min(years))
    ymax = int(np.max(years))
    if ymin < 2018 or ymax > 2027:
        bad_mask = (years < 2018) | (years > 2027)
        bad_values = parsed.loc[bad_mask].head(10).tolist()
        raise ValueError(
            "Rango temporal inesperado despues del parseo mixto: "
            f"{parsed.min()} -> {parsed.max()}. "
            f"Ejemplos fuera de rango={bad_values}"
        )

    print(
        "Timestamp range audit:",
        parsed.min(),
        "->",
        parsed.max(),
    )
    return parsed

def load_bronze(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Bronze parquet not found:\n{path}\n"
            "Expected the authoritative one-minute Bronze materialization."
        )
    df = pd.read_parquet(path)

    ts = find_col(df.columns, "open_time", "open_time_utc", "fecha_hora_utc", "timestamp", "datetime")
    close = find_col(df.columns, "close")
    open_ = find_col(df.columns, "open")
    high = find_col(df.columns, "high")
    low = find_col(df.columns, "low")
    volume = find_col(df.columns, "volume")
    trades = find_col(df.columns, "number_of_trades", "trades")

    rename = {
        ts: "timestamp",
        close: "close",
        open_: "open",
        high: "high",
        low: "low",
        volume: "volume",
        trades: "number_of_trades",
    }

    optional_candidates = {
        "quote_asset_volume": ("quote_asset_volume", "quote_volume"),
        "taker_buy_base_volume": (
            "taker_buy_base_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_base",
        ),
        "taker_buy_quote_volume": (
            "taker_buy_quote_volume",
            "taker_buy_quote_asset_volume",
            "taker_buy_quote",
        ),
    }
    for out_name, candidates in optional_candidates.items():
        try:
            source = find_col(df.columns, *candidates)
            rename[source] = out_name
        except KeyError:
            pass

    keep = list(dict.fromkeys(rename))
    x = df[keep].rename(columns=rename).copy()
    x["timestamp"] = to_datetime_utc(x["timestamp"])
    x = x.sort_values("timestamp").reset_index(drop=True)

    print(
        "Bronze timestamp audit:",
        x["timestamp"].min(),
        "->",
        x["timestamp"].max(),
        "| rows:",
        f"{len(x):,}",
    )

    for c in [c for c in x.columns if c != "timestamp"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Derived analytical view. No rows are removed from the authoritative source.
    previous_close = x["close"].shift(1)
    x["log_return_1m"] = np.log(x["close"] / previous_close)
    x["abs_log_return_1m"] = x["log_return_1m"].abs()
    x["relative_high_low_range"] = (x["high"] - x["low"]) / x["close"]
    x["candle_body_ratio"] = (x["close"] - x["open"]).abs() / x["close"]
    x["rolling_volatility_60m"] = x["log_return_1m"].rolling(60).std()

    return x


def panel_letters(axes) -> None:
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], np.ravel(axes)):
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )


def figure03_overview(df: pd.DataFrame, outdir: Path) -> dict:
    # Figure 3(c,d): central 99% display range keeps linear units but prevents
    # a few extreme values from collapsing the histogram against the left edge.
    vol_q99 = float(df["volume"].quantile(0.99))
    trades_q99 = float(df["number_of_trades"].quantile(0.99))

    display_ts = df.iloc[::240].copy()
    vol = df.loc[df["volume"].between(0, vol_q99), "volume"].dropna()
    trades = df.loc[
        df["number_of_trades"].between(0, trades_q99),
        "number_of_trades",
    ].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    axes[0, 0].plot(display_ts["timestamp"].dt.to_pydatetime(), display_ts["close"].to_numpy(), linewidth=0.8)
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Close price (USDT)")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axes[0, 0].xaxis.set_major_locator(locator)
    axes[0, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    axes[0, 1].hist(df["close"].dropna(), bins=70)
    axes[0, 1].set_xlabel("Close price (USDT)")
    axes[0, 1].set_ylabel("Frequency")

    axes[1, 0].hist(vol, bins=70)
    axes[1, 0].set_xlim(0, vol_q99)
    axes[1, 0].set_xlabel("Trading volume (central 99% display range)")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(trades, bins=70)
    axes[1, 1].set_xlim(0, trades_q99)
    axes[1, 1].set_xlabel("Number of trades (central 99% display range)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    panel_letters(axes)
    save(fig, outdir, "Fig03_bronze_overview_CORRECTED")
    return {"volume_p99": vol_q99, "trades_p99": trades_q99}


def figure04_monthly_price(df: pd.DataFrame, outdir: Path) -> None:
    monthly = (
        df.set_index("timestamp")["close"]
        .resample("MS")
        .mean()
        .dropna()
        .rename("mean_close")
        .reset_index()
    )
    monthly["month"] = monthly["timestamp"].dt.strftime("%Y-%m")
    low = monthly.nsmallest(10, "mean_close").sort_values("mean_close")
    high = monthly.nlargest(10, "mean_close").sort_values("mean_close", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    axes[0].bar(low["month"], low["mean_close"])
    axes[0].set_ylabel("Monthly average close (USDT)")
    axes[0].set_xlabel("Month")
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].text(0.01, 0.98, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")

    axes[1].bar(high["month"], high["mean_close"])
    axes[1].set_ylabel("Monthly average close (USDT)")
    axes[1].set_xlabel("Month")
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].text(0.01, 0.98, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")

    save(fig, outdir, "Fig04_monthly_price_contrasts")


def figure05_returns(df: pd.DataFrame, outdir: Path) -> dict:
    r = df["log_return_1m"].dropna()
    abs_r = r.abs()
    q999 = float(abs_r.quantile(0.999))

    # Figure 5(a): display the central ±P99.9(|log r|) range. The omitted
    # extreme tail remains in the dataset and is reported separately in tables.
    central = r[r.between(-q999, q999)]

    monthly_vol = (
        df.set_index("timestamp")["log_return_1m"]
        .resample("MS")
        .std()
        .dropna()
    )
    rolling = df[["timestamp", "rolling_volatility_60m"]].dropna().iloc[::60]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes[0, 0].hist(central, bins=80)
    axes[0, 0].set_xlim(-q999, q999)
    axes[0, 0].set_xlabel("One-minute log return (±P99.9 |log r| display)")
    axes[0, 0].set_ylabel("Frequency")

    axes[0, 1].boxplot(r.to_numpy(), vert=True, showfliers=True)
    axes[0, 1].set_xticks([1], ["Bronze 1-minute"])
    axes[0, 1].set_ylabel("One-minute log return")

    axes[1, 0].plot(monthly_vol.index.to_pydatetime(), monthly_vol.values, linewidth=1.0)
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Monthly volatility (std. dev.)")
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axes[1, 0].xaxis.set_major_locator(loc)
    axes[1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    axes[1, 1].plot(
        rolling["timestamp"].dt.to_pydatetime(),
        rolling["rolling_volatility_60m"].to_numpy(),
        linewidth=0.65,
    )
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Rolling volatility (60 min)")
    loc2 = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axes[1, 1].xaxis.set_major_locator(loc2)
    axes[1, 1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc2))

    panel_letters(axes)
    save(fig, outdir, "Fig05_high_frequency_returns_CORRECTED")
    return {"p999_abs_log_return": q999}


def figure06_volatility_extremes(df: pd.DataFrame, outdir: Path) -> None:
    monthly = (
        df.set_index("timestamp")["log_return_1m"]
        .resample("MS")
        .agg(["std", "count"])
        .dropna()
    )
    monthly["month"] = monthly.index.strftime("%Y-%m")
    high = monthly.nlargest(10, "std")
    low = monthly.nsmallest(10, "std")
    p999 = float(df["abs_log_return_1m"].quantile(0.999))

    monthly_extreme = (
        df.assign(extreme=df["abs_log_return_1m"] >= p999)
        .set_index("timestamp")["extreme"]
        .resample("MS")
        .sum()
    )

    sample_range = df["relative_high_low_range"].dropna().iloc[::10]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes[0, 0].bar(high["month"], high["std"])
    axes[0, 0].set_ylabel("Monthly volatility")
    axes[0, 0].tick_params(axis="x", rotation=55)

    axes[0, 1].bar(low["month"], low["std"])
    axes[0, 1].set_ylabel("Monthly volatility")
    axes[0, 1].tick_params(axis="x", rotation=55)

    axes[1, 0].boxplot(sample_range.to_numpy(), vert=True, showfliers=True)
    axes[1, 0].set_xticks([1], ["Bronze 1-minute"])
    axes[1, 0].set_ylabel("Relative high-low range")

    axes[1, 1].plot(monthly_extreme.index.to_pydatetime(), monthly_extreme.values, linewidth=0.9)
    axes[1, 1].set_ylabel("Extreme-event count")
    axes[1, 1].set_xlabel("Month")
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axes[1, 1].xaxis.set_major_locator(loc)
    axes[1, 1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    panel_letters(axes)
    save(fig, outdir, "Fig06_volatility_extreme_observations")


def figure07_relationships(df: pd.DataFrame, outdir: Path) -> None:
    sample = df.iloc[::240].copy()

    corr_cols = [
        "close",
        "volume",
        "number_of_trades",
        "relative_high_low_range",
        "candle_body_ratio",
        "log_return_1m",
        "rolling_volatility_60m",
    ]
    for optional in [
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]:
        if optional in df.columns:
            corr_cols.append(optional)

    corr = df[corr_cols].corr(numeric_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    axes[0, 0].scatter(
        sample["abs_log_return_1m"],
        sample["volume"],
        s=5,
        alpha=0.22,
    )
    axes[0, 0].set_xlabel("Absolute one-minute log return")
    axes[0, 0].set_ylabel("Trading volume")

    axes[0, 1].scatter(
        sample["close"],
        sample["volume"],
        s=5,
        alpha=0.22,
    )
    axes[0, 1].set_xlabel("Close price (USDT)")
    axes[0, 1].set_ylabel("Trading volume")

    axes[1, 0].scatter(
        sample["number_of_trades"],
        sample["volume"],
        s=5,
        alpha=0.22,
    )
    axes[1, 0].set_xlabel("Number of trades")
    axes[1, 0].set_ylabel("Trading volume")

    im = axes[1, 1].imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    labels = [
        c.replace("number_of_trades", "trades")
        .replace("relative_high_low_range", "HL range")
        .replace("candle_body_ratio", "body")
        .replace("log_return_1m", "log r")
        .replace("rolling_volatility_60m", "roll vol")
        .replace("quote_asset_volume", "quote vol")
        .replace("taker_buy_base_volume", "taker base")
        .replace("taker_buy_quote_volume", "taker quote")
        for c in corr.columns
    ]
    axes[1, 1].set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    axes[1, 1].set_yticks(np.arange(len(labels)), labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            axes[1, 1].text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.03, label="Pearson correlation")

    panel_letters(axes)
    save(fig, outdir, "Fig07_bronze_relationships")


def figure08_gold_monthly(gold_path: Path, outdir: Path) -> None:
    if not gold_path.exists():
        raise FileNotFoundError(gold_path)
    g = pd.read_parquet(gold_path)
    ts_col = find_col(
        g.columns,
        "fecha_hora_utc",
        "open_datetime_utc",
        "datetime_utc",
        "open_time",
        "timestamp",
        "datetime",
    )
    close_col = find_col(g.columns, "close")
    ts = to_datetime_utc(g[ts_col])
    close = pd.to_numeric(g[close_col], errors="coerce")
    tmp = pd.DataFrame({"timestamp": ts, "close": close}).dropna()
    monthly = tmp.set_index("timestamp")["close"].resample("MS").mean()

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.plot(monthly.index.to_pydatetime(), monthly.values, linewidth=1.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average monthly close (USDT)")
    loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    save(fig, outdir, "Fig08_gold_monthly_close")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate current manuscript Figures 3-8 from authoritative Bronze/Gold data."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--bronze", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    bronze = (args.bronze or root / BRONZE_DEFAULT).resolve()

    if args.gold is not None:
        gold = args.gold.resolve()
    else:
        gold = next(
            (root / p for p in GOLD_CANDIDATES if (root / p).exists()),
            root / GOLD_CANDIDATES[0],
        )

    outdir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "outputs" / "figures_q1_final_complete"
    )

    style()
    df = load_bronze(bronze)

    audit = {}
    audit.update(figure03_overview(df, outdir))
    figure04_monthly_price(df, outdir)
    audit.update(figure05_returns(df, outdir))
    figure06_volatility_extremes(df, outdir)
    figure07_relationships(df, outdir)
    figure08_gold_monthly(gold, outdir)

    audit_path = outdir / "Fig03_Fig05_axis_display_audit.txt"
    audit_path.write_text(
        "\n".join(
            [
                "Display-only axis controls; no source rows are deleted from the analytical dataset.",
                f"Fig. 3(c) volume P99 upper display limit: {audit['volume_p99']:.12g}",
                f"Fig. 3(d) trades P99 upper display limit: {audit['trades_p99']:.12g}",
                f"Fig. 5(a) symmetric P99.9 |log return| display limit: ±{audit['p999_abs_log_return']:.12g}",
                "Full maxima/extreme observations remain available in the manuscript tables.",
            ]
        ),
        encoding="utf-8",
    )

    print("Figures 3-8 generated in:", outdir)
    print(audit_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
