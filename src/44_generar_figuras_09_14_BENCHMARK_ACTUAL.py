from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


MODEL_DISPLAY = {
    "baseline_persistencia": "Persistence",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn1d": "CNN-1D",
    "cnn_lstm": "CNN-LSTM",
}
MODEL_ORDER = ["baseline_persistencia", "lstm", "gru", "cnn1d", "cnn_lstm"]
NEURAL_ORDER = ["lstm", "gru", "cnn1d", "cnn_lstm"]

SCENARIO_ORDER = [
    "global_2019_2026",
    "pre_covid",
    "covid",
    "post_covid",
    "trimestre_mas_bajo",
    "trimestre_mas_alto",
    "intra_2024_Q1",
    "intra_2024_Q3",
    "intra_2025_Q1",
    "cross_2024_Q1_to_2024_Q3",
    "cross_2024_Q1_to_2025_Q1",
    "cross_pre_to_post",
    "cross_bajo_to_alto",
]
SCENARIO_DISPLAY = {
    "global_2019_2026": "Global chronological split",
    "pre_covid": "Pre-COVID",
    "covid": "COVID-labelled",
    "post_covid": "Post-COVID",
    "trimestre_mas_bajo": "Lowest-price quarter",
    "trimestre_mas_alto": "Highest-price quarter",
    "intra_2024_Q1": "Intra 2024-Q1",
    "intra_2024_Q3": "Intra 2024-Q3",
    "intra_2025_Q1": "Intra 2025-Q1",
    "cross_2024_Q1_to_2024_Q3": "2024-Q1 → 2024-Q3",
    "cross_2024_Q1_to_2025_Q1": "2024-Q1 → 2025-Q1",
    "cross_pre_to_post": "Pre-COVID → Post-COVID",
    "cross_bajo_to_alto": "Low → high quarter",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
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


def load(root: Path):
    metrics = pd.read_csv(
        root / "outputs/scenarios_final_seed42/manifest/final_13_scenarios_seed42_metrics.csv"
    )
    multiseed = pd.read_csv(
        root / "outputs/reproducibility/multiseed_final_global_resumen.csv"
    )
    stats = pd.read_csv(
        root / "outputs/statistical_analysis/global_final_dm_hierarchical_bootstrap.csv"
    )
    return metrics, multiseed, stats


def fig09(root: Path, outdir: Path, metrics: pd.DataFrame, multiseed: pd.DataFrame):
    baseline_rmse = float(
        metrics.loc[
            (metrics["model"] == "baseline_persistencia")
            & (metrics["scenario"] == "global_2019_2026"),
            "test_RMSE",
        ].iloc[0]
    )
    rows = [{"model": "Persistence", "mean": baseline_rmse, "sd": 0.0}]
    for key in NEURAL_ORDER:
        r = multiseed.loc[multiseed["model"] == key].iloc[0]
        rows.append(
            {
                "model": MODEL_DISPLAY[key],
                "mean": float(r["mean_RMSE"]),
                "sd": float(r["sd_RMSE"]),
            }
        )
    plot = pd.DataFrame(rows)
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.errorbar(x, plot["mean"], yerr=plot["sd"], fmt="o", capsize=4, linewidth=1.0)
    ax.axhline(baseline_rmse, linestyle="--", linewidth=0.9)
    ax.set_xticks(x, plot["model"])
    ax.set_ylabel("Global test RMSE (USDT)")
    ax.set_xlabel("Forecasting method")
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ymin = min((plot["mean"] - plot["sd"]).min(), baseline_rmse)
    ymax = max((plot["mean"] + plot["sd"]).max(), baseline_rmse)
    pad = max((ymax - ymin) * 0.20, 0.02)
    ax.set_ylim(ymin - pad, ymax + pad)
    for xi, mean in zip(x, plot["mean"]):
        ax.annotate(f"{mean:.3f}", (xi, mean), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
    save(fig, outdir, "Fig09_global_multiseed_rmse")


def fig10(root: Path, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    baseline_val = None
    for key in NEURAL_ORDER:
        path = root / "outputs/scenarios_final_seed42/history" / f"{key}__global_2019_2026__history.csv"
        df = pd.read_csv(path)
        ax.plot(df["epoca"], df["val_RMSE"], marker="o", markersize=3, linewidth=1.1, label=MODEL_DISPLAY[key])
        baseline_val = float(df["baseline_val_RMSE"].iloc[0])
    if baseline_val is not None:
        ax.axhline(baseline_val, linestyle="--", linewidth=0.9, label="Persistence validation RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (USDT)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, outdir, "Fig10_global_validation_rmse")


def fig11(outdir: Path, metrics: pd.DataFrame):
    pivot = metrics.pivot(index="scenario", columns="model", values="test_RMSE").reindex(
        index=SCENARIO_ORDER, columns=MODEL_ORDER
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("Test RMSE (USDT)")
    ax.set_xticks(np.arange(len(MODEL_ORDER)), [MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCENARIO_ORDER)), [SCENARIO_DISPLAY[s] for s in SCENARIO_ORDER])
    ax.set_xlabel("Forecasting method")
    ax.set_ylabel("Evaluation scenario")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i,j]:.1f}", ha="center", va="center", fontsize=6.2)
    save(fig, outdir, "Fig11_scenario_test_rmse")


def fig12(outdir: Path, metrics: pd.DataFrame):
    neural = metrics[metrics["model"].isin(NEURAL_ORDER)].copy()
    pivot = neural.pivot(
        index="scenario", columns="model", values="delta_RMSE_vs_persistence"
    ).reindex(index=SCENARIO_ORDER, columns=NEURAL_ORDER)
    values = pivot.to_numpy()
    vmax = float(np.nanmax(np.abs(values)))
    norm = SymLogNorm(linthresh=0.10, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(values, aspect="auto", norm=norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("ΔRMSE vs persistence (USDT)")
    ax.set_xticks(np.arange(len(NEURAL_ORDER)), [MODEL_DISPLAY[m] for m in NEURAL_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCENARIO_ORDER)), [SCENARIO_DISPLAY[s] for s in SCENARIO_ORDER])
    ax.set_xlabel("Neural model")
    ax.set_ylabel("Evaluation scenario")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i,j]:+.2f}", ha="center", va="center", fontsize=6.2)
    save(fig, outdir, "Fig12_delta_rmse_vs_persistence")


def fig13(root: Path, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for model in NEURAL_ORDER:
        path = root / "outputs/scenarios_final_seed42/predictions" / f"{model}__global_2019_2026__test.csv"
        df = pd.read_csv(path)
        df["fecha_hora_utc"] = pd.to_datetime(df["fecha_hora_utc"], utc=True, errors="raise")
        df = df.sort_values("fecha_hora_utc").reset_index(drop=True)
        end = df["fecha_hora_utc"].max()
        start = end - pd.Timedelta(days=14)
        view = df.loc[df["fecha_hora_utc"] >= start].copy()
        correction = view["valor_predicho"].to_numpy(float) - view["valor_baseline"].to_numpy(float)
        ax.plot(view["fecha_hora_utc"], correction, linewidth=0.9, label=MODEL_DISPLAY[model])
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("UTC date")
    ax.set_ylabel("Forecast − persistence (USDT)")
    ax.legend(ncol=2, frameon=False)
    loc = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    save(fig, outdir, "Fig13_global_residual_corrections_14d")


def fig14(outdir: Path, stats: pd.DataFrame):
    df = stats.set_index("model").reindex(NEURAL_ORDER).reset_index()
    y = np.arange(len(df))
    delta = df["delta_RMSE"].to_numpy(float)
    low = df["delta_RMSE_ci_low"].to_numpy(float)
    high = df["delta_RMSE_ci_high"].to_numpy(float)
    xerr = np.vstack([delta - low, high - delta])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.errorbar(delta, y, xerr=xerr, fmt="o", capsize=4, linewidth=1.0)
    ax.axvline(0.0, linestyle="--", linewidth=0.9)
    ax.set_yticks(y, [MODEL_DISPLAY[m] for m in df["model"]])
    ax.set_xlabel("ΔRMSE vs persistence (USDT)")
    ax.set_ylabel("Model")
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.invert_yaxis()
    for yi, row in df.iterrows():
        ax.annotate(
            f"Holm p={row['dm_p_holm']:.3f}",
            (float(row["delta_RMSE_ci_high"]), yi),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    save(fig, outdir, "Fig14_global_delta_rmse_bootstrap_ci")


def main():
    parser = argparse.ArgumentParser(description="Generate current manuscript Figures 9-14.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    outdir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "outputs" / "figures_q1_final_complete"
    )
    style()
    metrics, multiseed, stats = load(root)
    fig09(root, outdir, metrics, multiseed)
    fig10(root, outdir)
    fig11(outdir, metrics)
    fig12(outdir, metrics)
    fig13(root, outdir)
    fig14(outdir, stats)

    print("Figures 9-14 generated in:", outdir)


if __name__ == "__main__":
    main()
