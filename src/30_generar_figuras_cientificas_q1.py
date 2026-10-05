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

MODEL_ORDER = [
    "baseline_persistencia",
    "lstm",
    "gru",
    "cnn1d",
    "cnn_lstm",
]

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
    "global_2019_2026": "Global 2019–2026",
    "pre_covid": "Pre-COVID",
    "covid": "COVID",
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


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_inputs(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_metrics = pd.read_csv(
        root / "outputs/scenarios_final_seed42/manifest/"
        "final_13_scenarios_seed42_metrics.csv"
    )
    multiseed = pd.read_csv(
        root / "outputs/reproducibility/" "multiseed_final_global_resumen.csv"
    )
    stats = pd.read_csv(
        root / "outputs/statistical_analysis/"
        "global_final_dm_hierarchical_bootstrap.csv"
    )
    efficiency = pd.read_csv(
        root / "outputs/performance/" "model_computational_efficiency_final.csv"
    )
    return scenario_metrics, multiseed, stats, efficiency


def figure10_multiseed_rmse(root: Path, outdir: Path, multiseed: pd.DataFrame) -> None:
    baseline = pd.read_csv(
        root / "outputs/scenarios_final_seed42/manifest/"
        "final_13_scenarios_seed42_metrics.csv"
    )
    baseline_values = pd.to_numeric(
        baseline.loc[
            (baseline["model"] == "baseline_persistencia")
            & (baseline["scenario"] == "global_2019_2026"),
            "test_RMSE",
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)
    if baseline_values.size != 1:
        raise ValueError("Expected exactly one global persistence RMSE value.")
    baseline_rmse: float = float(baseline_values[0])

    rows = [{"model": "Persistence", "mean": baseline_rmse, "sd": 0.0}]
    for key in NEURAL_ORDER:
        subset = multiseed.loc[multiseed["model"] == key]
        if len(subset) != 1:
            raise ValueError(f"Expected exactly one multiseed summary row for {key}.")
        mean_values = pd.to_numeric(subset["mean_RMSE"], errors="raise").to_numpy(
            dtype=np.float64
        )
        sd_values = pd.to_numeric(subset["sd_RMSE"], errors="raise").to_numpy(
            dtype=np.float64
        )
        rows.append(
            {
                "model": MODEL_DISPLAY[key],
                "mean": float(mean_values[0]),
                "sd": float(sd_values[0]),
            }
        )

    plot = pd.DataFrame(rows)
    x = np.arange(len(plot), dtype=float)
    means = pd.to_numeric(plot["mean"], errors="raise").to_numpy(dtype=float)
    sds = pd.to_numeric(plot["sd"], errors="raise").to_numpy(dtype=float)
    labels: list[str] = [str(value) for value in plot["model"].tolist()]

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="o",
        capsize=4,
        linewidth=1.0,
        markersize=5,
    )
    ax.axhline(baseline_rmse, linestyle="--", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Global test RMSE (USDT)")
    ax.set_xlabel("Forecasting method")
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)

    ymin = min(float(np.min(means - sds)), baseline_rmse)
    ymax = max(float(np.max(means + sds)), baseline_rmse)
    pad = max((ymax - ymin) * 0.20, 0.02)
    ax.set_ylim(ymin - pad, ymax + pad)

    for xi, mean in zip(x, means):
        ax.annotate(
            f"{float(mean):.3f}",
            (float(xi), float(mean)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    save_figure(fig, outdir, "Fig10_global_multiseed_rmse")


def figure11_validation_history(root: Path, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    baseline_val = None

    for key in NEURAL_ORDER:
        path = (
            root
            / "outputs/scenarios_final_seed42/history"
            / f"{key}__global_2019_2026__history.csv"
        )
        df = pd.read_csv(path)
        ax.plot(
            df["epoca"],
            df["val_RMSE"],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=MODEL_DISPLAY[key],
        )
        baseline_values = pd.to_numeric(
            df["baseline_val_RMSE"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if baseline_values.size == 0:
            raise ValueError(f"No baseline validation RMSE found for {key}.")
        baseline_val = float(baseline_values[0])

    if baseline_val is not None:
        ax.axhline(
            baseline_val,
            linestyle="--",
            linewidth=0.9,
            label="Persistence validation RMSE",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (USDT)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(fig, outdir, "Fig11_global_validation_rmse")


def figure12_scenario_rmse(outdir: Path, metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot(
        index="scenario", columns="model", values="test_RMSE"
    ).reindex(index=SCENARIO_ORDER, columns=MODEL_ORDER)

    values = pivot.to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(values, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("Test RMSE (USDT)")

    x_ticks: list[float] = [float(i) for i in range(len(MODEL_ORDER))]
    x_labels: list[str] = [str(MODEL_DISPLAY[m]) for m in MODEL_ORDER]
    y_ticks: list[float] = [float(i) for i in range(len(SCENARIO_ORDER))]
    y_labels: list[str] = [str(SCENARIO_DISPLAY[s]) for s in SCENARIO_ORDER]

    ax.set_xticks(
        x_ticks,
        x_labels,
        rotation=25,
        ha="right",
    )
    ax.set_yticks(
        y_ticks,
        y_labels,
    )
    ax.set_xlabel("Forecasting method")
    ax.set_ylabel("Evaluation scenario")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value: float = float(values[i, j])
            ax.text(
                float(j),
                float(i),
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.2,
            )

    save_figure(fig, outdir, "Fig12_scenario_test_rmse")


def figure13_delta_vs_persistence(outdir: Path, metrics: pd.DataFrame) -> None:
    neural = metrics[metrics["model"].isin(NEURAL_ORDER)].copy()
    pivot = neural.pivot(
        index="scenario",
        columns="model",
        values="delta_RMSE_vs_persistence",
    ).reindex(index=SCENARIO_ORDER, columns=NEURAL_ORDER)

    values = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values)))
    norm = SymLogNorm(linthresh=0.10, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(values, aspect="auto", norm=norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("ΔRMSE vs persistence (USDT)")

    x_ticks: list[float] = [float(i) for i in range(len(NEURAL_ORDER))]
    x_labels: list[str] = [str(MODEL_DISPLAY[m]) for m in NEURAL_ORDER]
    y_ticks: list[float] = [float(i) for i in range(len(SCENARIO_ORDER))]
    y_labels: list[str] = [str(SCENARIO_DISPLAY[s]) for s in SCENARIO_ORDER]

    ax.set_xticks(
        x_ticks,
        x_labels,
        rotation=25,
        ha="right",
    )
    ax.set_yticks(
        y_ticks,
        y_labels,
    )
    ax.set_xlabel("Neural model")
    ax.set_ylabel("Evaluation scenario")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value: float = float(values[i, j])
            ax.text(
                float(j),
                float(i),
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
            )

    save_figure(fig, outdir, "Fig13_delta_rmse_vs_persistence")


def figure14_prediction_trace(root: Path, outdir: Path) -> None:
    pred_path = (
        root / "outputs/scenarios_final_seed42/predictions/"
        "gru__global_2019_2026__test.csv"
    )
    df = pd.read_csv(pred_path)
    df["fecha_hora_utc"] = pd.to_datetime(df["fecha_hora_utc"], utc=True)
    df = df.sort_values("fecha_hora_utc").reset_index(drop=True)

    end = df["fecha_hora_utc"].max()
    start = end - pd.Timedelta(days=14)
    view = df[df["fecha_hora_utc"] >= start]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(
        view["fecha_hora_utc"],
        view["valor_real"],
        linewidth=1.2,
        label="Observed close",
    )
    ax.plot(
        view["fecha_hora_utc"],
        view["valor_predicho"],
        linewidth=1.0,
        label="GRU forecast",
    )
    ax.plot(
        view["fecha_hora_utc"],
        view["valor_baseline"],
        linewidth=0.9,
        linestyle="--",
        label="Persistence",
    )
    ax.set_xlabel("UTC date")
    ax.set_ylabel("BTCUSDT close (USDT)")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    save_figure(fig, outdir, "Fig14_global_prediction_trace_14d")


def figure15_bootstrap_forest(outdir: Path, stats: pd.DataFrame) -> None:
    df = stats.set_index("model").reindex(NEURAL_ORDER).reset_index()
    y = np.arange(len(df), dtype=np.float64)

    delta = df["delta_RMSE"].to_numpy(float)
    low = df["delta_RMSE_ci_low"].to_numpy(float)
    high = df["delta_RMSE_ci_high"].to_numpy(float)
    xerr = np.vstack([delta - low, high - delta])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.errorbar(
        delta,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.0,
        markersize=5,
    )
    ax.axvline(0.0, linestyle="--", linewidth=0.9)
    model_labels: list[str] = [
        str(MODEL_DISPLAY[str(model)]) for model in df["model"].tolist()
    ]
    ax.set_yticks(y, model_labels)
    ax.set_xlabel("ΔRMSE vs persistence (USDT)")
    ax.set_ylabel("Model")
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.invert_yaxis()

    holm_p = pd.to_numeric(df["dm_p_holm"], errors="raise").to_numpy(dtype=float)
    ci_high = pd.to_numeric(df["delta_RMSE_ci_high"], errors="raise").to_numpy(
        dtype=float
    )

    for yi in range(len(df)):
        ax.annotate(
            f"Holm p={float(holm_p[yi]):.3f}",
            (float(ci_high[yi]), float(yi)),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )

    save_figure(fig, outdir, "Fig15_global_delta_rmse_bootstrap_ci")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality Q1 figures 10–15."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of Bitcoin_Benchmark_Q1_v2_STEADY_START.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    outdir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "outputs/figures_q1_final"
    )

    apply_publication_style()
    metrics, multiseed, stats, _ = load_inputs(root)

    figure10_multiseed_rmse(root, outdir, multiseed)
    figure11_validation_history(root, outdir)
    figure12_scenario_rmse(outdir, metrics)
    figure13_delta_vs_persistence(outdir, metrics)
    figure14_prediction_trace(root, outdir)
    figure15_bootstrap_forest(outdir, stats)

    print("Publication figures generated:")
    for path in sorted(outdir.glob("Fig1*.*")):
        print(path)


if __name__ == "__main__":
    main()
