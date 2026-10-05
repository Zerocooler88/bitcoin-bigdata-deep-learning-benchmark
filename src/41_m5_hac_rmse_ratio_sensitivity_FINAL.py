from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [11, 23, 42, 84, 168]
MODELS = ["lstm", "gru", "cnn1d", "cnn_lstm"]
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn1d": "CNN-1D",
    "cnn_lstm": "CNN-LSTM",
}

# Includes lag 0 as an IID/no-autocorrelation reference and lag 10 as the
# common Newey-West rule-of-thumb floor(4*(n/100)^(2/9)) for n=9358.
HAC_LAGS = [0, 1, 6, 10, 12, 24, 48, 72, 168]

BLOCK_LENGTH = 168
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261004
EXPECTED_N_TEST = 9358
METRIC_TOLERANCE = 1e-3


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_prediction_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV:\n{path}")

    df = pd.read_csv(path)
    required = {
        "fecha_hora_utc",
        "valor_real",
        "valor_predicho",
        "valor_baseline",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    df["fecha_hora_utc"] = pd.to_datetime(
        df["fecha_hora_utc"], utc=True, errors="raise"
    )
    return df.sort_values("fecha_hora_utc").reset_index(drop=True)


def assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    if len(candidate) != len(reference):
        raise ValueError(f"{label}: n={len(candidate)} != {len(reference)}")
    if not reference["fecha_hora_utc"].equals(candidate["fecha_hora_utc"]):
        raise ValueError(f"{label}: target timestamps are not identical.")
    for col in ["valor_real", "valor_baseline"]:
        a = reference[col].to_numpy(float)
        b = candidate[col].to_numpy(float)
        if not np.allclose(a, b, rtol=0.0, atol=1e-9):
            raise ValueError(f"{label}: {col} does not match reference.")


def newey_west_long_run_variance(d: np.ndarray, max_lag: int) -> float:
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    centered = d - d.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    max_lag = min(int(max_lag), n - 1)

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        lrv += 2.0 * weight * gamma

    return max(lrv, 1e-15)


def dm_from_loss_differential(d: np.ndarray, hac_lag: int) -> tuple[float, float]:
    lrv = newey_west_long_run_variance(d, hac_lag)
    se = math.sqrt(lrv / len(d))
    stat = float(d.mean() / se)
    p = float(math.erfc(abs(stat) / math.sqrt(2.0)))
    return stat, p


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def circular_block_index_batch(
    n: int,
    block_length: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(batch_size, n_blocks))
    rel = np.arange(block_length, dtype=np.int64)
    idx = (starts[:, :, None] + rel[None, None, :]) % n
    return idx.reshape(batch_size, -1)[:, :n]


def bootstrap_rmse_ratio(
    model_sqerr: np.ndarray,
    baseline_sqerr: np.ndarray,
    reps: int,
    block_length: int,
    seed: int,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      hierarchical_ratio: resamples the five seeds AND circular time blocks.
      time_only_ratio: keeps the five seeds fixed and resamples time blocks only.
    Point estimand is mean(seed-specific RMSE) / persistence RMSE.
    """
    rng = np.random.default_rng(seed)
    n_seeds, n_time = model_sqerr.shape
    hierarchical = np.empty(reps, dtype=np.float64)
    time_only = np.empty(reps, dtype=np.float64)

    cursor = 0
    while cursor < reps:
        b = min(batch_size, reps - cursor)
        idx = circular_block_index_batch(
            n=n_time,
            block_length=block_length,
            batch_size=b,
            rng=rng,
        )

        # B x n_seeds: MSE for each original seed under each temporal resample.
        mse_by_seed = np.empty((b, n_seeds), dtype=np.float64)
        for s in range(n_seeds):
            mse_by_seed[:, s] = model_sqerr[s][idx].mean(axis=1)

        rmse_by_seed = np.sqrt(mse_by_seed)
        base_rmse = np.sqrt(baseline_sqerr[idx].mean(axis=1))

        # Fixed-seed temporal bootstrap.
        time_only[cursor : cursor + b] = (
            rmse_by_seed.mean(axis=1) / base_rmse
        )

        # Hierarchical seed + time bootstrap.
        sampled_seed_idx = rng.integers(
            0, n_seeds, size=(b, n_seeds)
        )
        sampled_rmse = np.take_along_axis(
            rmse_by_seed, sampled_seed_idx, axis=1
        )
        hierarchical[cursor : cursor + b] = (
            sampled_rmse.mean(axis=1) / base_rmse
        )

        cursor += b

    return hierarchical, time_only


def percentile_interval(x: np.ndarray) -> tuple[float, float]:
    return (
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    )


def load_inputs(final_root: Path):
    baseline_path = (
        final_root
        / "baseline_persistencia"
        / "predictions"
        / "baseline_persistencia__global_2019_2026__test.csv"
    )
    reference = read_prediction_file(baseline_path)

    if len(reference) != EXPECTED_N_TEST:
        raise RuntimeError(
            f"Global test has {len(reference)} rows; expected {EXPECTED_N_TEST}."
        )

    y = reference["valor_real"].to_numpy(float)
    baseline_pred = reference["valor_baseline"].to_numpy(float)
    baseline_sqerr = (baseline_pred - y) ** 2

    matrices = {}
    audit_rows = []

    for model in MODELS:
        preds = []
        for seed in SEEDS:
            path = (
                final_root
                / f"seed_{seed}"
                / "predictions"
                / f"{model}__global_2019_2026__test.csv"
            )
            df = read_prediction_file(path)
            assert_aligned(reference, df, f"{model} seed={seed}")
            pred = df["valor_predicho"].to_numpy(float)
            preds.append(pred)
            audit_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "n": len(df),
                    "csv_RMSE": rmse(y, pred),
                    "sha256": sha256_file(path),
                    "path": str(path),
                }
            )
        matrices[model] = np.vstack(preds)

    baseline_audit = {
        "n": len(reference),
        "baseline_RMSE": float(np.sqrt(baseline_sqerr.mean())),
        "baseline_MAE": float(np.mean(np.abs(baseline_pred - y))),
        "first_target_utc": str(reference["fecha_hora_utc"].iloc[0]),
        "last_target_utc": str(reference["fecha_hora_utc"].iloc[-1]),
        "baseline_sha256": sha256_file(baseline_path),
        "baseline_path": str(baseline_path),
    }

    return reference, y, baseline_pred, baseline_sqerr, matrices, pd.DataFrame(audit_rows), baseline_audit


def crosscheck_against_definitive_metrics(
    audit: pd.DataFrame,
    expected_detail_path: Path,
    skip: bool,
) -> dict:
    status = {
        "performed": False,
        "expected_detail_path": str(expected_detail_path),
        "max_abs_rmse_difference": None,
        "passed": None,
    }

    if skip:
        return status

    if not expected_detail_path.exists():
        raise FileNotFoundError(
            "Definitive metrics detail not found; refusing to run without "
            "cross-check. Expected:\n"
            f"{expected_detail_path}\n"
            "Use --skip-metric-crosscheck only for debugging, never for manuscript results."
        )

    expected = pd.read_csv(expected_detail_path)
    required = {"seed", "model", "test_RMSE"}
    missing = required - set(expected.columns)
    if missing:
        raise ValueError(
            f"{expected_detail_path.name}: missing {sorted(missing)}"
        )

    merged = audit.merge(
        expected[["seed", "model", "test_RMSE"]],
        on=["seed", "model"],
        how="left",
        validate="one_to_one",
    )
    if merged["test_RMSE"].isna().any():
        raise RuntimeError("Some seed/model pairs are absent from definitive metrics detail.")

    merged["abs_diff"] = np.abs(
        merged["csv_RMSE"] - merged["test_RMSE"]
    )
    max_diff = float(merged["abs_diff"].max())

    status.update(
        {
            "performed": True,
            "max_abs_rmse_difference": max_diff,
            "passed": bool(max_diff <= METRIC_TOLERANCE),
        }
    )

    if max_diff > METRIC_TOLERANCE:
        bad = merged.sort_values("abs_diff", ascending=False).head(8)
        raise RuntimeError(
            "Prediction CSVs do NOT match the definitive multi-seed metrics. "
            f"Max |RMSE_csv - RMSE_definitive|={max_diff:.6f} > {METRIC_TOLERANCE}.\n"
            "This usually means an older pre-strict archive was selected.\n\n"
            + bad[
                ["seed", "model", "csv_RMSE", "test_RMSE", "abs_diff"]
            ].to_string(index=False)
        )

    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    default_project = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--final-root",
        type=Path,
        default=default_project / "outputs" / "multiseed_final_deterministic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_project / "outputs" / "statistical_analysis_m5_sensitivity",
    )
    parser.add_argument(
        "--expected-detail",
        type=Path,
        default=default_project / "outputs" / "reproducibility" / "multiseed_final_global_detalle.csv",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=BOOTSTRAP_REPS,
    )
    parser.add_argument(
        "--skip-metric-crosscheck",
        action="store_true",
        help="Debug only. Do not use for manuscript results.",
    )
    args = parser.parse_args()

    final_root = args.final_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        reference,
        y,
        baseline_pred,
        baseline_sqerr,
        matrices,
        audit,
        baseline_audit,
    ) = load_inputs(final_root)

    metric_crosscheck = crosscheck_against_definitive_metrics(
        audit=audit,
        expected_detail_path=args.expected_detail.resolve(),
        skip=args.skip_metric_crosscheck,
    )

    audit.to_csv(
        output_dir / "m5_prediction_input_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 1) HAC / Newey-West sensitivity for squared-loss DM tests
    # ------------------------------------------------------------------
    hac_rows = []
    loss_differentials = {}

    for model in MODELS:
        model_sqerr = (matrices[model] - y[None, :]) ** 2
        d = model_sqerr.mean(axis=0) - baseline_sqerr
        loss_differentials[model] = d

        for lag in HAC_LAGS:
            stat, p = dm_from_loss_differential(d, lag)
            hac_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "hac_lag": lag,
                    "dm_statistic": stat,
                    "dm_p_value": p,
                    "mean_squared_loss_difference": float(d.mean()),
                }
            )

    hac = pd.DataFrame(hac_rows)

    # Holm correction is applied across the four model comparisons separately
    # at every HAC lag.
    out_parts = []
    for lag, block in hac.groupby("hac_lag", sort=True):
        block = block.copy()
        block["dm_p_holm"] = holm_adjust(
            block["dm_p_value"].to_numpy(float)
        )
        block["significant_after_holm_0_05"] = (
            block["dm_p_holm"] < 0.05
        )
        out_parts.append(block)

    hac = pd.concat(out_parts, ignore_index=True).sort_values(
        ["hac_lag", "model"]
    )
    hac.to_csv(
        output_dir / "m5_hac_sensitivity_dm.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hac_summary_rows = []
    for model in MODELS:
        b = hac[hac["model"] == model]
        hac_summary_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "min_raw_p_across_lags": float(b["dm_p_value"].min()),
                "max_raw_p_across_lags": float(b["dm_p_value"].max()),
                "min_holm_p_across_lags": float(b["dm_p_holm"].min()),
                "max_holm_p_across_lags": float(b["dm_p_holm"].max()),
                "any_significant_after_holm_0_05": bool(
                    b["significant_after_holm_0_05"].any()
                ),
                "dm_stat_min": float(b["dm_statistic"].min()),
                "dm_stat_max": float(b["dm_statistic"].max()),
            }
        )
    hac_summary = pd.DataFrame(hac_summary_rows)
    hac_summary.to_csv(
        output_dir / "m5_hac_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 2) Hierarchical circular-block bootstrap of RMSE ratio
    # ------------------------------------------------------------------
    ratio_rows = []

    base_rmse = float(np.sqrt(baseline_sqerr.mean()))
    for model_index, model in enumerate(MODELS):
        model_sqerr = (matrices[model] - y[None, :]) ** 2
        seed_rmses = np.sqrt(model_sqerr.mean(axis=1))
        mean_model_rmse = float(seed_rmses.mean())
        point_ratio = mean_model_rmse / base_rmse

        hierarchical, time_only = bootstrap_rmse_ratio(
            model_sqerr=model_sqerr,
            baseline_sqerr=baseline_sqerr,
            reps=int(args.bootstrap_reps),
            block_length=BLOCK_LENGTH,
            seed=BOOTSTRAP_SEED + model_index,
        )
        h_lo, h_hi = percentile_interval(hierarchical)
        t_lo, t_hi = percentile_interval(time_only)

        ratio_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_seeds": len(SEEDS),
                "n_timestamps": len(y),
                "block_length_hours": BLOCK_LENGTH,
                "bootstrap_reps": int(args.bootstrap_reps),
                "baseline_RMSE": base_rmse,
                "mean_model_RMSE": mean_model_rmse,
                "sd_model_RMSE": float(np.std(seed_rmses, ddof=1)),
                "RMSE_ratio": point_ratio,
                "RMSE_skill": 1.0 - point_ratio,
                "hierarchical_ratio_ci_low": h_lo,
                "hierarchical_ratio_ci_high": h_hi,
                "hierarchical_ci_contains_1": bool(h_lo <= 1.0 <= h_hi),
                "time_only_ratio_ci_low": t_lo,
                "time_only_ratio_ci_high": t_hi,
                "time_only_ci_contains_1": bool(t_lo <= 1.0 <= t_hi),
            }
        )

    ratio = pd.DataFrame(ratio_rows).sort_values("RMSE_ratio")
    ratio.to_csv(
        output_dir / "m5_rmse_ratio_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Manuscript-ready compact table.
    manuscript = ratio[
        [
            "model_label",
            "mean_model_RMSE",
            "sd_model_RMSE",
            "RMSE_ratio",
            "RMSE_skill",
            "hierarchical_ratio_ci_low",
            "hierarchical_ratio_ci_high",
            "hierarchical_ci_contains_1",
        ]
    ].copy()
    manuscript.to_csv(
        output_dir / "m5_rmse_ratio_manuscript_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "status": "COMPLETE_M5_HAC_AND_RMSE_RATIO_SENSITIVITY",
        "final_root": str(final_root),
        "seeds": SEEDS,
        "models": MODELS,
        "n_test": len(y),
        "first_target_utc": baseline_audit["first_target_utc"],
        "last_target_utc": baseline_audit["last_target_utc"],
        "baseline_RMSE": base_rmse,
        "hac_lags": HAC_LAGS,
        "hac_loss": "squared error",
        "dm_seed_aggregation": "mean squared loss across five seeds at each timestamp",
        "holm_scope": "four model comparisons independently at each HAC lag",
        "bootstrap": {
            "type": "hierarchical circular block bootstrap",
            "block_length_hours": BLOCK_LENGTH,
            "reps": int(args.bootstrap_reps),
            "seed": BOOTSTRAP_SEED,
            "seed_resampling": "five seeds sampled with replacement",
            "time_resampling": "circular contiguous blocks sampled with replacement",
            "estimand": "mean seed-specific model RMSE / persistence RMSE",
            "ci": "percentile 2.5%-97.5%",
        },
        "metric_crosscheck": metric_crosscheck,
        "baseline_audit": baseline_audit,
        "important_interpretation": (
            "A confidence interval containing 1 does not establish equivalence. "
            "It indicates that the analysis does not establish a difference in RMSE ratio "
            "at the reported confidence level."
        ),
        "limitation": (
            "The hierarchical bootstrap resamples only five observed initialization seeds. "
            "It therefore quantifies uncertainty conditional on this limited seed set and "
            "does not constitute evidence of population-level equivalence."
        ),
    }
    (output_dir / "m5_statistical_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = output_dir / "m5_statistical_sensitivity_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("M5 STATISTICAL SENSITIVITY ANALYSIS\n")
        f.write("HAC / NEWEY-WEST SENSITIVITY + HIERARCHICAL BOOTSTRAP RMSE RATIO\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"Test timestamps : {len(y):,}\n")
        f.write(f"Target period   : {baseline_audit['first_target_utc']} -> {baseline_audit['last_target_utc']}\n")
        f.write(f"Persistence RMSE: {base_rmse:.9f}\n")
        f.write(f"Seeds           : {SEEDS}\n")
        f.write(f"HAC lags        : {HAC_LAGS}\n")
        f.write(f"Bootstrap reps  : {int(args.bootstrap_reps):,}\n")
        f.write(f"Block length    : {BLOCK_LENGTH} h\n")
        f.write(f"Metric cross-check passed: {metric_crosscheck.get('passed')}\n\n")

        f.write("HAC SENSITIVITY SUMMARY\n")
        f.write(hac_summary.to_string(index=False))
        f.write("\n\nRMSE RATIO BOOTSTRAP\n")
        f.write(ratio.to_string(index=False))
        f.write("\n\nINTERPRETATION RULE\n")
        f.write(
            "Do not interpret non-significance or a ratio CI containing 1 as proof of equivalence.\n"
            "Use: 'the analysis did not establish a statistically significant difference/superiority.'\n"
        )
        f.write(
            "The hierarchical bootstrap still resamples only five observed seeds; this remains an explicit limitation.\n"
        )

    print("=" * 110)
    print("M5 HAC + RMSE RATIO SENSITIVITY - COMPLETE")
    print("=" * 110)
    print(f"Input root       : {final_root}")
    print(f"Test timestamps  : {len(y):,}")
    print(f"Persistence RMSE : {base_rmse:.9f}")
    print(f"Metric crosscheck: {metric_crosscheck}")
    print()
    print("HAC SENSITIVITY SUMMARY")
    print(hac_summary.to_string(index=False))
    print()
    print("RMSE RATIO BOOTSTRAP")
    print(
        ratio[
            [
                "model_label",
                "mean_model_RMSE",
                "RMSE_ratio",
                "hierarchical_ratio_ci_low",
                "hierarchical_ratio_ci_high",
                "hierarchical_ci_contains_1",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
