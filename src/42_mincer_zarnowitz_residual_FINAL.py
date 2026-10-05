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

HAC_LAGS = [0, 1, 6, 10, 12, 24, 48, 72, 168]
PRIMARY_HAC_LAG = 10
EXPECTED_N = 9358
METRIC_TOLERANCE = 1e-3
Z_975 = 1.959963984540054


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_prediction_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {
        "fecha_hora_utc",
        "valor_real",
        "valor_predicho",
        "valor_baseline",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}")

    df["fecha_hora_utc"] = pd.to_datetime(
        df["fecha_hora_utc"],
        utc=True,
        errors="raise",
    )
    return df.sort_values("fecha_hora_utc").reset_index(drop=True)


def assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    if len(candidate) != len(reference):
        raise ValueError(f"{label}: n={len(candidate)} != {len(reference)}")

    if not reference["fecha_hora_utc"].equals(candidate["fecha_hora_utc"]):
        raise ValueError(f"{label}: timestamps distintos")

    for col in ["valor_real", "valor_baseline"]:
        a = reference[col].to_numpy(dtype=np.float64)
        b = candidate[col].to_numpy(dtype=np.float64)
        if not np.allclose(a, b, rtol=0.0, atol=1e-9):
            raise ValueError(f"{label}: {col} no coincide")


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def hac_ols_mincer_zarnowitz(
    realized_change: np.ndarray,
    predicted_correction: np.ndarray,
    hac_lag: int,
) -> dict:
    """
    Mincer-Zarnowitz calibration regression in the residual-correction domain:

        Delta P_{t+1} = alpha + beta * delta_hat_{t+1} + epsilon_{t+1}

    where:
        Delta P_{t+1}   = P_{t+1} - P_t
        delta_hat       = P_hat_{t+1} - P_t

    HAC covariance uses Bartlett/Newey-West weights.
    """
    y = np.asarray(realized_change, dtype=np.float64).reshape(-1)
    x = np.asarray(predicted_correction, dtype=np.float64).reshape(-1)

    if y.shape != x.shape:
        raise ValueError("y/x shapes incompatibles")
    if len(y) < 10:
        raise ValueError("Muestra insuficiente")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("NaN/Inf en MZ inputs")
    if np.std(x, ddof=1) <= 1e-14:
        raise ValueError(
            "La corrección pronosticada es prácticamente constante; "
            "Mincer-Zarnowitz no es identificable."
        )

    n = len(y)
    X = np.column_stack([np.ones(n, dtype=np.float64), x])

    xtx = X.T @ X
    inv_xtx = np.linalg.inv(xtx)
    coef = inv_xtx @ (X.T @ y)
    resid = y - X @ coef

    # Newey-West/Bartlett HAC meat on score vectors z_t = x_t * residual_t.
    z = X * resid[:, None]
    S = z.T @ z

    L = min(int(hac_lag), n - 1)
    for lag in range(1, L + 1):
        weight = 1.0 - lag / (L + 1.0)
        G = z[lag:].T @ z[:-lag]
        S += weight * (G + G.T)

    cov = inv_xtx @ S @ inv_xtx
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    alpha = float(coef[0])
    beta = float(coef[1])
    se_alpha = float(se[0])
    se_beta = float(se[1])

    z_alpha0 = alpha / se_alpha if se_alpha > 0 else float("nan")
    z_beta1 = (beta - 1.0) / se_beta if se_beta > 0 else float("nan")
    z_beta0 = beta / se_beta if se_beta > 0 else float("nan")

    p_alpha0 = normal_two_sided_p(z_alpha0) if np.isfinite(z_alpha0) else float("nan")
    p_beta1 = normal_two_sided_p(z_beta1) if np.isfinite(z_beta1) else float("nan")
    p_beta0 = normal_two_sided_p(z_beta0) if np.isfinite(z_beta0) else float("nan")

    # Joint Wald H0: alpha = 0 and beta = 1.
    q = np.array([alpha, beta - 1.0], dtype=np.float64)
    try:
        W = float(q.T @ np.linalg.inv(cov) @ q)
        # chi-square survival for df=2 is exactly exp(-W/2).
        p_joint = float(math.exp(-0.5 * max(W, 0.0)))
    except np.linalg.LinAlgError:
        W = float("nan")
        p_joint = float("nan")

    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum(resid ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")

    return {
        "n": n,
        "hac_lag": L,
        "alpha": alpha,
        "alpha_se_hac": se_alpha,
        "alpha_ci_low": alpha - Z_975 * se_alpha,
        "alpha_ci_high": alpha + Z_975 * se_alpha,
        "z_alpha_eq_0": z_alpha0,
        "p_alpha_eq_0": p_alpha0,
        "beta": beta,
        "beta_se_hac": se_beta,
        "beta_ci_low": beta - Z_975 * se_beta,
        "beta_ci_high": beta + Z_975 * se_beta,
        "z_beta_eq_1": z_beta1,
        "p_beta_eq_1": p_beta1,
        "z_beta_eq_0": z_beta0,
        "p_beta_eq_0": p_beta0,
        "wald_alpha0_beta1": W,
        "p_wald_alpha0_beta1": p_joint,
        "R2": r2,
        "mean_realized_change": float(np.mean(y)),
        "sd_realized_change": float(np.std(y, ddof=1)),
        "mean_predicted_correction": float(np.mean(x)),
        "sd_predicted_correction": float(np.std(x, ddof=1)),
        "mean_abs_predicted_correction": float(np.mean(np.abs(x))),
        "corr_predicted_vs_realized": (
            float(np.corrcoef(x, y)[0, 1])
            if np.std(x) > 0 and np.std(y) > 0
            else float("nan")
        ),
    }


def load_inputs(final_root: Path):
    baseline_path = (
        final_root
        / "baseline_persistencia"
        / "predictions"
        / "baseline_persistencia__global_2019_2026__test.csv"
    )
    reference = read_prediction_file(baseline_path)

    if len(reference) != EXPECTED_N:
        raise RuntimeError(
            f"n_test={len(reference)}; se esperaba {EXPECTED_N}"
        )

    y_price = reference["valor_real"].to_numpy(dtype=np.float64)
    baseline_price = reference["valor_baseline"].to_numpy(dtype=np.float64)
    realized_change = y_price - baseline_price

    matrices = {}
    audit_rows = []

    for model in MODELS:
        model_preds = []
        for seed in SEEDS:
            path = (
                final_root
                / f"seed_{seed}"
                / "predictions"
                / f"{model}__global_2019_2026__test.csv"
            )
            df = read_prediction_file(path)
            assert_aligned(reference, df, f"{model} seed={seed}")

            pred_price = df["valor_predicho"].to_numpy(dtype=np.float64)
            predicted_correction = pred_price - baseline_price
            model_preds.append(predicted_correction)

            audit_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "n": len(df),
                    "price_RMSE": rmse(y_price, pred_price),
                    "mean_predicted_correction": float(np.mean(predicted_correction)),
                    "sd_predicted_correction": float(np.std(predicted_correction, ddof=1)),
                    "mean_abs_predicted_correction": float(np.mean(np.abs(predicted_correction))),
                    "sha256": sha256_file(path),
                    "path": str(path),
                }
            )

        matrices[model] = np.vstack(model_preds)

    baseline_audit = {
        "n": int(len(reference)),
        "first_target_utc": str(reference["fecha_hora_utc"].iloc[0]),
        "last_target_utc": str(reference["fecha_hora_utc"].iloc[-1]),
        "baseline_path": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "persistence_price_RMSE": rmse(y_price, baseline_price),
        "realized_change_mean": float(np.mean(realized_change)),
        "realized_change_sd": float(np.std(realized_change, ddof=1)),
    }

    return (
        reference,
        y_price,
        baseline_price,
        realized_change,
        matrices,
        pd.DataFrame(audit_rows),
        baseline_audit,
    )


def crosscheck_metrics(
    audit: pd.DataFrame,
    expected_detail: Path,
) -> dict:
    if not expected_detail.exists():
        raise FileNotFoundError(
            "No existe el resumen multiseed definitivo:\n"
            f"{expected_detail}"
        )

    expected = pd.read_csv(expected_detail)
    required = {"seed", "model", "test_RMSE"}
    missing = required - set(expected.columns)
    if missing:
        raise ValueError(
            f"{expected_detail.name}: faltan {sorted(missing)}"
        )

    merged = audit.merge(
        expected[["seed", "model", "test_RMSE"]],
        on=["seed", "model"],
        how="left",
        validate="one_to_one",
    )
    if merged["test_RMSE"].isna().any():
        raise RuntimeError("Faltan pares seed/model en el resumen definitivo")

    merged["abs_rmse_difference"] = np.abs(
        merged["price_RMSE"] - merged["test_RMSE"]
    )
    max_diff = float(merged["abs_rmse_difference"].max())

    if max_diff > METRIC_TOLERANCE:
        bad = merged.sort_values(
            "abs_rmse_difference", ascending=False
        ).head(10)
        raise RuntimeError(
            "Los CSV de predicciones NO corresponden a la generación "
            "multiseed definitiva.\n"
            f"Max |RMSE_csv - RMSE_definitivo|={max_diff:.9f} "
            f"> {METRIC_TOLERANCE}.\n\n"
            + bad[
                [
                    "seed",
                    "model",
                    "price_RMSE",
                    "test_RMSE",
                    "abs_rmse_difference",
                ]
            ].to_string(index=False)
        )

    return {
        "performed": True,
        "expected_detail_path": str(expected_detail),
        "max_abs_rmse_difference": max_diff,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--final-root",
        type=Path,
        default=project_root / "outputs" / "multiseed_final_deterministic",
    )
    parser.add_argument(
        "--expected-detail",
        type=Path,
        default=project_root / "outputs" / "reproducibility" / "multiseed_final_global_detalle.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "mincer_zarnowitz_residual",
    )
    args = parser.parse_args()

    final_root = args.final_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        reference,
        y_price,
        baseline_price,
        realized_change,
        matrices,
        audit,
        baseline_audit,
    ) = load_inputs(final_root)

    crosscheck = crosscheck_metrics(
        audit,
        args.expected_detail.resolve(),
    )

    audit.to_csv(
        output_dir / "mz_residual_input_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # A) Full bandwidth sensitivity: ensemble and every individual seed
    # --------------------------------------------------------------
    ensemble_rows = []
    seed_rows = []

    for model in MODELS:
        ensemble_correction = matrices[model].mean(axis=0)

        for lag in HAC_LAGS:
            result = hac_ols_mincer_zarnowitz(
                realized_change,
                ensemble_correction,
                hac_lag=lag,
            )
            ensemble_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "forecast_type": "five_seed_ensemble_mean",
                    **result,
                }
            )

        for seed_index, seed in enumerate(SEEDS):
            correction = matrices[model][seed_index]
            for lag in HAC_LAGS:
                result = hac_ols_mincer_zarnowitz(
                    realized_change,
                    correction,
                    hac_lag=lag,
                )
                seed_rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "forecast_type": "individual_seed",
                        **result,
                    }
                )

    ensemble_full = pd.DataFrame(ensemble_rows)
    seed_full = pd.DataFrame(seed_rows)

    ensemble_full.to_csv(
        output_dir / "mz_residual_ensemble_hac_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_full.to_csv(
        output_dir / "mz_residual_seed_level_hac_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # B) Primary lag = 10 tables
    # --------------------------------------------------------------
    ensemble_primary = (
        ensemble_full[
            ensemble_full["hac_lag"] == PRIMARY_HAC_LAG
        ]
        .copy()
        .sort_values("model_label")
        .reset_index(drop=True)
    )

    seed_primary = (
        seed_full[
            seed_full["hac_lag"] == PRIMARY_HAC_LAG
        ]
        .copy()
        .sort_values(["model_label", "seed"])
        .reset_index(drop=True)
    )

    ensemble_primary.to_csv(
        output_dir / "mz_residual_ensemble_primary_lag10.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_primary.to_csv(
        output_dir / "mz_residual_seed_level_primary_lag10.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # C) Sensitivity summary for manuscript
    # --------------------------------------------------------------
    summary_rows = []
    for model in MODELS:
        b = ensemble_full[ensemble_full["model"] == model]
        primary = b[b["hac_lag"] == PRIMARY_HAC_LAG].iloc[0]

        summary_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "alpha_lag10": primary["alpha"],
                "alpha_ci_low_lag10": primary["alpha_ci_low"],
                "alpha_ci_high_lag10": primary["alpha_ci_high"],
                "p_alpha_eq_0_lag10": primary["p_alpha_eq_0"],
                "beta_lag10": primary["beta"],
                "beta_ci_low_lag10": primary["beta_ci_low"],
                "beta_ci_high_lag10": primary["beta_ci_high"],
                "p_beta_eq_1_lag10": primary["p_beta_eq_1"],
                "p_beta_eq_0_lag10": primary["p_beta_eq_0"],
                "joint_wald_p_lag10": primary["p_wald_alpha0_beta1"],
                "R2_lag10": primary["R2"],
                "corr_predicted_vs_realized": primary["corr_predicted_vs_realized"],
                "mean_abs_predicted_correction": primary["mean_abs_predicted_correction"],
                "min_joint_wald_p_across_lags": float(b["p_wald_alpha0_beta1"].min()),
                "max_joint_wald_p_across_lags": float(b["p_wald_alpha0_beta1"].max()),
                "joint_rejected_at_any_lag_0_05": bool(
                    (b["p_wald_alpha0_beta1"] < 0.05).any()
                ),
                "min_p_beta_eq_0_across_lags": float(b["p_beta_eq_0"].min()),
                "max_p_beta_eq_0_across_lags": float(b["p_beta_eq_0"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "mz_residual_manuscript_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # D) Machine-readable manifest and text report
    # --------------------------------------------------------------
    manifest = {
        "status": "COMPLETE_MINCER_ZARNOWITZ_RESIDUAL_BENCHMARK",
        "analysis_domain": "price residual correction",
        "regression": (
            "realized_change = alpha + beta * predicted_correction + error"
        ),
        "realized_change_definition": "valor_real - valor_baseline",
        "predicted_correction_definition": "valor_predicho - valor_baseline",
        "primary_hac_lag": PRIMARY_HAC_LAG,
        "hac_sensitivity_lags": HAC_LAGS,
        "hac_kernel": "Bartlett / Newey-West",
        "joint_null": "alpha = 0 and beta = 1",
        "individual_tests": [
            "alpha = 0",
            "beta = 1",
            "beta = 0 (predictive-content diagnostic)",
        ],
        "ensemble_definition": (
            "timestamp-wise arithmetic mean of the five seed-specific "
            "predicted residual corrections"
        ),
        "seeds": SEEDS,
        "models": MODELS,
        "n_test": int(len(reference)),
        "metric_crosscheck": crosscheck,
        "baseline_audit": baseline_audit,
        "interpretation_warning": (
            "Failure to reject alpha=0,beta=1 is not proof of calibration "
            "or equivalence, especially when predicted corrections have "
            "very small variance. Results must be interpreted jointly with "
            "beta=0, R2, correlation and correction magnitude."
        ),
    }
    (output_dir / "mz_residual_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = output_dir / "mz_residual_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("=" * 118 + "\n")
        f.write("MINCER-ZARNOWITZ - PRICE RESIDUAL CORRECTION BENCHMARK\n")
        f.write("=" * 118 + "\n")
        f.write(f"N test              : {len(reference):,}\n")
        f.write(
            f"Target period        : {baseline_audit['first_target_utc']} "
            f"-> {baseline_audit['last_target_utc']}\n"
        )
        f.write(
            f"Persistence RMSE     : "
            f"{baseline_audit['persistence_price_RMSE']:.9f}\n"
        )
        f.write(f"Seeds               : {SEEDS}\n")
        f.write(f"Primary HAC lag     : {PRIMARY_HAC_LAG}\n")
        f.write(f"HAC sensitivity     : {HAC_LAGS}\n")
        f.write(f"Metric cross-check  : {crosscheck}\n\n")

        f.write("PRIMARY FIVE-SEED ENSEMBLE RESULTS (HAC LAG 10)\n")
        cols = [
            "model_label",
            "alpha",
            "alpha_ci_low",
            "alpha_ci_high",
            "p_alpha_eq_0",
            "beta",
            "beta_ci_low",
            "beta_ci_high",
            "p_beta_eq_1",
            "p_beta_eq_0",
            "p_wald_alpha0_beta1",
            "R2",
            "corr_predicted_vs_realized",
            "mean_abs_predicted_correction",
        ]
        f.write(ensemble_primary[cols].to_string(index=False))
        f.write("\n\nBANDWIDTH SENSITIVITY SUMMARY\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nINTERPRETATION\n")
        f.write(
            "Do not interpret failure to reject the joint calibration "
            "restrictions as proof of calibration/equivalence. When the "
            "forecast correction is nearly constant, beta may be weakly "
            "identified and the joint test can have low power. Inspect "
            "beta=0, R2, correlation, and correction magnitude jointly.\n"
        )

    print("=" * 118)
    print("MINCER-ZARNOWITZ RESIDUAL BENCHMARK - COMPLETE")
    print("=" * 118)
    print(f"Input root          : {final_root}")
    print(f"N test              : {len(reference):,}")
    print(
        "Target period        : "
        f"{baseline_audit['first_target_utc']} -> "
        f"{baseline_audit['last_target_utc']}"
    )
    print(
        f"Persistence RMSE     : "
        f"{baseline_audit['persistence_price_RMSE']:.9f}"
    )
    print(f"Metric cross-check  : {crosscheck}")
    print()
    print("PRIMARY FIVE-SEED ENSEMBLE RESULTS (HAC LAG 10)")
    display_cols = [
        "model_label",
        "alpha",
        "alpha_ci_low",
        "alpha_ci_high",
        "p_alpha_eq_0",
        "beta",
        "beta_ci_low",
        "beta_ci_high",
        "p_beta_eq_1",
        "p_beta_eq_0",
        "p_wald_alpha0_beta1",
        "R2",
        "corr_predicted_vs_realized",
        "mean_abs_predicted_correction",
    ]
    print(ensemble_primary[display_cols].to_string(index=False))
    print()
    print("BANDWIDTH SENSITIVITY SUMMARY")
    print(summary.to_string(index=False))
    print()
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
