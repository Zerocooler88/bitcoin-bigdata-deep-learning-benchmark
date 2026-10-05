from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import logreturn_sensitivity_common as lr


SEEDS = [11, 23, 42, 84, 168]


def require_pilot_pass() -> None:
    manifest = lr.PILOT_ROOT / "analysis" / "pilot_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            "Falta el piloto aprobado. Ejecuta primero "
            "37_logreturn_sensitivity_pilot_global_seed42.py"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_LOGRETURN_GLOBAL_SEED42":
        raise RuntimeError(f"Piloto no aprobado: {payload.get('status')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Archivar una corrida full anterior y regenerar.",
    )
    args = parser.parse_args()

    require_pilot_pass()
    lr.configure_determinism(require_cuda=True)
    lr.build_logreturn_dataset(force=False)
    common = lr.load_official_common()
    builders = lr.load_model_builders()
    catalog = lr.scenario_catalog_with_logreturn_data(common)

    backup = lr.prepare_clean_root(
        lr.FULL_ROOT,
        allow_backup=args.backup_existing,
    )

    scenario_root = lr.FULL_ROOT / "scenario_seed42"
    multiseed_root = lr.FULL_ROOT / "global_multiseed"
    analysis_root = lr.FULL_ROOT / "analysis"
    scenario_root.mkdir(parents=True, exist_ok=True)
    multiseed_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    scenario_rows = []
    scenario_return_rows = []

    print("=" * 112)
    print("LOG-RETURN SENSITIVITY - 13 ESCENARIOS SEED 42")
    print("=" * 112)

    for scenario_key, scenario in catalog.items():
        print()
        print("-" * 112)
        print(f"SCENARIO: {scenario_key} | {scenario.display_name}")
        print("-" * 112)

        run_root = scenario_root / scenario_key
        run_root.mkdir(parents=True, exist_ok=False)

        baseline = lr.run_persistence(
            common,
            scenario=scenario,
            experiment_root=run_root,
            seed=42,
        )
        baseline_pred = Path(baseline["paths"]["pred_test"])
        bp = lr.price_metrics_from_prediction(baseline_pred)
        br = lr.prediction_return_diagnostics(baseline_pred)

        scenario_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario": scenario.display_name,
                "model": "baseline_persistencia",
                "seed": 42,
                **bp,
            }
        )
        scenario_return_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario": scenario.display_name,
                "model": "baseline_persistencia",
                "seed": 42,
                **br,
            }
        )

        for model in lr.NEURAL_MODELS:
            result = lr.run_neural(
                common,
                builder=builders[model],
                model_name=model,
                scenario=scenario,
                experiment_root=run_root,
                seed=42,
            )
            pred_path = Path(result["paths"]["pred_test"])
            p = lr.price_metrics_from_prediction(pred_path)
            r = lr.prediction_return_diagnostics(pred_path)

            scenario_rows.append(
                {
                    "scenario_key": scenario_key,
                    "scenario": scenario.display_name,
                    "model": model,
                    "seed": 42,
                    "best_epoch": result["best_epoch"],
                    **p,
                }
            )
            scenario_return_rows.append(
                {
                    "scenario_key": scenario_key,
                    "scenario": scenario.display_name,
                    "model": model,
                    "seed": 42,
                    "best_epoch": result["best_epoch"],
                    **r,
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_return_df = pd.DataFrame(scenario_return_rows)
    scenario_df.to_csv(
        analysis_root / "logreturn_13_scenarios_seed42_price_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_return_df.to_csv(
        analysis_root / "logreturn_13_scenarios_seed42_return_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Global multi-seed. Seed 42 is rerun in an isolated root so that all five
    # seeds have identical directory structure and independent manifests.
    multiseed_rows = []
    multiseed_return_rows = []
    global_scenario = catalog["global"]

    print()
    print("=" * 112)
    print("LOG-RETURN SENSITIVITY - GLOBAL FIVE SEEDS")
    print("=" * 112)

    # Persistence once, stored in its own deterministic reference root.
    baseline_root = multiseed_root / "baseline_persistencia"
    baseline_root.mkdir(parents=True, exist_ok=False)
    baseline = lr.run_persistence(
        common,
        scenario=global_scenario,
        experiment_root=baseline_root,
        seed=42,
    )
    base_pred = Path(baseline["paths"]["pred_test"])
    base_price = lr.price_metrics_from_prediction(base_pred)
    base_return = lr.prediction_return_diagnostics(base_pred)

    for seed in SEEDS:
        seed_root = multiseed_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        for model in lr.NEURAL_MODELS:
            result = lr.run_neural(
                common,
                builder=builders[model],
                model_name=model,
                scenario=global_scenario,
                experiment_root=seed_root,
                seed=seed,
            )
            pred_path = Path(result["paths"]["pred_test"])
            p = lr.price_metrics_from_prediction(pred_path)
            r = lr.prediction_return_diagnostics(pred_path)

            if p["n"] != lr.EXPECTED_GLOBAL_TEST_N:
                raise RuntimeError(
                    f"{model}/seed={seed}: n_test={p['n']} != {lr.EXPECTED_GLOBAL_TEST_N}"
                )

            multiseed_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "best_epoch": result["best_epoch"],
                    **p,
                }
            )
            multiseed_return_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "best_epoch": result["best_epoch"],
                    **r,
                }
            )

    multi_df = pd.DataFrame(multiseed_rows)
    multi_return_df = pd.DataFrame(multiseed_return_rows)

    multi_df.to_csv(
        analysis_root / "logreturn_global_multiseed_price_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    multi_return_df.to_csv(
        analysis_root / "logreturn_global_multiseed_return_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        multi_df.groupby("model", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_price_RMSE=("price_RMSE", "mean"),
            sd_price_RMSE=("price_RMSE", "std"),
            mean_relative_RMSE=("relative_RMSE", "mean"),
            geometric_mean_relative_RMSE=(
                "relative_RMSE",
                lambda x: float(pd.Series(x).apply(np.log).mean())
            ),
            mean_RMSE_skill=("RMSE_skill", "mean"),
            mean_price_MAE=("price_MAE", "mean"),
        )
    )
    # Convert stored mean(log ratio) to geometric mean.
    summary["geometric_mean_relative_RMSE"] = np.exp(
        summary["geometric_mean_relative_RMSE"]
    )

    ret_summary = (
        multi_return_df.groupby("model", as_index=False)
        .agg(
            mean_return_RMSE=("return_RMSE", "mean"),
            sd_return_RMSE=("return_RMSE", "std"),
            mean_R2_oos_vs_zero_return=("R2_oos_vs_zero_return", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            mean_corr_pred_realized=(
                "corr_predicted_vs_realized_log_return",
                "mean",
            ),
        )
    )

    summary = summary.merge(ret_summary, on="model", how="left")
    summary["persistence_price_RMSE"] = base_price["persistence_price_RMSE"]
    summary["persistence_return_RMSE"] = base_return["persistence_return_RMSE"]
    summary = summary.sort_values("mean_price_RMSE").reset_index(drop=True)

    summary.to_csv(
        analysis_root / "logreturn_global_multiseed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "status": "COMPLETE_LOGRETURN_SENSITIVITY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_mode": "log_return",
        "target_formula": "log(P_t+1 / P_t)",
        "price_reconstruction": "P_t * exp(predicted_log_return)",
        "feature_count": 22,
        "features": lr.LOGRETURN_FEATURES_22,
        "scenario_count": len(catalog),
        "scenario_seed": 42,
        "global_seeds": SEEDS,
        "expected_global_test_n": lr.EXPECTED_GLOBAL_TEST_N,
        "main_benchmark_modified": False,
        "backup_of_previous_full": str(backup) if backup else None,
    }
    lr.write_json(
        analysis_root / "logreturn_full_manifest.json",
        manifest,
    )

    print()
    print("=" * 112)
    print("LOG-RETURN SENSITIVITY - COMPLETE")
    print("=" * 112)
    print(summary.to_string(index=False))
    print()
    print(f"Analysis: {analysis_root}")


if __name__ == "__main__":
    import numpy as np
    main()
