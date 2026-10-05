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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Si existe un piloto anterior, renombrarlo con timestamp y volver a ejecutar.",
    )
    args = parser.parse_args()

    lr.configure_determinism(require_cuda=True)
    dataset_manifest = lr.build_logreturn_dataset(force=False)
    common = lr.load_official_common()
    builders = lr.load_model_builders()
    catalog = lr.scenario_catalog_with_logreturn_data(common)
    scenario = catalog["global"]

    backup = lr.prepare_clean_root(
        lr.PILOT_ROOT,
        allow_backup=args.backup_existing,
    )

    summaries = []
    return_rows = []
    price_rows = []

    print("=" * 108)
    print("PILOTO LOG-RETURN - GLOBAL - SEED 42")
    print("=" * 108)
    print("Target     : log(P_t+1 / P_t)")
    print("Reconstruye: P_t * exp(predicted_log_return)")
    print("Features   : 22 scale-robust/cyclic")
    print("Ventanas   : strict 48h -> +1h del pipeline oficial")
    print("Main outputs modificados: NO")
    print(f"Workspace  : {lr.PILOT_ROOT}")
    print()

    baseline = lr.run_persistence(
        common,
        scenario=scenario,
        experiment_root=lr.PILOT_ROOT,
        seed=42,
    )
    summaries.append(
        {
            "model": "baseline_persistencia",
            "best_epoch": 0,
            "test_RMSE": baseline["test_metrics"]["RMSE"],
            "test_MAE": baseline["test_metrics"]["MAE"],
        }
    )

    baseline_pred = Path(baseline["paths"]["pred_test"])
    b_return = lr.prediction_return_diagnostics(baseline_pred)
    b_price = lr.price_metrics_from_prediction(baseline_pred)
    return_rows.append({"model": "baseline_persistencia", **b_return})
    price_rows.append({"model": "baseline_persistencia", **b_price})

    for model in lr.NEURAL_MODELS:
        result = lr.run_neural(
            common,
            builder=builders[model],
            model_name=model,
            scenario=scenario,
            experiment_root=lr.PILOT_ROOT,
            seed=42,
        )
        summaries.append(
            {
                "model": model,
                "best_epoch": result["best_epoch"],
                "test_RMSE": result["test_metrics"]["RMSE"],
                "test_MAE": result["test_metrics"]["MAE"],
            }
        )
        pred_path = Path(result["paths"]["pred_test"])
        rdiag = lr.prediction_return_diagnostics(pred_path)
        pdiag = lr.price_metrics_from_prediction(pred_path)
        return_rows.append({"model": model, **rdiag})
        price_rows.append({"model": model, **pdiag})

    summary_df = pd.DataFrame(summaries)
    return_df = pd.DataFrame(return_rows)
    price_df = pd.DataFrame(price_rows)

    # All methods must evaluate identical strict timestamps / sample count.
    if not (return_df["n"] == lr.EXPECTED_GLOBAL_TEST_N).all():
        raise RuntimeError(
            "El piloto no conserva n_test=9358 en todos los métodos. "
            f"Conteos: {return_df[['model','n']].to_dict('records')}"
        )

    status = "PASS_LOGRETURN_GLOBAL_SEED42"
    manifest = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "scenario": "global_2019_2026",
        "target_mode": "log_return",
        "target_formula": "log(P_t+1 / P_t)",
        "price_reconstruction": "P_t * exp(predicted_log_return)",
        "feature_count": 22,
        "features": lr.LOGRETURN_FEATURES_22,
        "strict_global_test_n": lr.EXPECTED_GLOBAL_TEST_N,
        "dataset_manifest": dataset_manifest,
        "backup_of_previous_pilot": str(backup) if backup else None,
    }

    out = lr.PILOT_ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out / "pilot_price_summary.csv", index=False, encoding="utf-8-sig")
    return_df.to_csv(out / "pilot_return_metrics.csv", index=False, encoding="utf-8-sig")
    price_df.to_csv(out / "pilot_relative_price_metrics.csv", index=False, encoding="utf-8-sig")
    lr.write_json(out / "pilot_manifest.json", manifest)

    print()
    print("=" * 108)
    print("PILOTO LOG-RETURN - RESULTADO FINAL")
    print("=" * 108)
    print(f"Estado      : {status}")
    print(f"Test n      : {lr.EXPECTED_GLOBAL_TEST_N:,}")
    print(f"Features    : {len(lr.LOGRETURN_FEATURES_22)}")
    print("Main benchmark overwritten: NO")
    print()
    print("PRICE DOMAIN")
    print(price_df.to_string(index=False))
    print()
    print("RETURN DOMAIN")
    print(return_df.to_string(index=False))
    print()
    print(f"Manifest: {out / 'pilot_manifest.json'}")


if __name__ == "__main__":
    main()
