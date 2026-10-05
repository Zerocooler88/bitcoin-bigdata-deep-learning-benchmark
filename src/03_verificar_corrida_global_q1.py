from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FEATURES = [
    "anio",
    "mes",
    "dia",
    "hora",
    "trimestre",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "rango_hl",
    "cuerpo_oc",
    "return_1h",
    "log_return_1h",
    "sma_24h",
    "sma_168h",
    "volatilidad_24h",
    "volumen_sma_24h",
]

FILES = {
    "baseline_persistencia": METRICS_DIR / "baseline_persistencia__global_2019_2026__metrics.json",
    "lstm": METRICS_DIR / "lstm__global_2019_2026__metrics.json",
    "gru": METRICS_DIR / "gru__global_2019_2026__metrics.json",
    "cnn1d": METRICS_DIR / "cnn1d__global_2019_2026__metrics.json",
    "cnn_lstm": METRICS_DIR / "cnn_lstm__global_2019_2026__metrics.json",
}


def almost_equal(a, b, tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def main() -> None:
    print("=" * 88)
    print("AUDITORIA DE LA CORRIDA GLOBAL Q1 - SEED 42")
    print("=" * 88)

    missing_files = [str(path) for path in FILES.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Faltan archivos JSON:\n" + "\n".join(missing_files)
        )

    payloads = {}
    rows = []

    for model, path in FILES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[model] = payload

        features = payload.get("feature_columns", [])
        config = payload.get("config", {})

        expected_target_mode = "price" if model == "baseline_persistencia" else "delta"

        checks = {
            "scenario": payload.get("scenario") == "global_2019_2026",
            "features_count_22": len(features) == 22,
            "features_exact_match": features == EXPECTED_FEATURES,
            "lookback_48": config.get("lookback") == 48,
            "horizon_1": config.get("horizon") == 1,
            "batch_size_64": config.get("batch_size") == 64,
            "epochs_50": config.get("epochs") == 50,
            "patience_5": config.get("patience") == 5,
            "learning_rate_0001": almost_equal(config.get("learning_rate"), 0.001),
            "seed_42": config.get("random_seed") == 42,
            "device_cuda": config.get("device") == "cuda",
            "target_mode_expected": payload.get("training_target_mode") == expected_target_mode,
        }

        failed = [name for name, ok in checks.items() if not ok]

        print()
        print(f"MODELO: {model}")
        print(f"  features: {len(features)}")
        print(f"  scenario: {payload.get('scenario')}")
        print(f"  target mode: {payload.get('training_target_mode')}")
        print(f"  device: {config.get('device')}")
        print(f"  seed: {config.get('random_seed')}")
        print(f"  best epoch: {payload.get('best_epoch')}")
        print(f"  test RMSE: {payload['test']['RMSE']:.6f}")
        print(f"  test MAE: {payload['test']['MAE']:.6f}")

        if failed:
            print("  RESULTADO: ERROR")
            for item in failed:
                print(f"    - fallo: {item}")
        else:
            print("  RESULTADO: OK")

        rows.append(
            {
                "model": model,
                "features": len(features),
                "features_exact_match": features == EXPECTED_FEATURES,
                "scenario": payload.get("scenario"),
                "training_target_mode": payload.get("training_target_mode"),
                "device": config.get("device"),
                "seed": config.get("random_seed"),
                "lookback": config.get("lookback"),
                "horizon": config.get("horizon"),
                "batch_size": config.get("batch_size"),
                "epochs": config.get("epochs"),
                "patience": config.get("patience"),
                "learning_rate": config.get("learning_rate"),
                "best_epoch": payload.get("best_epoch"),
                "test_MAE": payload["test"]["MAE"],
                "test_RMSE": payload["test"]["RMSE"],
                "test_MSE": payload["test"]["MSE"],
                "test_MAPE": payload["test"]["MAPE"],
                "test_R2": payload["test"]["R2"],
                "all_checks_ok": not failed,
            }
        )

    if not all(row["all_checks_ok"] for row in rows):
        raise RuntimeError(
            "La auditoria encontro inconsistencias. NO avances a multi-seed."
        )

    baseline_rmse = payloads["baseline_persistencia"]["test"]["RMSE"]
    baseline_mae = payloads["baseline_persistencia"]["test"]["MAE"]

    for row in rows:
        row["delta_RMSE_vs_persistence"] = row["test_RMSE"] - baseline_rmse
        row["delta_MAE_vs_persistence"] = row["test_MAE"] - baseline_mae

    df = pd.DataFrame(rows)

    output_csv = OUTPUT_DIR / "auditoria_global_seed42.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print()
    print("=" * 88)
    print("COMPARACION TEST")
    print("=" * 88)
    print(
        df[
            [
                "model",
                "test_MAE",
                "test_RMSE",
                "delta_MAE_vs_persistence",
                "delta_RMSE_vs_persistence",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 88)
    print("AUDITORIA COMPLETADA")
    print("=" * 88)
    print("OK: los 5 metodos usan el manifiesto esperado de 22 variables.")
    print("OK: lookback=48, horizon=1, batch=64, epochs=50, patience=5, lr=0.001.")
    print("OK: seed=42.")
    print("OK: los modelos neuronales usaron CUDA.")
    print(f"CSV de auditoria: {output_csv}")
    print()
    print("Esta corrida es una validacion de control. Todavia NO es el analisis multi-seed final.")


if __name__ == "__main__":
    main()
