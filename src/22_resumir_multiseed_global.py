from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTISEED_DIR = PROJECT_ROOT / "outputs" / "multiseed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIRS = {
    11: MULTISEED_DIR / "seed_11",
    23: MULTISEED_DIR / "seed_23",
    42: MULTISEED_DIR / "seed_42_control",
    84: MULTISEED_DIR / "seed_84",
    168: MULTISEED_DIR / "seed_168",
}

MODELS = {
    "lstm": "lstm__global_2019_2026__metrics.json",
    "gru": "gru__global_2019_2026__metrics.json",
    "cnn1d": "cnn1d__global_2019_2026__metrics.json",
    "cnn_lstm": "cnn_lstm__global_2019_2026__metrics.json",
}

BASELINE_FILE = (
    SEED_DIRS[42]
    / "metrics"
    / "baseline_persistencia__global_2019_2026__metrics.json"
)

EXPECTED_FEATURE_COUNT = 22
EXPECTED_SEEDS = [11, 23, 42, 84, 168]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido:\n{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_neural_payload(payload: dict, seed: int, model: str) -> None:
    config = payload.get("config", {})
    features = payload.get("feature_columns", [])

    errors = []

    if payload.get("scenario") != "global_2019_2026":
        errors.append("scenario != global_2019_2026")

    if len(features) != EXPECTED_FEATURE_COUNT:
        errors.append(f"features={len(features)}; se esperaban 22")

    if config.get("random_seed") != seed:
        errors.append(
            f"seed registrada={config.get('random_seed')}; se esperaba {seed}"
        )

    if config.get("lookback") != 48:
        errors.append("lookback != 48")

    if config.get("horizon") != 1:
        errors.append("horizon != 1")

    if config.get("batch_size") != 64:
        errors.append("batch_size != 64")

    if config.get("epochs") != 50:
        errors.append("epochs != 50")

    if config.get("patience") != 5:
        errors.append("patience != 5")

    if abs(float(config.get("learning_rate", -1)) - 0.001) > 1e-12:
        errors.append("learning_rate != 0.001")

    if payload.get("training_target_mode") != "delta":
        errors.append("training_target_mode != delta")

    if config.get("device") != "cuda":
        errors.append("device != cuda")

    if errors:
        raise ValueError(
            f"Inconsistencia en model={model}, seed={seed}:\n- "
            + "\n- ".join(errors)
        )


def main() -> None:
    print("=" * 92)
    print("RESUMEN MULTI-SEED GLOBAL Q1")
    print("=" * 92)

    baseline = load_json(BASELINE_FILE)
    baseline_test = baseline["test"]

    baseline_rmse = float(baseline_test["RMSE"])
    baseline_mae = float(baseline_test["MAE"])
    baseline_mse = float(baseline_test["MSE"])
    baseline_mape = float(baseline_test["MAPE"])
    baseline_r2 = float(baseline_test["R2"])

    print(f"Persistence RMSE de referencia: {baseline_rmse:.6f}")
    print()

    detail_rows = []

    for seed in EXPECTED_SEEDS:
        seed_dir = SEED_DIRS[seed]

        if not seed_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta esperada para seed {seed}:\n{seed_dir}"
            )

        for model, filename in MODELS.items():
            path = seed_dir / "metrics" / filename
            payload = load_json(path)

            validate_neural_payload(
                payload=payload,
                seed=seed,
                model=model,
            )

            test = payload["test"]

            rmse = float(test["RMSE"])
            mae = float(test["MAE"])
            mse = float(test["MSE"])
            mape = float(test["MAPE"])
            r2 = float(test["R2"])

            detail_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "best_epoch": payload.get("best_epoch"),
                    "test_MAE": mae,
                    "test_RMSE": rmse,
                    "test_MSE": mse,
                    "test_MAPE": mape,
                    "test_R2": r2,
                    "delta_MAE_vs_persistence": mae - baseline_mae,
                    "delta_RMSE_vs_persistence": rmse - baseline_rmse,
                    "beats_persistence_MAE": mae < baseline_mae,
                    "beats_persistence_RMSE": rmse < baseline_rmse,
                }
            )

    detail = pd.DataFrame(detail_rows).sort_values(
        ["model", "seed"]
    ).reset_index(drop=True)

    summary = (
        detail.groupby("model", as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_MAE=("test_MAE", "mean"),
            sd_MAE=("test_MAE", lambda x: x.std(ddof=1)),
            min_MAE=("test_MAE", "min"),
            max_MAE=("test_MAE", "max"),
            mean_RMSE=("test_RMSE", "mean"),
            sd_RMSE=("test_RMSE", lambda x: x.std(ddof=1)),
            min_RMSE=("test_RMSE", "min"),
            max_RMSE=("test_RMSE", "max"),
            mean_MAPE=("test_MAPE", "mean"),
            sd_MAPE=("test_MAPE", lambda x: x.std(ddof=1)),
            mean_R2=("test_R2", "mean"),
            sd_R2=("test_R2", lambda x: x.std(ddof=1)),
            mean_delta_MAE_vs_persistence=(
                "delta_MAE_vs_persistence",
                "mean",
            ),
            sd_delta_MAE_vs_persistence=(
                "delta_MAE_vs_persistence",
                lambda x: x.std(ddof=1),
            ),
            mean_delta_RMSE_vs_persistence=(
                "delta_RMSE_vs_persistence",
                "mean",
            ),
            sd_delta_RMSE_vs_persistence=(
                "delta_RMSE_vs_persistence",
                lambda x: x.std(ddof=1),
            ),
            wins_MAE_vs_persistence=(
                "beats_persistence_MAE",
                "sum",
            ),
            wins_RMSE_vs_persistence=(
                "beats_persistence_RMSE",
                "sum",
            ),
            mean_best_epoch=("best_epoch", "mean"),
            sd_best_epoch=("best_epoch", lambda x: x.std(ddof=1)),
        )
    )

    summary["rank_mean_RMSE"] = (
        summary["mean_RMSE"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    summary = summary.sort_values(
        ["rank_mean_RMSE", "model"]
    ).reset_index(drop=True)

    detail_path = OUTPUT_DIR / "multiseed_global_detalle.csv"
    summary_path = OUTPUT_DIR / "multiseed_global_resumen.csv"

    detail.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 92)
    print("RESULTADOS POR SEMILLA")
    print("=" * 92)
    print(
        detail[
            [
                "seed",
                "model",
                "best_epoch",
                "test_MAE",
                "test_RMSE",
                "delta_RMSE_vs_persistence",
                "beats_persistence_RMSE",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 92)
    print("RESUMEN DE 5 SEMILLAS")
    print("=" * 92)
    print(
        summary[
            [
                "rank_mean_RMSE",
                "model",
                "n_seeds",
                "mean_RMSE",
                "sd_RMSE",
                "min_RMSE",
                "max_RMSE",
                "mean_delta_RMSE_vs_persistence",
                "wins_RMSE_vs_persistence",
                "mean_MAE",
                "sd_MAE",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 92)
    print("INTERPRETACION AUTOMATICA BASICA")
    print("=" * 92)

    for _, row in summary.iterrows():
        relation = (
            "menor"
            if row["mean_delta_RMSE_vs_persistence"] < 0
            else "mayor"
        )

        print(
            f"{row['model']}: RMSE medio={row['mean_RMSE']:.6f} "
            f"± {row['sd_RMSE']:.6f}; "
            f"delta medio vs persistence="
            f"{row['mean_delta_RMSE_vs_persistence']:+.6f} USDT "
            f"({relation} que persistence); "
            f"victorias RMSE={int(row['wins_RMSE_vs_persistence'])}/5."
        )

    print()
    print("IMPORTANTE:")
    print(
        "Este resumen describe variabilidad entre seeds, pero NO prueba significancia estadistica."
    )
    print(
        "El siguiente paso sera Diebold-Mariano + block bootstrap sobre predicciones alineadas."
    )

    print()
    print("Archivos generados:")
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
