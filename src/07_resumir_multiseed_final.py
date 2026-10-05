from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "multiseed_final_deterministic"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reproducibility"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [11, 23, 42, 84, 168]

MODELS = {
    "lstm": "lstm__global_2019_2026__metrics.json",
    "gru": "gru__global_2019_2026__metrics.json",
    "cnn1d": "cnn1d__global_2019_2026__metrics.json",
    "cnn_lstm": "cnn_lstm__global_2019_2026__metrics.json",
}

BASELINE_PATH = (
    FINAL_ROOT
    / "baseline_persistencia"
    / "metrics"
    / "baseline_persistencia__global_2019_2026__metrics.json"
)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe:\n{path}"
        )
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    print("=" * 94)
    print("RESUMEN MULTI-SEED FINAL DETERMINISTA")
    print("=" * 94)

    baseline = read_json(
        BASELINE_PATH
    )

    baseline_rmse = float(
        baseline["test"]["RMSE"]
    )

    baseline_mae = float(
        baseline["test"]["MAE"]
    )

    rows = []

    for seed in SEEDS:
        seed_dir = (
            FINAL_ROOT
            / f"seed_{seed}"
            / "metrics"
        )

        for model, filename in MODELS.items():
            payload = read_json(
                seed_dir / filename
            )

            features = payload.get(
                "feature_columns",
                [],
            )

            config = payload.get(
                "config",
                {},
            )

            if len(features) != 22:
                raise ValueError(
                    f"{model} seed={seed}: features != 22"
                )

            if config.get(
                "random_seed"
            ) != seed:
                raise ValueError(
                    f"{model}: seed registrada incorrectamente."
                )

            if config.get(
                "device"
            ) != "cuda":
                raise ValueError(
                    f"{model} seed={seed}: no uso CUDA."
                )

            test = payload["test"]

            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "best_epoch": payload.get(
                        "best_epoch"
                    ),
                    "test_MAE": float(
                        test["MAE"]
                    ),
                    "test_RMSE": float(
                        test["RMSE"]
                    ),
                    "test_MSE": float(
                        test["MSE"]
                    ),
                    "test_MAPE": float(
                        test["MAPE"]
                    ),
                    "test_R2": float(
                        test["R2"]
                    ),
                    "delta_MAE_vs_persistence": (
                        float(
                            test["MAE"]
                        )
                        - baseline_mae
                    ),
                    "delta_RMSE_vs_persistence": (
                        float(
                            test["RMSE"]
                        )
                        - baseline_rmse
                    ),
                    "beats_persistence_RMSE": (
                        float(
                            test["RMSE"]
                        )
                        < baseline_rmse
                    ),
                }
            )

    detail = pd.DataFrame(
        rows
    )

    summary = (
        detail.groupby(
            "model",
            as_index=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "count",
            ),
            mean_RMSE=(
                "test_RMSE",
                "mean",
            ),
            sd_RMSE=(
                "test_RMSE",
                "std",
            ),
            min_RMSE=(
                "test_RMSE",
                "min",
            ),
            max_RMSE=(
                "test_RMSE",
                "max",
            ),
            mean_MAE=(
                "test_MAE",
                "mean",
            ),
            sd_MAE=(
                "test_MAE",
                "std",
            ),
            mean_delta_RMSE_vs_persistence=(
                "delta_RMSE_vs_persistence",
                "mean",
            ),
            wins_RMSE_vs_persistence=(
                "beats_persistence_RMSE",
                "sum",
            ),
            mean_best_epoch=(
                "best_epoch",
                "mean",
            ),
        )
    )

    summary[
        "rank_mean_RMSE"
    ] = (
        summary[
            "mean_RMSE"
        ]
        .rank(
            ascending=True,
            method="min",
        )
        .astype(int)
    )

    summary = (
        summary.sort_values(
            [
                "rank_mean_RMSE",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    detail_path = (
        OUTPUT_DIR
        / "multiseed_final_global_detalle.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "multiseed_final_global_resumen.csv"
    )

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

    print()
    print(
        f"Persistence RMSE: {baseline_rmse:.6f}"
    )

    print()
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
        ].to_string(
            index=False
        )
    )

    print()
    print("Archivos:")
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
