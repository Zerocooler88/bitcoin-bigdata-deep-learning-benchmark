from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

FINAL_ROOT = OUTPUTS / "multiseed_final_deterministic"

ORCHESTRATOR = SRC_DIR / "16_run_experimentos.py"

SEEDS = [11, 23, 42, 84, 168]
NEURAL_MODELS = ["lstm", "gru", "cnn1d", "cnn_lstm"]
SCENARIOS = ["global"]


def run_command(command: list[str]) -> None:
    print()
    print("=" * 96)
    print("EJECUTANDO:")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))
    print("=" * 96)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def copy_files_by_prefix(
    source_dir: Path,
    destination_dir: Path,
    prefixes: tuple[str, ...],
) -> int:
    if not source_dir.exists():
        return 0

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    for path in source_dir.iterdir():
        if (
            path.is_file()
            and path.name.startswith(prefixes)
        ):
            shutil.copy2(
                path,
                destination_dir / path.name,
            )
            count += 1

    return count


def archive_seed(seed: int) -> Path:
    seed_dir = FINAL_ROOT / f"seed_{seed}"

    if seed_dir.exists():
        raise FileExistsError(
            f"Ya existe la corrida final para seed={seed}:\n{seed_dir}\n"
            "Se detiene para evitar sobrescritura accidental."
        )

    prefixes = tuple(
        f"{model}__"
        for model in NEURAL_MODELS
    )

    n_metrics = copy_files_by_prefix(
        OUTPUTS / "metrics",
        seed_dir / "metrics",
        prefixes,
    )

    n_predictions = copy_files_by_prefix(
        OUTPUTS / "predictions",
        seed_dir / "predictions",
        prefixes,
    )

    n_history = copy_files_by_prefix(
        OUTPUTS / "history",
        seed_dir / "history",
        prefixes,
    )

    n_weights = 0

    for model in NEURAL_MODELS:
        src = MODELS_DIR / model
        dst = seed_dir / "models" / model

        if not src.exists():
            continue

        for path in src.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() == ".pt"
                and path.name.startswith(
                    f"{model}__global_2019_2026"
                )
            ):
                dst.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    path,
                    dst / path.name,
                )

                n_weights += 1

    metadata = {
        "seed": seed,
        "scenario": "global_2019_2026",
        "lookback": 48,
        "horizon": 1,
        "batch_size": 64,
        "epochs": 50,
        "patience": 5,
        "learning_rate": 0.001,
        "target_mode": "delta",
        "feature_count": 22,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "archived_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "files": {
            "metrics": n_metrics,
            "predictions": n_predictions,
            "history": n_history,
            "weights": n_weights,
        },
    }

    (
        seed_dir / "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"OK: seed {seed} archivada en {seed_dir}"
    )

    print(
        f"metrics={n_metrics}, predictions={n_predictions}, "
        f"history={n_history}, weights={n_weights}"
    )

    return seed_dir


def archive_baseline() -> Path:
    baseline_dir = (
        FINAL_ROOT / "baseline_persistencia"
    )

    if baseline_dir.exists():
        raise FileExistsError(
            f"Ya existe el baseline final:\n{baseline_dir}"
        )

    prefix = (
        "baseline_persistencia__"
    )

    n_metrics = copy_files_by_prefix(
        OUTPUTS / "metrics",
        baseline_dir / "metrics",
        (prefix,),
    )

    n_predictions = copy_files_by_prefix(
        OUTPUTS / "predictions",
        baseline_dir / "predictions",
        (prefix,),
    )

    metadata = {
        "model": "baseline_persistencia",
        "scenario": "global_2019_2026",
        "lookback": 48,
        "horizon": 1,
        "feature_count": 22,
        "deterministic": True,
        "archived_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "files": {
            "metrics": n_metrics,
            "predictions": n_predictions,
        },
    }

    (
        baseline_dir / "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"OK: persistence final archivada en {baseline_dir}"
    )

    return baseline_dir


def main() -> None:
    print("=" * 96)
    print("MULTI-SEED FINAL DETERMINISTA - GLOBAL Q1")
    print("=" * 96)
    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Seeds: {SEEDS}")
    print("Modelos: LSTM, GRU, CNN-1D, CNN-LSTM")
    print("Persistence se ejecuta una sola vez.")
    print()

    if FINAL_ROOT.exists():
        raise FileExistsError(
            f"Ya existe la carpeta final:\n{FINAL_ROOT}\n"
            "No se sobrescribe automaticamente. Esto protege la generacion final."
        )

    FINAL_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    # 1) Baseline final una sola vez.
    baseline_command = [
        sys.executable,
        str(ORCHESTRATOR),
        "--models",
        "baseline_persistencia",
        "--scenarios",
        *SCENARIOS,
        "--lookback",
        "48",
        "--horizon",
        "1",
        "--batch-size",
        "64",
        "--epochs",
        "50",
        "--patience",
        "5",
        "--learning-rate",
        "0.001",
        "--target-mode",
        "delta",
        "--num-workers",
        "0",
        "--seed",
        "42",
    ]

    run_command(
        baseline_command
    )

    archive_baseline()

    # 2) Cinco seeds neuronales.
    for seed in SEEDS:
        command = [
            sys.executable,
            str(ORCHESTRATOR),
            "--models",
            *NEURAL_MODELS,
            "--scenarios",
            *SCENARIOS,
            "--lookback",
            "48",
            "--horizon",
            "1",
            "--batch-size",
            "64",
            "--epochs",
            "50",
            "--patience",
            "5",
            "--learning-rate",
            "0.001",
            "--target-mode",
            "delta",
            "--num-workers",
            "0",
            "--seed",
            str(seed),
        ]

        run_command(
            command
        )

        archive_seed(
            seed
        )

    print()
    print("=" * 96)
    print("MULTI-SEED FINAL DETERMINISTA COMPLETADO")
    print("=" * 96)
    print(f"Resultados finales: {FINAL_ROOT}")
    print()
    print("Carpetas esperadas:")
    print(" - baseline_persistencia")
    for seed in SEEDS:
        print(f" - seed_{seed}")


if __name__ == "__main__":
    main()
