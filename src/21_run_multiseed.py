from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
ORCHESTRATOR = SRC_DIR / "16_run_experimentos.py"

NEURAL_MODELS = ["lstm", "gru", "cnn1d", "cnn_lstm"]
DEFAULT_SEEDS = [11, 23, 84, 168]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta y archiva corridas multi-seed sin sobrescribir los archivos archivados."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Semillas adicionales. Seed 42 ya debe estar archivada como control.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["global"],
        help="Usa 'global' para la primera prueba multi-seed; más adelante puede usarse 'all'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permite reemplazar una carpeta seed_N ya existente. No usar salvo que sepas por qué.",
    )
    return parser.parse_args()

def copy_prefix_files(source: Path, destination: Path, prefixes: tuple[str, ...]) -> int:
    if not source.exists():
        return 0
    count = 0
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name.startswith(prefixes):
            shutil.copy2(path, destination / path.name)
            count += 1
    return count

def archive_seed(seed: int, force: bool) -> Path:
    dest = OUTPUTS / "multiseed" / f"seed_{seed}"

    if dest.exists():
        if not force:
            raise FileExistsError(
                f"Ya existe {dest}. Se detiene para no sobrescribir una corrida previa."
            )
        shutil.rmtree(dest)

    prefixes = tuple(f"{m}__" for m in NEURAL_MODELS)

    copy_prefix_files(OUTPUTS / "metrics", dest / "metrics", prefixes)
    copy_prefix_files(OUTPUTS / "predictions", dest / "predictions", prefixes)
    copy_prefix_files(OUTPUTS / "history", dest / "history", prefixes)

    for model_name in NEURAL_MODELS:
        src = MODELS_DIR / model_name
        dst = dest / "models" / model_name
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.iterdir():
            if path.is_file() and path.name.endswith(".pt"):
                shutil.copy2(path, dst / path.name)

    meta = dest / "run_metadata.txt"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        "\n".join(
            [
                f"seed={seed}",
                "lookback=48",
                "horizon=1",
                "batch_size=64",
                "epochs=50",
                "patience=5",
                "learning_rate=0.001",
                "target_mode=delta",
                f"archived_at_utc={datetime.now(timezone.utc).isoformat()}",
            ]
        ),
        encoding="utf-8",
    )

    return dest

def run_seed(seed: int, scenarios: list[str], force: bool) -> None:
    dest = OUTPUTS / "multiseed" / f"seed_{seed}"
    if dest.exists() and not force:
        raise FileExistsError(
            f"Ya existe {dest}. No se ejecuta la semilla {seed} para proteger resultados previos."
        )

    cmd = [
        sys.executable,
        str(ORCHESTRATOR),
        "--models",
        *NEURAL_MODELS,
        "--scenarios",
        *scenarios,
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

    print()
    print("=" * 90)
    print(f"EJECUTANDO SEED {seed}")
    print("=" * 90)
    print("Comando:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    archived = archive_seed(seed, force=force)
    print()
    print(f"OK: seed {seed} archivada en:")
    print(archived)

def main() -> None:
    args = parse_args()

    control = OUTPUTS / "multiseed" / "seed_42_control"
    if not control.exists():
        raise FileNotFoundError(
            "Primero debes ejecutar 04_archivar_control_seed42.py para proteger la corrida actual."
        )

    print("=" * 90)
    print("MULTI-SEED Q1")
    print("=" * 90)
    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Seeds adicionales: {args.seeds}")
    print(f"Escenarios: {args.scenarios}")
    print()
    print("IMPORTANTE: persistence no se vuelve a entrenar porque es determinista.")

    for seed in args.seeds:
        if seed == 42:
            raise ValueError(
                "Seed 42 ya está reservada como corrida de control. No la incluyas aquí."
            )
        run_seed(seed, args.scenarios, force=args.force)

    print()
    print("=" * 90)
    print("MULTI-SEED COMPLETADO")
    print("=" * 90)
    print("Corridas disponibles:")
    print(" - seed_42_control")
    for seed in args.seeds:
        print(f" - seed_{seed}")

if __name__ == "__main__":
    main()
