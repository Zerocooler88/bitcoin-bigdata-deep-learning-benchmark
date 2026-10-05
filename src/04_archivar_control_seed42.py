from __future__ import annotations

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"
DEST = OUTPUTS / "multiseed" / "seed_42_control"

NEURAL_PREFIXES = ("lstm__", "gru__", "cnn1d__", "cnn_lstm__")
BASELINE_PREFIX = "baseline_persistencia__"

def copy_matching(source_dir: Path, dest_dir: Path, prefixes: tuple[str, ...]) -> int:
    if not source_dir.exists():
        return 0
    count = 0
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(prefixes):
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / path.name)
        count += 1
    return count

def copy_model_weights() -> int:
    count = 0
    for model_name in ["lstm", "gru", "cnn1d", "cnn_lstm"]:
        src = MODELS / model_name
        dst = DEST / "models" / model_name
        if not src.exists():
            continue
        for path in src.iterdir():
            if path.is_file() and path.name.endswith(".pt"):
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst / path.name)
                count += 1
    return count

def main() -> None:
    print("=" * 76)
    print("ARCHIVANDO CORRIDA DE CONTROL GLOBAL - SEED 42")
    print("=" * 76)

    if DEST.exists():
        raise FileExistsError(
            f"Ya existe {DEST}\n"
            "No se sobrescribe automaticamente para proteger la corrida de control."
        )

    n_metrics = copy_matching(
        OUTPUTS / "metrics",
        DEST / "metrics",
        NEURAL_PREFIXES + (BASELINE_PREFIX,),
    )
    n_predictions = copy_matching(
        OUTPUTS / "predictions",
        DEST / "predictions",
        NEURAL_PREFIXES + (BASELINE_PREFIX,),
    )
    n_history = copy_matching(
        OUTPUTS / "history",
        DEST / "history",
        NEURAL_PREFIXES,
    )
    n_weights = copy_model_weights()

    # Guardar también la auditoría producida anteriormente.
    audit = OUTPUTS / "reproducibility" / "auditoria_global_seed42.csv"
    if audit.exists():
        (DEST / "reproducibility").mkdir(parents=True, exist_ok=True)
        shutil.copy2(audit, DEST / "reproducibility" / audit.name)

    print(f"Destino: {DEST}")
    print(f"Archivos metrics copiados     : {n_metrics}")
    print(f"Archivos predictions copiados : {n_predictions}")
    print(f"Archivos history copiados     : {n_history}")
    print(f"Pesos .pt copiados            : {n_weights}")
    print()
    print("OK: la corrida seed=42 quedó protegida antes de ejecutar nuevas semillas.")

if __name__ == "__main__":
    main()
