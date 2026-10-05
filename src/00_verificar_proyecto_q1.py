from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER = PROJECT_ROOT / ".q1_isolated_project"

def main():
    print("=" * 78)
    print("VERIFICACION DEL PROYECTO Q1 AISLADO")
    print("=" * 78)
    print("Raiz detectada:")
    print(PROJECT_ROOT)
    print()

    if not MARKER.exists():
        print("ERROR: no se encontro .q1_isolated_project")
        print("No ejecutes experimentos hasta revisar la carpeta abierta en VS Code.")
        sys.exit(1)

    required = [
        PROJECT_ROOT / "src" / "10_common_training_pipeline.py",
        PROJECT_ROOT / "src" / "16_run_experimentos.py",
        PROJECT_ROOT / "data" / "1h" / "global",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "models",
    ]
    faltan = [str(p) for p in required if not p.exists()]

    if faltan:
        print("ERROR: faltan elementos esperados:")
        for p in faltan:
            print(" -", p)
        sys.exit(1)

    print("OK: estas trabajando dentro del proyecto Q1 aislado.")
    print("Todavia NO se ha tocado tu proyecto anterior.")

if __name__ == "__main__":
    main()
