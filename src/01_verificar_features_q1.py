from pathlib import Path
import importlib.util
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = CURRENT_DIR / "10_common_training_pipeline.py"


def cargar_pipeline():
    spec = importlib.util.spec_from_file_location(
        "pipeline_q1",
        PIPELINE_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError("No se pudo cargar 10_common_training_pipeline.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_q1"] = module
    spec.loader.exec_module(module)

    return module


def main():
    pipeline = cargar_pipeline()

    print("=" * 70)
    print("VERIFICACION DE FEATURES Q1")
    print("=" * 70)

    features = pipeline.CANONICAL_FEATURES_22

    print(f"Cantidad de features: {len(features)}")
    print()

    for numero, feature in enumerate(features, start=1):
        print(f"{numero:02d}. {feature}")

    prohibidas = {
        "filas_en_hora",
        "hora_completa",
        "target_close",
        "year_quarter",
        "regimen_mercado",
    }

    encontradas = prohibidas.intersection(features)

    print()
    print("=" * 70)

    if len(features) != 22:
        raise ValueError(f"ERROR: se esperaban 22 features y existen {len(features)}.")

    if encontradas:
        raise ValueError(
            f"ERROR: existen variables no permitidas: {sorted(encontradas)}"
        )

    print("OK: existen exactamente 22 variables predictoras.")
    print("OK: no aparece filas_en_hora.")
    print("OK: no aparece hora_completa.")
    print("OK: no aparece target_close.")
    print("OK: manifiesto canonico listo para el benchmark Q1.")
    print("=" * 70)


if __name__ == "__main__":
    main()
