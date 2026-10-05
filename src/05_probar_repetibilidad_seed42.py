from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_ROOT = OUTPUTS / "reproducibility" / "repeatability_seed42"

MODEL_TO_SCRIPT = {
    "lstm": SRC_DIR / "12_entrenar_lstm.py",
    "gru": SRC_DIR / "13_entrenar_gru.py",
    "cnn1d": SRC_DIR / "14_entrenar_cnn1d.py",
    "cnn_lstm": SRC_DIR / "15_entrenar_cnn_lstm.py",
}

SEED = 42
SCENARIO = "global"
DISPLAY_SCENARIO = "global_2019_2026"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_model(model: str, run_number: int) -> Path:
    script = MODEL_TO_SCRIPT[model]

    command = [
        sys.executable,
        str(script),
        "--scenarios",
        SCENARIO,
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
        str(SEED),
    ]

    print()
    print("=" * 92)
    print(f"{model.upper()} | REPETICION {run_number}/2 | seed={SEED}")
    print("=" * 92)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    run_dir = TEST_ROOT / model / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=False)

    files_to_copy = [
        OUTPUTS / "predictions" / f"{model}__{DISPLAY_SCENARIO}__validation.csv",
        OUTPUTS / "predictions" / f"{model}__{DISPLAY_SCENARIO}__test.csv",
        OUTPUTS / "metrics" / f"{model}__{DISPLAY_SCENARIO}__metrics.csv",
        OUTPUTS / "metrics" / f"{model}__{DISPLAY_SCENARIO}__metrics.json",
        OUTPUTS / "history" / f"{model}__{DISPLAY_SCENARIO}__history.csv",
        MODELS_DIR / model / f"{model}__{DISPLAY_SCENARIO}__best.pt",
    ]

    for source in files_to_copy:
        if not source.exists():
            raise FileNotFoundError(
                f"No se genero el archivo esperado:\n{source}"
            )
        shutil.copy2(source, run_dir / source.name)

    return run_dir


def compare_prediction_csv(
    first: Path,
    second: Path,
) -> dict:
    df1 = pd.read_csv(first)
    df2 = pd.read_csv(second)

    if list(df1.columns) != list(df2.columns):
        raise ValueError(
            f"Las columnas difieren entre:\n{first}\n{second}"
        )

    same_shape = df1.shape == df2.shape

    numeric_columns = [
        "valor_real",
        "valor_predicho",
        "valor_baseline",
        "residual_modelo",
        "residual_baseline",
        "error_absoluto_modelo",
        "error_absoluto_baseline",
    ]

    max_abs_diff = 0.0
    numeric_exact = True

    if same_shape:
        for column in numeric_columns:
            a = df1[column].to_numpy(dtype=np.float64)
            b = df2[column].to_numpy(dtype=np.float64)

            if not np.array_equal(a, b):
                numeric_exact = False
                difference = np.max(np.abs(a - b))
                max_abs_diff = max(
                    max_abs_diff,
                    float(difference),
                )
    else:
        numeric_exact = False
        max_abs_diff = float("nan")

    timestamps_exact = (
        same_shape
        and df1["fecha_hora_utc"].equals(
            df2["fecha_hora_utc"]
        )
    )

    return {
        "same_shape": same_shape,
        "timestamps_exact": timestamps_exact,
        "numeric_exact": numeric_exact,
        "max_abs_diff": max_abs_diff,
        "sha256_equal": sha256_file(first) == sha256_file(second),
    }


def main() -> None:
    print("=" * 92)
    print("PRUEBA DE REPETIBILIDAD DETERMINISTA - GLOBAL seed=42")
    print("=" * 92)
    print("Se ejecutaran dos veces LSTM, GRU, CNN-1D y CNN-LSTM.")
    print("Los resultados de cada repeticion se archivaran antes de la siguiente.")
    print()

    if TEST_ROOT.exists():
        raise FileExistsError(
            f"Ya existe la carpeta de prueba:\n{TEST_ROOT}\n"
            "No se sobrescribe automaticamente. Renombrala o borrela solo si deseas repetir la prueba."
        )

    summary_rows = []

    for model in MODEL_TO_SCRIPT:
        run1 = run_model(
            model=model,
            run_number=1,
        )

        run2 = run_model(
            model=model,
            run_number=2,
        )

        pred1 = run1 / f"{model}__{DISPLAY_SCENARIO}__test.csv"
        pred2 = run2 / f"{model}__{DISPLAY_SCENARIO}__test.csv"

        comparison = compare_prediction_csv(
            pred1,
            pred2,
        )

        result = (
            "PASS"
            if (
                comparison["same_shape"]
                and comparison["timestamps_exact"]
                and comparison["numeric_exact"]
            )
            else "FAIL"
        )

        summary_rows.append(
            {
                "model": model,
                "seed": SEED,
                "same_shape": comparison["same_shape"],
                "timestamps_exact": comparison["timestamps_exact"],
                "numeric_exact": comparison["numeric_exact"],
                "max_abs_diff": comparison["max_abs_diff"],
                "prediction_csv_sha256_equal": comparison["sha256_equal"],
                "result": result,
            }
        )

        print()
        print(f"RESULTADO {model.upper()}: {result}")
        print(f"  misma forma           : {comparison['same_shape']}")
        print(f"  timestamps identicos  : {comparison['timestamps_exact']}")
        print(f"  valores exactos       : {comparison['numeric_exact']}")
        print(f"  max abs diff          : {comparison['max_abs_diff']}")
        print(f"  SHA256 CSV igual      : {comparison['sha256_equal']}")

    summary = pd.DataFrame(summary_rows)

    summary_path = TEST_ROOT / "repeatability_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 92)
    print("RESUMEN")
    print("=" * 92)
    print(summary.to_string(index=False))
    print()

    if (summary["result"] == "PASS").all():
        print("OK: los cuatro modelos son repetibles bajo la misma seed=42.")
        print("Ya se puede generar la serie multi-seed FINAL.")
    else:
        print("ATENCION: al menos un modelo no fue exactamente repetible.")
        print("NO ejecutes todavía la serie multi-seed final.")

    print()
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
