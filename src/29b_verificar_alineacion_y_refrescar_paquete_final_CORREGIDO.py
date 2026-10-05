from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

ORIGINAL_SCRIPT = (
    SRC_DIR / "29_verificar_alineacion_y_refrescar_paquete_final.py"
)

EXPECTED_CUBLAS = ":4096:8"
EXPECTED_MODELS = ["lstm", "gru", "cnn1d", "cnn_lstm"]
SCENARIO_NAME = "global_2019_2026"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_original_module():
    if not ORIGINAL_SCRIPT.exists():
        raise FileNotFoundError(
            f"No existe el script original:\n{ORIGINAL_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "package_final_original",
        ORIGINAL_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"No se pudo cargar:\n{ORIGINAL_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["package_final_original"] = module
    spec.loader.exec_module(module)
    return module


def normalize_numeric_column(
    df: pd.DataFrame,
    column: str,
) -> np.ndarray:
    if column not in df.columns:
        raise KeyError(
            f"Falta la columna requerida {column!r}"
        )
    return pd.to_numeric(
        df[column],
        errors="raise",
    ).to_numpy(dtype=np.float64)


def rebuild_alignment(module) -> pd.DataFrame:
    """
    Reconstruye la alineación seed42 DESPUÉS de que ya existe el bloque
    multiseed final.

    Corrige el caso en que el script 28 generó
    global_seed42_vs_multiseed_archive.csv vacío porque
    multiseed_final_deterministic todavía no existía en ese momento.
    """

    scenario_final = module.SCENARIO_FINAL
    scenario_manifest = module.SCENARIO_MANIFEST
    multiseed_final = module.MULTISEED_FINAL

    alignment_path = (
        scenario_manifest
        / "global_seed42_vs_multiseed_archive.csv"
    )

    scenario_predictions = (
        scenario_final / "predictions"
    )

    seed42_predictions = (
        multiseed_final / "seed_42" / "predictions"
    )

    if not scenario_predictions.exists():
        raise FileNotFoundError(
            f"Falta carpeta de predicciones de escenarios:\n"
            f"{scenario_predictions}"
        )

    if not seed42_predictions.exists():
        raise FileNotFoundError(
            f"Falta carpeta seed_42 multiseed:\n"
            f"{seed42_predictions}"
        )

    # Preservar el CSV viejo/vacío para trazabilidad.
    if alignment_path.exists():
        stamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        backup = alignment_path.with_name(
            alignment_path.stem
            + f"__PRE_REBUILD_{stamp}"
            + alignment_path.suffix
        )

        shutil.copy2(
            alignment_path,
            backup,
        )

        print(
            f"Backup del alignment previo: {backup}"
        )

    legacy_rows: list[dict] = []
    detailed_rows: list[dict] = []

    for model in EXPECTED_MODELS:
        filename = (
            f"{model}__{SCENARIO_NAME}__test.csv"
        )

        scenario_file = (
            scenario_predictions / filename
        )

        multiseed_file = (
            seed42_predictions / filename
        )

        # Fallback conservador si la estructura cambia ligeramente.
        if not multiseed_file.exists():
            candidates = list(
                (
                    multiseed_final / "seed_42"
                ).rglob(filename)
            )

            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"{model}: se esperaba exactamente un archivo "
                    f"seed42 para {filename}, encontrados={len(candidates)}\n"
                    + "\n".join(
                        str(p)
                        for p in candidates
                    )
                )

            multiseed_file = candidates[0]

        if not scenario_file.exists():
            raise FileNotFoundError(
                f"{model}: falta predicción final de escenario:\n"
                f"{scenario_file}"
            )

        a = pd.read_csv(
            scenario_file
        )
        b = pd.read_csv(
            multiseed_file
        )

        same_shape = (
            a.shape == b.shape
        )

        same_columns = (
            list(a.columns)
            == list(b.columns)
        )

        dataframe_exact_equal = (
            same_shape
            and same_columns
            and a.equals(b)
        )

        if (
            same_shape
            and same_columns
            and "valor_predicho" in a.columns
            and "valor_predicho" in b.columns
        ):
            av = normalize_numeric_column(
                a,
                "valor_predicho",
            )
            bv = normalize_numeric_column(
                b,
                "valor_predicho",
            )

            max_abs_diff = float(
                np.max(
                    np.abs(
                        av - bv
                    )
                )
            )
        else:
            max_abs_diff = float("nan")

        sha_scenario = sha256_file(
            scenario_file
        )
        sha_multiseed = sha256_file(
            multiseed_file
        )

        byte_identical = (
            sha_scenario
            == sha_multiseed
        )

        status = (
            "match"
            if (
                same_shape
                and same_columns
                and dataframe_exact_equal
                and max_abs_diff == 0.0
            )
            else "different"
        )

        legacy_rows.append(
            {
                "model": model,
                "status": status,
                "archived_path": str(
                    multiseed_file.resolve()
                ),
                "same_shape": bool(
                    same_shape
                ),
                "max_abs_prediction_diff": (
                    max_abs_diff
                ),
            }
        )

        detailed_rows.append(
            {
                "model": model,
                "scenario_file": (
                    scenario_file
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "multiseed_seed42_file": (
                    multiseed_file
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "same_shape": bool(
                    same_shape
                ),
                "same_columns": bool(
                    same_columns
                ),
                "dataframe_exact_equal": bool(
                    dataframe_exact_equal
                ),
                "scenario_sha256": (
                    sha_scenario
                ),
                "multiseed_sha256": (
                    sha_multiseed
                ),
                "byte_identical": bool(
                    byte_identical
                ),
                "max_abs_prediction_diff": (
                    max_abs_diff
                ),
            }
        )

        print(
            f"{model:8s} | "
            f"shape={same_shape} | "
            f"columns={same_columns} | "
            f"dataframe_equal={dataframe_exact_equal} | "
            f"byte_identical={byte_identical} | "
            f"max_abs_diff={max_abs_diff:.12g}"
        )

    legacy_df = pd.DataFrame(
        legacy_rows
    )

    detailed_df = pd.DataFrame(
        detailed_rows
    )

    # Escribir SIEMPRE con cabecera, incluso si algo luego falla.
    legacy_df.to_csv(
        alignment_path,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostic_path = (
        scenario_manifest
        / "global_seed42_alignment_rebuilt_diagnostic.csv"
    )

    detailed_df.to_csv(
        diagnostic_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Alignment reconstruido: {alignment_path}"
    )
    print(
        f"Diagnóstico detallado : {diagnostic_path}"
    )

    # No permitir que el paquete se cierre con una alineación falsa.
    if set(
        legacy_df["model"]
    ) != set(
        EXPECTED_MODELS
    ):
        raise RuntimeError(
            "La reconstrucción no contiene exactamente los cuatro modelos."
        )

    if not (
        legacy_df["status"]
        .astype(str)
        .str.lower()
        .eq("match")
        .all()
    ):
        raise RuntimeError(
            "Al menos un modelo seed42 NO coincide entre "
            "13 escenarios y multiseed."
        )

    if not (
        legacy_df[
            "same_shape"
        ]
        .astype(bool)
        .all()
    ):
        raise RuntimeError(
            "Al menos un modelo tiene distinta forma."
        )

    if not (
        pd.to_numeric(
            legacy_df[
                "max_abs_prediction_diff"
            ],
            errors="raise",
        )
        .eq(0.0)
        .all()
    ):
        raise RuntimeError(
            "Al menos un modelo tiene diferencia numérica distinta de cero."
        )

    if not (
        detailed_df[
            "dataframe_exact_equal"
        ]
        .astype(bool)
        .all()
    ):
        raise RuntimeError(
            "Al menos un DataFrame no es exactamente igual."
        )

    # El manuscrito/paquete ha usado la afirmación byte-identical;
    # la exigimos explícitamente para mantener esa afirmación.
    if not (
        detailed_df[
            "byte_identical"
        ]
        .astype(bool)
        .all()
    ):
        raise RuntimeError(
            "Los DataFrames coinciden, pero al menos un CSV no es "
            "byte-identical. No se cierra el paquete hasta revisar "
            "esa discrepancia."
        )

    return detailed_df


def main() -> None:
    print(
        "=" * 108
    )
    print(
        "RECONSTRUIR ALINEACIÓN SEED42 + CERRAR PAQUETE FINAL"
    )
    print(
        "=" * 108
    )

    # No entrena nada. Solo mantiene coherente el manifest del entorno.
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] = EXPECTED_CUBLAS

    module = load_original_module()

    # Sustituir SOLO verify_alignment() en memoria.
    # El resto del script 29 original permanece intacto.
    module.verify_alignment = (
        lambda: rebuild_alignment(
            module
        )
    )

    print()
    print(
        "1/2 Reconstruyendo alineación desde los outputs actuales..."
    )
    print(
        "2/2 Si la alineación pasa, se ejecutará el cierre original del paquete."
    )
    print()

    module.main()


if __name__ == "__main__":
    main()
