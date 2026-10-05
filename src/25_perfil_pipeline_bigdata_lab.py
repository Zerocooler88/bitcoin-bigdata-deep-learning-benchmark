from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
import time
import hashlib
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

LAB_SCRIPT = (
    SRC_DIR
    / "18b_exportar_gold_lab_desde_spark_windows.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "performance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPEATS = 3


def load_lab_module():
    if not LAB_SCRIPT.exists():
        raise FileNotFoundError(
            f"No existe:\n{LAB_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "gold_lab_export_module",
        LAB_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"No se pudo cargar:\n{LAB_SCRIPT}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "gold_lab_export_module"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def main() -> None:
    print(
        "=" * 100
    )
    print(
        "PERFIL BIG DATA - SPARK + MINIO LAB"
    )
    print(
        "=" * 100
    )

    lab = load_lab_module()

    endpoint = (
        lab.construir_endpoint_minio()
    )

    print(
        f"Spark master : {lab.SPARK_MASTER_URL}"
    )
    print(
        f"MinIO endpoint: {endpoint}"
    )
    print(
        f"Bucket       : {lab.MINIO_BUCKET_GOLD}"
    )
    print(
        f"Objeto       : {lab.MINIO_OBJECT_GOLD}"
    )
    print(
        f"Destino LAB  : {lab.RUTA_GOLD_WINDOWS}"
    )
    print(
        f"Repeticiones : {REPEATS}"
    )
    print()

    # Healthcheck fuera de la medicion principal.
    lab.verificar_minio_desde_windows(
        endpoint
    )

    rows = []

    expected_rows = None
    expected_sha256 = None

    for run in range(
        1,
        REPEATS + 1,
    ):
        print()
        print(
            "-" * 100
        )
        print(
            f"REPETICION {run}/{REPEATS}"
        )
        print(
            "-" * 100
        )

        total_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # 1. Spark session startup
        # ----------------------------------------------------

        spark_start = (
            time.perf_counter()
        )

        spark = (
            lab.crear_spark_session(
                endpoint
            )
        )

        spark_startup_seconds = (
            time.perf_counter()
            - spark_start
        )

        try:
            spark_version = (
                spark.version
            )

            # ------------------------------------------------
            # 2. Spark read + real action count()
            # ------------------------------------------------

            spark_read_start = (
                time.perf_counter()
            )

            df = (
                spark.read.parquet(
                    lab.RUTA_GOLD_S3A
                )
                .orderBy(
                    "open_datetime_utc"
                )
            )

            total_rows = int(
                df.count()
            )

            spark_read_count_seconds = (
                time.perf_counter()
                - spark_read_start
            )

            spark_rows_per_second = (
                total_rows
                / spark_read_count_seconds
            )

            print(
                f"Spark rows         : {total_rows:,}"
            )
            print(
                f"Spark startup      : {spark_startup_seconds:.4f} s"
            )
            print(
                f"Spark read + count : {spark_read_count_seconds:.4f} s"
            )
            print(
                f"Spark throughput   : {spark_rows_per_second:,.2f} rows/s"
            )

            # ------------------------------------------------
            # 3. MinIO download
            # ------------------------------------------------

            minio_start = (
                time.perf_counter()
            )

            lab.descargar_part_desde_minio(
                endpoint
            )

            minio_download_seconds = (
                time.perf_counter()
                - minio_start
            )

            output_path = Path(
                lab.RUTA_GOLD_WINDOWS
            )

            if not output_path.exists():
                raise FileNotFoundError(
                    f"No se genero:\n{output_path}"
                )

            output_bytes = int(
                output_path.stat().st_size
            )

            output_mb = (
                output_bytes
                / (
                    1024 ** 2
                )
            )

            minio_mb_per_second = (
                output_mb
                / minio_download_seconds
            )

            output_sha256 = (
                sha256_file(
                    output_path
                )
            )

            total_seconds = (
                time.perf_counter()
                - total_start
            )

            total_rows_per_second = (
                total_rows
                / total_seconds
            )

            print(
                f"MinIO download     : {minio_download_seconds:.4f} s"
            )
            print(
                f"Output size        : {output_mb:.3f} MB"
            )
            print(
                f"MinIO throughput   : {minio_mb_per_second:.3f} MB/s"
            )
            print(
                f"Total measured     : {total_seconds:.4f} s"
            )
            print(
                f"Total throughput   : {total_rows_per_second:,.2f} rows/s"
            )

            if expected_rows is None:
                expected_rows = (
                    total_rows
                )
            elif total_rows != expected_rows:
                raise RuntimeError(
                    "La cantidad de filas cambio entre repeticiones."
                )

            if expected_sha256 is None:
                expected_sha256 = (
                    output_sha256
                )
            elif (
                output_sha256
                != expected_sha256
            ):
                raise RuntimeError(
                    "El SHA-256 del parquet LAB cambio entre repeticiones."
                )

            rows.append(
                {
                    "run": run,
                    "timestamp_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "spark_version": (
                        spark_version
                    ),
                    "spark_master": (
                        lab.SPARK_MASTER_URL
                    ),
                    "minio_endpoint": (
                        endpoint
                    ),
                    "minio_bucket": (
                        lab.MINIO_BUCKET_GOLD
                    ),
                    "rows": (
                        total_rows
                    ),
                    "spark_startup_seconds": (
                        spark_startup_seconds
                    ),
                    "spark_read_count_seconds": (
                        spark_read_count_seconds
                    ),
                    "spark_rows_per_second": (
                        spark_rows_per_second
                    ),
                    "minio_download_seconds": (
                        minio_download_seconds
                    ),
                    "output_bytes": (
                        output_bytes
                    ),
                    "output_mb": (
                        output_mb
                    ),
                    "minio_mb_per_second": (
                        minio_mb_per_second
                    ),
                    "total_measured_seconds": (
                        total_seconds
                    ),
                    "total_rows_per_second": (
                        total_rows_per_second
                    ),
                    "output_sha256": (
                        output_sha256
                    ),
                }
            )

        finally:
            spark.stop()

    detail = pd.DataFrame(
        rows
    )

    numeric_metrics = [
        "spark_startup_seconds",
        "spark_read_count_seconds",
        "spark_rows_per_second",
        "minio_download_seconds",
        "output_mb",
        "minio_mb_per_second",
        "total_measured_seconds",
        "total_rows_per_second",
    ]

    summary_rows = []

    for metric in numeric_metrics:
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(
                    detail[
                        metric
                    ].mean()
                ),
                "sd": float(
                    detail[
                        metric
                    ].std(
                        ddof=1
                    )
                ),
                "min": float(
                    detail[
                        metric
                    ].min()
                ),
                "max": float(
                    detail[
                        metric
                    ].max()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    detail_path = (
        OUTPUT_DIR
        / "bigdata_pipeline_lab_detail.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "bigdata_pipeline_lab_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "bigdata_pipeline_lab_summary.json"
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

    json_payload = {
        "repeats": REPEATS,
        "rows": int(
            expected_rows
        ),
        "output_sha256": (
            expected_sha256
        ),
        "spark_version": (
            str(
                detail.iloc[
                    0
                ][
                    "spark_version"
                ]
            )
        ),
        "spark_master": (
            str(
                detail.iloc[
                    0
                ][
                    "spark_master"
                ]
            )
        ),
        "minio_endpoint": (
            str(
                detail.iloc[
                    0
                ][
                    "minio_endpoint"
                ]
            )
        ),
        "minio_bucket": (
            str(
                detail.iloc[
                    0
                ][
                    "minio_bucket"
                ]
            )
        ),
        "metrics": {
            row[
                "metric"
            ]: {
                "mean": float(
                    row[
                        "mean"
                    ]
                ),
                "sd": float(
                    row[
                        "sd"
                    ]
                ),
                "min": float(
                    row[
                        "min"
                    ]
                ),
                "max": float(
                    row[
                        "max"
                    ]
                ),
            }
            for _, row in summary.iterrows()
        },
    }

    json_path.write_text(
        json.dumps(
            json_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )
    print(
        "RESUMEN BIG DATA LAB"
    )
    print(
        "=" * 100
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Filas consistentes: {expected_rows:,}"
    )
    print(
        f"SHA-256 consistente: {expected_sha256}"
    )

    print()
    print(
        "Archivos generados:"
    )
    print(
        detail_path
    )
    print(
        summary_path
    )
    print(
        json_path
    )

    print()
    print(
        "IMPORTANTE:"
    )
    print(
        "Este benchmark mide el flujo LAB Spark/MinIO existente; no demuestra escalabilidad horizontal."
    )
    print(
        "Debe reportarse como performance/throughput del pipeline bajo la configuracion de laboratorio utilizada."
    )


if __name__ == "__main__":
    main()
