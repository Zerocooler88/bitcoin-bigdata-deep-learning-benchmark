from __future__ import annotations

import os
import socket
import urllib.request
from pathlib import Path

import boto3
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# AJUSTES DE MEMORIA DEL DRIVER PYSPARK EN WINDOWS
# ---------------------------------------------------------
os.environ["JAVA_TOOL_OPTIONS"] = "-Xms128m -Xmx512m -XX:+UseSerialGC"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--conf spark.driver.memory=512m "
    "--conf spark.driver.maxResultSize=256m "
    "--conf spark.executor.memory=1g "
    "--conf spark.sql.shuffle.partitions=4 "
    "pyspark-shell"
)


# ---------------------------------------------------------
# AJUSTES MINIO / S3A PARA LAB
# ---------------------------------------------------------
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "").strip()
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "").strip()

if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
    raise RuntimeError(
        "Missing MinIO credentials. Set MINIO_ACCESS_KEY and "
        "MINIO_SECRET_KEY as environment variables."
    )
MINIO_PORT = 9010

MINIO_BUCKET_GOLD = "btc-gold-lab"
MINIO_OBJECT_GOLD = "spot/BTCUSDT/1h/btcusdt_spot_1h_gold_2019_2026"

RUTA_GOLD_S3A = f"s3a://{MINIO_BUCKET_GOLD}/{MINIO_OBJECT_GOLD}"

# Ruta local paralela para no tocar tu Gold original
RUTA_GOLD_WINDOWS = Path(
    "data/gold_lab/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_lab_2019_2026.parquet"
)

RUTA_IVY_WINDOWS = Path(".ivy2").resolve()

# Spark master del laboratorio
SPARK_MASTER_URL = "spark://localhost:7078"

DRIVER_HOST = "host.docker.internal"
DRIVER_PORT = "4050"
BLOCKMANAGER_PORT = "4051"


def detectar_ip_host_compartida() -> str:
    manual = os.getenv("MINIO_SHARED_HOST", "").strip()
    if manual:
        return manual

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def construir_endpoint_minio() -> str:
    manual = os.getenv("MINIO_ENDPOINT_SHARED", "").strip()
    if manual:
        return manual

    ip_host = detectar_ip_host_compartida()
    return f"http://{ip_host}:{MINIO_PORT}"


def verificar_minio_desde_windows(endpoint: str) -> None:
    health_url = f"{endpoint}/minio/health/live"

    with urllib.request.urlopen(health_url, timeout=10) as response:
        status = response.status

    print(f"Healthcheck MinIO-lab desde Windows: {status}")

    if status != 200:
        raise ConnectionError(f"No fue posible validar MinIO-lab en {health_url}")


def crear_spark_session(endpoint_minio: str) -> SparkSession:
    RUTA_IVY_WINDOWS.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName(
            "exportar_gold_lab_desde_windows_pyspark_7078_minio_lab"
        )
        .master(SPARK_MASTER_URL)
        .config("spark.submit.deployMode", "client")
        .config("spark.driver.host", DRIVER_HOST)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.port", DRIVER_PORT)
        .config("spark.blockManager.port", BLOCKMANAGER_PORT)
        .config("spark.jars.ivy", str(RUTA_IVY_WINDOWS))
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint_minio)
        .config("spark.hadoop.fs.s3a.endpoint.region", "us-east-1")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def descargar_part_desde_minio(endpoint_minio: str) -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_minio,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

    response = s3.list_objects_v2(
        Bucket=MINIO_BUCKET_GOLD,
        Prefix=f"{MINIO_OBJECT_GOLD}/",
    )

    keys = [obj["Key"] for obj in response.get("Contents", [])]
    part_keys = sorted(
        key for key in keys if "/part-" in key and key.endswith(".parquet")
    )

    if not part_keys:
        raise FileNotFoundError(
            f"No se encontro part-*.parquet en s3a://{MINIO_BUCKET_GOLD}/{MINIO_OBJECT_GOLD}"
        )

    if len(part_keys) > 1:
        raise RuntimeError(
            "La Gold-lab en MinIO tiene varias partes. Esta exportacion local espera un unico part-*.parquet."
        )

    part_key = part_keys[0]

    RUTA_GOLD_WINDOWS.parent.mkdir(parents=True, exist_ok=True)
    if RUTA_GOLD_WINDOWS.exists():
        RUTA_GOLD_WINDOWS.unlink()

    s3.download_file(MINIO_BUCKET_GOLD, part_key, str(RUTA_GOLD_WINDOWS))

    if not RUTA_GOLD_WINDOWS.exists():
        raise FileNotFoundError(
            f"No se genero el archivo final en Windows: {RUTA_GOLD_WINDOWS}"
        )

    if RUTA_GOLD_WINDOWS.stat().st_size == 0:
        raise RuntimeError(f"El archivo final quedo vacio: {RUTA_GOLD_WINDOWS}")


def exportar_gold_lab() -> None:
    endpoint_minio = construir_endpoint_minio()
    verificar_minio_desde_windows(endpoint_minio)

    spark = crear_spark_session(endpoint_minio)

    print("=" * 70)
    print("EXPORTACION GOLD-LAB DESDE WINDOWS USANDO SPARK 7078 + MINIO-LAB")
    print(f"Master Spark-lab: {SPARK_MASTER_URL}")
    print(f"Endpoint MinIO-lab compartido: {endpoint_minio}")
    print(f"Leyendo Gold-lab desde: {RUTA_GOLD_S3A}")
    print(f"Destino final Windows: {RUTA_GOLD_WINDOWS}")
    print("-" * 70)

    try:
        df = spark.read.parquet(RUTA_GOLD_S3A).orderBy("open_datetime_utc")
        total_filas = df.count()
        print(f"Filas leidas por Spark: {total_filas:,}")

        descargar_part_desde_minio(endpoint_minio)

        print("-" * 70)
        print(f"Gold-lab exportada correctamente a Windows en: {RUTA_GOLD_WINDOWS}")
        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    exportar_gold_lab()
