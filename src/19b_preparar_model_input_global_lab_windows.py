from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import polars as pl


# --------------------------------------------------
# RUTAS PRINCIPALES
# --------------------------------------------------
RUTA_GOLD_LAB = Path(
    "data/gold_lab/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_lab_2019_2026.parquet"
)

# Copia paralela segura del model input preparado desde laboratorio
RUTA_GLOBAL_LAB = Path(
    "data/1h/global_lab/btcusdt_spot_1h_model_input_2019_2026_lab.parquet"
)

# Ruta exacta que consumen 09 y 10 sin modificar esos scripts
RUTA_GLOBAL_FINAL = Path("data/1h/global/btcusdt_spot_1h_model_input_2019_2026.parquet")

CARPETA_BACKUP = Path("data/1h/global_backup_pre_lab")
RUTA_MANIFIESTO = CARPETA_BACKUP / "ultimo_backup_lab.txt"


# --------------------------------------------------
# VALIDACIONES
# --------------------------------------------------
def validar_gold_lab(df: pl.DataFrame) -> None:
    columnas_obligatorias = [
        "open_datetime_utc",
        "regimen_mercado",
        "close",
        "open",
        "high",
        "low",
        "volume",
    ]

    faltantes = [c for c in columnas_obligatorias if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias en la Gold-lab exportada: {faltantes}"
        )


# --------------------------------------------------
# PREPARACION DEL MODEL INPUT
# --------------------------------------------------
def preparar_model_input(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("open_datetime_utc")

    # Asegurar tipo datetime por seguridad
    df = df.with_columns(
        [
            pl.col("open_datetime_utc").cast(pl.Datetime),
        ]
    )

    # Agregar columnas temporales base
    df = df.with_columns(
        [
            pl.col("open_datetime_utc").dt.year().alias("anio"),
            pl.col("open_datetime_utc").dt.month().alias("mes"),
            pl.col("open_datetime_utc").dt.day().alias("dia"),
            pl.col("open_datetime_utc").dt.hour().alias("hora"),
            pl.col("open_datetime_utc").dt.quarter().alias("trimestre"),
        ]
    )

    # Agregar year_quarter y target_close
    df = df.with_columns(
        [
            (
                pl.col("anio").cast(pl.Utf8)
                + pl.lit("-Q")
                + pl.col("trimestre").cast(pl.Utf8)
            ).alias("year_quarter"),
            pl.col("close").alias("target_close"),
        ]
    )

    # Orden final de columnas:
    # primero tiempo, luego calendario/regimen, luego OHLCV y luego derivadas
    columnas_preferidas = [
        "open_datetime_utc",
        "anio",
        "mes",
        "dia",
        "hora",
        "trimestre",
        "year_quarter",
        "regimen_mercado",
        "open",
        "high",
        "low",
        "close",
        "target_close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "filas_en_hora",
        "hora_completa",
        "rango_hl",
        "cuerpo_oc",
        "return_1h",
        "log_return_1h",
        "sma_24h",
        "sma_168h",
        "volatilidad_24h",
        "volumen_sma_24h",
    ]

    columnas_finales = [c for c in columnas_preferidas if c in df.columns]
    return df.select(columnas_finales)


# --------------------------------------------------
# BACKUP DEL GLOBAL ACTUAL
# --------------------------------------------------
def hacer_backup_si_existe() -> Path | None:
    if not RUTA_GLOBAL_FINAL.exists():
        return None

    CARPETA_BACKUP.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_backup = (
        CARPETA_BACKUP
        / f"btcusdt_spot_1h_model_input_2019_2026__backup_{timestamp}.parquet"
    )

    shutil.copy2(RUTA_GLOBAL_FINAL, ruta_backup)
    RUTA_MANIFIESTO.write_text(str(ruta_backup), encoding="utf-8")

    return ruta_backup


# --------------------------------------------------
# FUNCION PRINCIPAL
# --------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("PREPARACION MODEL_INPUT GLOBAL LAB EN WINDOWS")
    print(f"Leyendo Gold-lab local desde: {RUTA_GOLD_LAB}")
    print(f"Salida paralela lab: {RUTA_GLOBAL_LAB}")
    print(f"Salida final exacta para 09 y 10: {RUTA_GLOBAL_FINAL}")
    print("=" * 70)

    if not RUTA_GOLD_LAB.exists():
        raise FileNotFoundError(
            f"No existe la Gold-lab exportada en Windows: {RUTA_GOLD_LAB}"
        )

    df = pl.read_parquet(RUTA_GOLD_LAB)
    validar_gold_lab(df)

    print(f"Filas leidas desde Gold-lab local: {df.height:,}")
    print(f"Columnas leidas desde Gold-lab local: {df.width}")

    df_model_input = preparar_model_input(df)

    # Validar columnas mínimas exigidas por 09
    columnas_minimas_09 = [
        "open_datetime_utc",
        "year_quarter",
        "regimen_mercado",
        "close",
    ]
    faltantes_09 = [c for c in columnas_minimas_09 if c not in df_model_input.columns]
    if faltantes_09:
        raise ValueError(
            f"El model_input final no cumple con lo que exige 09_generar_escenarios_1h.py. Faltan: {faltantes_09}"
        )

    # Hacer backup del parquet global actual, si existe
    ruta_backup = hacer_backup_si_existe()

    # Crear carpetas de salida
    RUTA_GLOBAL_LAB.parent.mkdir(parents=True, exist_ok=True)
    RUTA_GLOBAL_FINAL.parent.mkdir(parents=True, exist_ok=True)

    # Guardar copia paralela lab
    df_model_input.write_parquet(RUTA_GLOBAL_LAB, compression="zstd")

    # Guardar archivo exacto que esperan 09 y 10
    df_model_input.write_parquet(RUTA_GLOBAL_FINAL, compression="zstd")

    print("-" * 70)
    if ruta_backup is None:
        print("No existia parquet global previo. No fue necesario backup.")
    else:
        print(f"Backup del parquet global previo guardado en: {ruta_backup}")

    print(f"Model input lab guardado en: {RUTA_GLOBAL_LAB}")
    print(f"Model input final exacto guardado en: {RUTA_GLOBAL_FINAL}")
    print(f"Filas finales model_input: {df_model_input.height:,}")
    print(f"Columnas finales model_input: {df_model_input.width}")
    print("=" * 70)
    print("MODEL_INPUT GLOBAL LAB PREPARADO CORRECTAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()
