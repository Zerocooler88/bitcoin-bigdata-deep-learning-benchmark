from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "1h"
    / "global"
    / "btcusdt_spot_1h_model_input_2019_2026.parquet"
)


FEATURES_22 = [
    "anio",
    "mes",
    "dia",
    "hora",
    "trimestre",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "rango_hl",
    "cuerpo_oc",
    "return_1h",
    "log_return_1h",
    "sma_24h",
    "sma_168h",
    "volatilidad_24h",
    "volumen_sma_24h",
]


def main():
    print("=" * 70)
    print("VERIFICACION DEL PARQUET GLOBAL Q1")
    print("=" * 70)

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"No existe el parquet:\n{PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)

    print(f"Archivo: {PARQUET_PATH}")
    print(f"Filas: {len(df):,}")
    print(f"Columnas totales: {len(df.columns)}")
    print()

    faltantes = [col for col in FEATURES_22 if col not in df.columns]

    print("VARIABLES DEL BENCHMARK:")
    print()

    for numero, feature in enumerate(FEATURES_22, start=1):
        estado = "OK" if feature in df.columns else "FALTA"
        print(f"{numero:02d}. {feature:<25} {estado}")

    print()
    print("=" * 70)

    if faltantes:
        print("ERROR")
        print("Faltan estas columnas:")
        for col in faltantes:
            print(f"- {col}")

        raise ValueError("El parquet no contiene todas las variables requeridas.")

    print("OK: las 22 variables existen fisicamente en el parquet.")

    columnas_control = [
        "filas_en_hora",
        "hora_completa",
        "target_close",
        "year_quarter",
        "regimen_mercado",
    ]

    print()
    print("COLUMNAS DE CONTROL / METADATOS PRESENTES:")
    print()

    for col in columnas_control:
        if col in df.columns:
            print(f"{col}: PRESENTE EN PARQUET, pero NO se usara como predictor.")
        else:
            print(f"{col}: no presente.")

    print()
    print("OK: parquet listo para el benchmark Q1.")
    print("=" * 70)


if __name__ == "__main__":
    main()
