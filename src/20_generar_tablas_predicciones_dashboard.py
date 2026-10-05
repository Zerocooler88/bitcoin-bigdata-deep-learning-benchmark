from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MONTH_NAMES_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}

PREDICTION_FILE_CANDIDATES = {
    "test": [
        "tabla_predicciones_test.csv",
        "tabla_predicciones_test.xlsx",
    ],
    "validation": [
        "tabla_predicciones_validation.csv",
        "tabla_predicciones_validation.xlsx",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera dos tablas auxiliares para Tableau a partir de las tablas ya exportadas "
            "de predicciones test y validation."
        )
    )
    parser.add_argument(
        "--dashboard-dir",
        type=str,
        default="outputs/dashboard",
        help="Carpeta donde ya existen tabla_predicciones_test y tabla_predicciones_validation.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="Codificación de salida de los CSV.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=";",
        help="Separador para los CSV finales.",
    )
    parser.add_argument(
        "--export-excel",
        action="store_true",
        help="Además de CSV, exporta también en XLSX.",
    )
    parser.add_argument(
        "--prefer-csv",
        action="store_true",
        help="Si existe CSV y XLSX, prioriza CSV. Si no se usa, igual se prioriza CSV por defecto.",
    )
    return parser.parse_args()


def print_line() -> None:
    print("=" * 96)


def print_title(title: str) -> None:
    print_line()
    print(title)
    print_line()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def find_existing_file(dashboard_dir: Path, candidates: list[str]) -> Path:
    for filename in candidates:
        file_path = dashboard_dir / filename
        if file_path.exists():
            return file_path
    raise FileNotFoundError(
        f"No se encontró ninguno de estos archivos en {dashboard_dir}: {candidates}"
    )


def read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        encodings = ["utf-8-sig", "utf-8", "latin-1"]
        separators = [";", ",", "\t", "|"]
        best_df: pd.DataFrame | None = None
        best_score = -1
        last_error: Exception | None = None

        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        file_path,
                        sep=sep,
                        encoding=encoding,
                        engine="python",
                    )
                    score = int(df.shape[1])
                    if score > best_score:
                        best_score = score
                        best_df = df
                    if df.shape[1] >= 5:
                        return df
                except Exception as exc:
                    last_error = exc

        if best_df is not None:
            return best_df

        raise RuntimeError(f"No se pudo leer el CSV '{file_path}': {last_error}")

    if suffix == ".xlsx":
        return pd.read_excel(file_path)

    raise ValueError(f"Formato no soportado: {file_path.suffix}")


def ensure_column(
    df: pd.DataFrame, column_name: str, default_value: Any = np.nan
) -> pd.DataFrame:
    if column_name not in df.columns:
        df[column_name] = default_value
    return df


def choose_datetime_column(df: pd.DataFrame) -> str:
    candidates = [
        "fecha_hora_utc",
        "fecha_hora",
        "datetime",
        "timestamp",
        "fecha",
        "date",
    ]
    normalized_map = {normalize_text(col): col for col in df.columns}

    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]

    raise KeyError(
        "No se encontró una columna de fecha válida. Se esperaba alguna como: "
        "fecha_hora_utc, fecha_hora, datetime, timestamp, fecha o date."
    )


def standardize_prediction_table(
    df: pd.DataFrame, forced_partition: str
) -> pd.DataFrame:
    current = df.copy()

    rename_map = {}
    for column in current.columns:
        key = normalize_text(column)
        if key in {
            "fecha_hora_utc",
            "fecha_hora",
            "datetime",
            "timestamp",
            "fecha",
            "date",
        }:
            rename_map[column] = "fecha_hora_utc"
        elif key in {"model", "modelo"}:
            rename_map[column] = "modelo"
        elif key in {"scenario", "escenario"}:
            rename_map[column] = "escenario"
        elif key in {"partition", "particion", "split", "set"}:
            rename_map[column] = "particion"
        elif key in {"y_true", "valor_real", "real", "actual", "target"}:
            rename_map[column] = "valor_real"
        elif key in {"y_pred", "valor_predicho", "pred", "prediction", "prediccion"}:
            rename_map[column] = "valor_predicho"

    current = current.rename(columns=rename_map)

    required_columns = [
        "fecha_hora_utc",
        "escenario",
        "modelo",
        "particion",
        "valor_real",
        "valor_predicho",
    ]
    for required in required_columns:
        ensure_column(current, required)

    current["particion"] = current["particion"].fillna(forced_partition)
    current["particion"] = current["particion"].astype(str).str.strip().str.lower()
    current.loc[current["particion"].isin(["nan", "none", "null", ""]), "particion"] = (
        forced_partition
    )

    datetime_column = choose_datetime_column(current)
    parsed_dt = pd.to_datetime(current[datetime_column], errors="coerce", utc=True)
    current["fecha_hora_utc"] = parsed_dt.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    current["fecha_hora_utc_iso"] = parsed_dt.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    numeric_candidates = [
        "valor_real",
        "valor_predicho",
        "valor_baseline",
        "residual_modelo",
        "residual_baseline",
        "error_absoluto_modelo",
        "error_absoluto_baseline",
        "lookback",
        "horizon",
    ]
    for column in numeric_candidates:
        if column in current.columns:
            current[column] = pd.to_numeric(current[column], errors="coerce")

    current = current.dropna(subset=["fecha_hora_utc_iso"]).reset_index(drop=True)

    if "id_registro" in current.columns:
        current = current.drop(columns=["id_registro"])
    current.insert(0, "id_registro", np.arange(1, len(current) + 1))

    return current


def build_tabla_predicciones_todas(
    test_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([test_df, validation_df], ignore_index=True, sort=False)
    combined = combined.sort_values(
        by=[
            col
            for col in ["escenario", "modelo", "particion", "fecha_hora_utc_iso"]
            if col in combined.columns
        ],
        kind="stable",
    ).reset_index(drop=True)
    combined["id_registro"] = np.arange(1, len(combined) + 1)
    return combined


def build_tabla_predicciones_dashboard(
    predicciones_todas: pd.DataFrame,
) -> pd.DataFrame:
    current = predicciones_todas.copy()

    dt = pd.to_datetime(current["fecha_hora_utc_iso"], errors="coerce", utc=True)
    current["anio"] = dt.dt.year
    current["mes"] = dt.dt.month
    current["nombre_mes"] = current["mes"].map(MONTH_NAMES_ES)
    current["trimestre"] = "Q" + dt.dt.quarter.astype("Int64").astype(str)
    current["year_month"] = dt.dt.strftime("%Y-%m")

    ordered_columns = [
        "id_registro",
        "fecha_hora_utc",
        "fecha_hora_utc_iso",
        "anio",
        "mes",
        "nombre_mes",
        "trimestre",
        "year_month",
        "escenario",
        "modelo",
        "particion",
        "modo_target_entrenamiento",
        "valor_real",
        "valor_predicho",
        "valor_baseline",
        "residual_modelo",
        "residual_baseline",
        "error_absoluto_modelo",
        "error_absoluto_baseline",
        "tipo_archivo",
        "nombre_archivo",
        "archivo_origen",
        "lookback",
        "horizon",
    ]

    ordered_columns_existing = [
        col for col in ordered_columns if col in current.columns
    ]
    remaining_columns = [
        col for col in current.columns if col not in ordered_columns_existing
    ]
    current = current[ordered_columns_existing + remaining_columns]

    return current


def export_csv(df: pd.DataFrame, export_path: Path, encoding: str, sep: str) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        export_path,
        index=False,
        encoding=encoding,
        sep=sep,
        quoting=csv.QUOTE_ALL,
    )


def export_excel(df: pd.DataFrame, export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="datos")


def main() -> None:
    args = parse_args()
    dashboard_dir = Path(args.dashboard_dir).resolve()

    print_title("GENERACION DE TABLAS UNIFICADAS DE PREDICCIONES PARA TABLEAU")
    print(f"dashboard_dir = {dashboard_dir}")
    print(f"encoding      = {args.encoding}")
    print(f"sep           = {args.sep}")
    print(f"export_excel  = {args.export_excel}")

    test_file = find_existing_file(dashboard_dir, PREDICTION_FILE_CANDIDATES["test"])
    validation_file = find_existing_file(
        dashboard_dir, PREDICTION_FILE_CANDIDATES["validation"]
    )

    print_title("1) LECTURA DE TABLAS BASE")
    print(f"tabla test        : {test_file}")
    print(f"tabla validation  : {validation_file}")

    raw_test_df = read_table(test_file)
    raw_validation_df = read_table(validation_file)

    test_df = standardize_prediction_table(raw_test_df, forced_partition="test")
    validation_df = standardize_prediction_table(
        raw_validation_df, forced_partition="validation"
    )

    print(f"filas test        : {len(test_df)}")
    print(f"filas validation  : {len(validation_df)}")

    print_title("2) CONSTRUCCION DE NUEVAS TABLAS")
    tabla_predicciones_todas = build_tabla_predicciones_todas(test_df, validation_df)
    tabla_predicciones_dashboard = build_tabla_predicciones_dashboard(
        tabla_predicciones_todas
    )

    print(f"tabla_predicciones_todas      : {len(tabla_predicciones_todas)} filas")
    print(f"tabla_predicciones_dashboard  : {len(tabla_predicciones_dashboard)} filas")

    print_title("3) EXPORTACION")
    output_csv_1 = dashboard_dir / "tabla_predicciones_todas.csv"
    output_csv_2 = dashboard_dir / "tabla_predicciones_dashboard.csv"

    export_csv(tabla_predicciones_todas, output_csv_1, args.encoding, args.sep)
    export_csv(tabla_predicciones_dashboard, output_csv_2, args.encoding, args.sep)

    print(f"OK CSV : {output_csv_1}")
    print(f"OK CSV : {output_csv_2}")

    if args.export_excel:
        output_xlsx_1 = dashboard_dir / "tabla_predicciones_todas.xlsx"
        output_xlsx_2 = dashboard_dir / "tabla_predicciones_dashboard.xlsx"
        export_excel(tabla_predicciones_todas, output_xlsx_1)
        export_excel(tabla_predicciones_dashboard, output_xlsx_2)
        print(f"OK XLSX: {output_xlsx_1}")
        print(f"OK XLSX: {output_xlsx_2}")

    print_title("4) FINAL")
    print("Proceso terminado correctamente.")
    print("No se modificaron las tablas originales.")
    print("Solo se crearon dos tablas nuevas a partir de test + validation.")


if __name__ == "__main__":
    main()
