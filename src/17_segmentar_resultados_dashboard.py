from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VALID_MODELS = {
    "baseline_persistencia",
    "lstm",
    "gru",
    "cnn1d",
    "cnn_lstm",
}

VALID_SCENARIOS = [
    "global_2019_2026",
    "pre_covid",
    "covid",
    "post_covid",
    "trimestre_mas_bajo",
    "trimestre_mas_alto",
    "intra_2024_q1",
    "intra_2024_q3",
    "intra_2025_q1",
    "cross_2024_q1_to_2024_q3",
    "cross_2024_q1_to_2025_q1",
    "cross_pre_to_post",
    "cross_bajo_to_alto",
]

METRIC_PRIORITY_FOR_RANKING = [
    "test_rmse",
    "rmse_test",
    "val_rmse",
    "validation_rmse",
    "rmse_validation",
    "rmse",
    "test_mae",
    "mae_test",
    "val_mae",
    "validation_mae",
    "mae_validation",
    "mae",
    "test_mape",
    "mape_test",
    "val_mape",
    "validation_mape",
    "mape",
    "test_r2",
    "r2_test",
    "val_r2",
    "validation_r2",
    "r2_validation",
    "r2",
]

DATE_COLUMN_HINTS = {
    "fecha",
    "date",
    "time",
    "timestamp",
    "datetime",
    "periodo",
    "mes",
    "inicio",
    "fin",
    "hora",
}

NUMERIC_COLUMN_HINTS = {
    "mae",
    "rmse",
    "mse",
    "mape",
    "smape",
    "r2",
    "loss",
    "epoch",
    "epoca",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "volumen",
    "precio",
    "price",
    "real",
    "pred",
    "prediccion",
    "y_true",
    "y_pred",
    "tiempo",
    "segundos",
    "duration",
    "duracion",
    "lr",
    "learning_rate",
    "batch_size",
    "lookback",
    "horizon",
}

TABLEAU_METRIC_COLUMN_HINTS = {
    "mae",
    "rmse",
    "mse",
    "mape",
    "smape",
    "r2",
    "train_loss",
    "val_loss",
    "test_loss",
    "train_mae",
    "val_mae",
    "test_mae",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "train_mse",
    "val_mse",
    "test_mse",
    "train_mape",
    "val_mape",
    "test_mape",
    "train_r2",
    "val_r2",
    "test_r2",
    "tiempo_entrenamiento_segundos",
    "tiempo_total_segundos",
    "segundos_por_epoca",
    "mejor_epoca",
}

COMMON_COLUMN_RENAMES = {
    "model": "modelo",
    "modelo_nombre": "modelo",
    "experiment": "experimento",
    "scenario": "escenario",
    "split": "particion",
    "partition": "particion",
    "set": "particion",
    "epoch": "epoca",
    "best_epoch": "mejor_epoca",
    "bestepoch": "mejor_epoca",
    "target_mode": "modo_target_entrenamiento",
    "targetmode": "modo_target_entrenamiento",
    "learningrate": "learning_rate",
    "batchsize": "batch_size",
    "numworkers": "num_workers",
    "file_source": "archivo_origen",
    "source_file": "archivo_origen",
    "filename": "nombre_archivo",
    "source_name": "nombre_archivo",
    "date": "fecha_hora",
    "datetime": "fecha_hora",
    "timestamp": "fecha_hora",
    "fecha": "fecha_hora",
    "time": "fecha_hora",
    "y_true": "valor_real",
    "y_pred": "valor_predicho",
    "pred": "valor_predicho",
    "prediction": "valor_predicho",
    "prediccion": "valor_predicho",
    "real": "valor_real",
    "actual": "valor_real",
    "target": "valor_real",
    "true": "valor_real",
    "metric": "metrica",
    "metric_name": "metrica",
    "metricvalue": "valor_metrica",
    "metric_value": "valor_metrica",
    "value": "valor_metrica",
    "close_real": "close_real",
    "close_pred": "close_pred",
    "pred_usd": "prediccion_usd",
    "real_usd": "real_usd",
    "close_real_usd": "close_real_usd",
    "close_pred_usd": "close_pred_usd",
}


@dataclass
class ProcessingReport:
    archivos_encontrados: dict[str, int] = field(default_factory=dict)
    tablas_exportadas: dict[str, int] = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)
    errores: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.advertencias.append(message)
        print(f"[ADVERTENCIA] {message}")

    def error(self, message: str) -> None:
        self.errores.append(message)
        print(f"[ERROR] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida resultados de experimentos y genera tablas limpias para Tableau."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Ruta raíz del proyecto. Por defecto usa la carpeta actual.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Carpeta donde están las salidas del proyecto. Por defecto: outputs",
    )
    parser.add_argument(
        "--dashboard-dir",
        type=str,
        default="outputs/dashboard",
        help="Carpeta donde se exportarán las tablas curadas para Tableau.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="Codificación con la que se guardarán los archivos finales.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=";",
        help="Separador para los CSV finales. Recomendado para Tableau/Excel: punto y coma.",
    )
    parser.add_argument(
        "--export-excel",
        action="store_true",
        help="Además de CSV, exporta cada tabla también en .xlsx.",
    )
    return parser.parse_args()


def print_line() -> None:
    print("=" * 96)


def print_title(title: str) -> None:
    print_line()
    print(title)
    print_line()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("%", " porcentaje ")
    text = text.replace("$", " usd ")
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def sanitize_text_for_tableau(value: Any) -> Any:
    if value is None:
        return np.nan

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()

    if isinstance(value, float) and pd.isna(value):
        return np.nan

    if not isinstance(value, str):
        return value

    text = value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("\t", " ").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return np.nan

    return text


def normalize_partition_value(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, float) and pd.isna(value):
        return np.nan

    text = normalize_text(value)

    exact_map = {
        "prueba": "test",
        "test": "test",
        "baseline_prueba": "test",
        "validation": "validation",
        "validacion": "validation",
        "val": "validation",
        "baseline_validacion": "validation",
        "train": "train",
        "training": "train",
        "entrenamiento": "train",
        "history": "history",
    }

    if text in exact_map:
        return exact_map[text]

    if "validacion" in text or "validation" in text or text.startswith("val"):
        return "validation"
    if "prueba" in text or re.search(r"(^|_)test($|_)", text):
        return "test"
    if "train" in text or "entrenamiento" in text:
        return "train"

    return text


def normalize_metric_value(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, float) and pd.isna(value):
        return np.nan
    return normalize_text(value)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed_columns: dict[str, str] = {}
    seen: dict[str, int] = {}

    for original in df.columns:
        normalized = normalize_text(original)
        normalized = COMMON_COLUMN_RENAMES.get(normalized, normalized)
        if not normalized:
            normalized = "columna"

        if normalized in seen:
            seen[normalized] += 1
            normalized = f"{normalized}_{seen[normalized]}"
        else:
            seen[normalized] = 0

        renamed_columns[original] = normalized

    return df.rename(columns=renamed_columns)


def smart_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()

    if isinstance(value, (dict, list, tuple, set)):
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def is_nested_value(value: Any) -> bool:
    return isinstance(value, (dict, list, tuple, set))


def flatten_nested_dataframe(
    df: pd.DataFrame, report: ProcessingReport, source_name: str
) -> pd.DataFrame:
    current = df.copy()
    max_rounds = 5

    for _ in range(max_rounds):
        nested_columns = [
            column
            for column in current.columns
            if current[column].map(is_nested_value).fillna(False).any()
        ]

        if not nested_columns:
            break

        for column in nested_columns:
            series = current[column]
            non_null_values = series.dropna()
            if non_null_values.empty:
                continue

            first_value = non_null_values.iloc[0]

            if isinstance(first_value, dict):
                dict_rows: list[dict[str, Any]] = []
                for value in series:
                    if isinstance(value, dict):
                        dict_rows.append(value)
                    elif value is None or (isinstance(value, float) and pd.isna(value)):
                        dict_rows.append({})
                    else:
                        dict_rows.append({"valor": smart_scalar(value)})

                flattened = pd.json_normalize(dict_rows, sep="_")
                flattened.columns = [
                    normalize_text(f"{column}_{subcol}") for subcol in flattened.columns
                ]
                current = pd.concat([current.drop(columns=[column]), flattened], axis=1)
            else:
                report.warn(
                    f"La columna '{column}' de '{source_name}' contenía listas u otras estructuras; se convirtió a texto JSON."
                )
                current[column] = series.map(
                    lambda value: (
                        json.dumps(value, ensure_ascii=False)
                        if is_nested_value(value)
                        else value
                    )
                )

    return current


def robust_read_csv(file_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    best_df: pd.DataFrame | None = None
    best_score = -1

    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    separators = [",", ";", "\t", "|"]

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding=encoding,
                    engine="python",
                    dtype=str,
                    keep_default_na=False,
                    na_values=[],
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines="skip",
                )
                score = int(df.shape[1])
                if score > best_score:
                    best_score = score
                    best_df = df

                if df.shape[1] >= 3:
                    return df
            except Exception as error:
                last_error = error

    if best_df is not None:
        return best_df

    raise RuntimeError(f"No se pudo leer el CSV '{file_path}': {last_error}")


def robust_read_json(file_path: Path) -> pd.DataFrame:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return pd.json_normalize(data, sep="_")

    if isinstance(data, dict):
        for key in [
            "records",
            "items",
            "data",
            "rows",
            "metricas",
            "predicciones",
            "history",
        ]:
            if key in data and isinstance(data[key], list):
                return pd.json_normalize(data[key], sep="_")
        return pd.json_normalize([data], sep="_")

    return pd.DataFrame({"valor": [data]})


def robust_read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return robust_read_csv(file_path)
    if suffix == ".json":
        return robust_read_json(file_path)
    raise ValueError(f"Formato no soportado: {file_path.suffix}")


def looks_like_date_column(column_name: str) -> bool:
    normalized = normalize_text(column_name)
    return any(hint in normalized for hint in DATE_COLUMN_HINTS)


def looks_like_numeric_column(column_name: str) -> bool:
    normalized = normalize_text(column_name)
    return any(hint in normalized for hint in NUMERIC_COLUMN_HINTS)


def parse_mixed_numeric(value: Any) -> Any:
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if text == "":
        return np.nan

    text = text.replace("\u00a0", " ").replace(" ", "")
    text = text.replace("%", "")
    text = text.replace("USD", "").replace("usd", "")

    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return np.nan

    has_comma = "," in text
    has_dot = "." in text

    if has_comma and has_dot:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        if text.count(",") > 1:
            text = text.replace(",", "")
        else:
            left, right = text.split(",")
            if len(right) == 3 and left.replace("-", "").isdigit() and right.isdigit():
                text = left + right
            else:
                text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return value


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()

    for column in current.columns:
        if current[column].dtype == bool:
            continue
        if pd.api.types.is_numeric_dtype(current[column]):
            continue

        series = current[column]
        if looks_like_numeric_column(column) or series.dtype == object:
            cleaned = series.map(parse_mixed_numeric)
            converted = pd.to_numeric(cleaned, errors="coerce")
            convertible_ratio = converted.notna().mean() if len(converted) > 0 else 0.0
            if convertible_ratio >= 0.80:
                current[column] = converted

    return current


def coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()

    for column in current.columns:
        if pd.api.types.is_datetime64_any_dtype(current[column]):
            current[column] = current[column].dt.strftime("%Y-%m-%d %H:%M:%S")
            continue

        if current[column].dtype != object:
            continue

        if not looks_like_date_column(column):
            continue

        parsed = pd.to_datetime(current[column], errors="coerce", utc=False)
        if len(parsed) > 0 and parsed.notna().mean() >= 0.70:
            current[column] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")

    return current


def trim_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    for column in current.columns:
        if current[column].dtype == object:
            current[column] = current[column].map(sanitize_text_for_tableau)
    return current


def drop_empty_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    current = current.dropna(axis=0, how="all")
    current = current.dropna(axis=1, how="all")
    return current.reset_index(drop=True)


def remove_duplicate_rows(
    df: pd.DataFrame, report: ProcessingReport, table_name: str
) -> pd.DataFrame:
    if df.empty:
        return df

    before = len(df)
    deduplicated = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(deduplicated)

    if removed > 0:
        report.warn(f"Se eliminaron {removed} filas duplicadas en '{table_name}'.")

    return deduplicated


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized_map = {normalize_text(column): column for column in df.columns}

    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    return None


def choose_first_existing_column(
    df: pd.DataFrame, candidates: Iterable[str]
) -> str | None:
    return find_first_column(df, candidates)


def infer_model_from_path(file_path: Path, df: pd.DataFrame) -> str | None:
    text = normalize_text(file_path.as_posix())

    for model in sorted(VALID_MODELS, key=len, reverse=True):
        if model in text:
            return model

    model_column = find_first_column(df, ["modelo", "model", "experimento"])
    if model_column is not None:
        values = df[model_column].dropna().astype(str).tolist()
        for value in values:
            normalized_value = normalize_text(value)
            if normalized_value in VALID_MODELS:
                return normalized_value
            for model in VALID_MODELS:
                if model in normalized_value:
                    return model

    return None


def infer_scenario_from_path(file_path: Path, df: pd.DataFrame) -> str | None:
    text = normalize_text(file_path.as_posix())

    for scenario in sorted(VALID_SCENARIOS, key=len, reverse=True):
        if scenario in text:
            return scenario

    scenario_column = find_first_column(df, ["escenario", "scenario"])
    if scenario_column is not None:
        values = df[scenario_column].dropna().astype(str).tolist()
        for value in values:
            normalized_value = normalize_text(value)
            for scenario in VALID_SCENARIOS:
                if normalized_value == scenario or scenario in normalized_value:
                    return scenario

    return None


def infer_split_from_path(file_path: Path, category: str, df: pd.DataFrame) -> str:
    split_column = find_first_column(df, ["particion", "split", "partition", "set"])
    if split_column is not None and df[split_column].notna().any():
        raw_split = df[split_column].dropna().astype(str).iloc[0]
        return str(normalize_partition_value(raw_split))

    text = normalize_text(file_path.as_posix())

    if (
        "validation" in text
        or "validacion" in text
        or category == "predictions_validation"
    ):
        return "validation"
    if re.search(r"(^|_)test($|_)", text) or category == "predictions_test":
        return "test"
    if "train" in text or "training" in text or "entrenamiento" in text:
        return "train"
    if category == "history":
        return "history"
    if category == "metrics":
        return "metricas"
    if category == "summary":
        return "resumen"

    return "desconocido"


def categorize_file(file_path: Path) -> str | None:
    text = normalize_text(file_path.as_posix())

    if file_path.suffix.lower() not in {".csv", ".json"}:
        return None

    if "resumen_consolidado" in text or "summary" in text or "resumen" in text:
        return "summary"

    if "history" in text:
        return "history"

    parts = {normalize_text(part) for part in file_path.parts}

    if "metrics" in parts or "metric" in text:
        return "metrics"

    if "predictions" in parts or "predicciones" in text or "predictions" in text:
        if "validation" in text or "validacion" in text:
            return "predictions_validation"
        if re.search(r"(^|_)test($|_)", text):
            return "predictions_test"

    if "validation" in text or "validacion" in text:
        return "predictions_validation"

    if re.search(r"(^|_)test($|_)", text):
        return "predictions_test"

    return None


def scan_candidate_files(
    outputs_dir: Path,
    report: ProcessingReport,
    excluded_dirs: Iterable[Path] | None = None,
) -> dict[str, list[Path]]:
    categories: dict[str, list[Path]] = {
        "metrics": [],
        "predictions_validation": [],
        "predictions_test": [],
        "history": [],
        "summary": [],
    }

    if not outputs_dir.exists():
        report.error(f"La carpeta de salida '{outputs_dir}' no existe.")
        return categories

    excluded_resolved = {path.resolve() for path in (excluded_dirs or [])}
    excluded_names = {"dashboard_manifest.json"}
    excluded_prefixes = ("tabla_",)

    all_files = sorted(outputs_dir.rglob("*"))
    skipped_dashboard_files = 0

    for file_path in all_files:
        if not file_path.is_file():
            continue

        resolved_file = file_path.resolve()
        if any(
            excluded_dir == resolved_file.parent
            or excluded_dir in resolved_file.parents
            for excluded_dir in excluded_resolved
        ):
            skipped_dashboard_files += 1
            continue

        if file_path.name in excluded_names or file_path.name.startswith(
            excluded_prefixes
        ):
            skipped_dashboard_files += 1
            continue

        category = categorize_file(file_path)
        if category is not None:
            categories[category].append(file_path)

    if skipped_dashboard_files > 0:
        report.warn(
            f"Se omitieron {skipped_dashboard_files} archivos dentro de carpetas de dashboard o tablas curadas previas para evitar relectura circular."
        )

    for category, paths in categories.items():
        report.archivos_encontrados[category] = len(paths)

    return categories


def standardize_metrics_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()

    rename_map = {
        "best_epoch": "mejor_epoca",
        "best_epoca": "mejor_epoca",
        "r_2": "r2",
        "r_cuadrado": "r2",
        "validation_mae": "val_mae",
        "validation_rmse": "val_rmse",
        "validation_mse": "val_mse",
        "validation_mape": "val_mape",
        "validation_r2": "val_r2",
        "test_mae": "test_mae",
        "test_rmse": "test_rmse",
        "test_mse": "test_mse",
        "test_mape": "test_mape",
        "test_r2": "test_r2",
    }
    current = current.rename(
        columns={column: rename_map.get(column, column) for column in current.columns}
    )

    if "metrica" in current.columns:
        current["metrica"] = current["metrica"].map(
            lambda value: normalize_metric_value(value) if pd.notna(value) else value
        )

    if "valor_metrica" in current.columns:
        current["valor_metrica"] = pd.to_numeric(
            current["valor_metrica"].map(parse_mixed_numeric),
            errors="coerce",
        )

    return current


def standardize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()

    rename_map: dict[str, str] = {}
    for column in current.columns:
        normalized = normalize_text(column)

        if normalized in {"close_real_usd", "real_usd_close"}:
            rename_map[column] = "close_real_usd"
        elif normalized in {"close_pred_usd", "pred_usd_close"}:
            rename_map[column] = "close_pred_usd"
        elif normalized in {"predicted_close", "pred_close", "close_pred"}:
            rename_map[column] = "close_pred"
        elif normalized in {"real_close", "close_true", "close_real"}:
            rename_map[column] = "close_real"
        elif normalized in {"timestamp", "datetime", "date", "fecha", "fecha_hora"}:
            rename_map[column] = "fecha_hora"
        elif normalized in {"y_true", "valor_real", "real", "actual", "target"}:
            rename_map[column] = "valor_real"
        elif normalized in {
            "y_pred",
            "valor_predicho",
            "pred",
            "prediction",
            "prediccion",
        }:
            rename_map[column] = "valor_predicho"

    current = current.rename(columns=rename_map)

    if "id_registro" not in current.columns:
        current.insert(0, "id_registro", np.arange(1, len(current) + 1))

    return current


def standardize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    rename_map = {
        "epoch": "epoca",
        "train_loss": "train_loss",
        "val_loss": "val_loss",
        "validation_loss": "val_loss",
        "validation_rmse": "val_rmse",
        "validation_mae": "val_mae",
        "train_rmse": "train_rmse",
        "train_mae": "train_mae",
        "seconds_per_epoch": "segundos_por_epoca",
        "epoch_seconds": "segundos_por_epoca",
    }
    current = current.rename(
        columns={column: rename_map.get(column, column) for column in current.columns}
    )
    return current


def standardize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    rename_map = {
        "best_epoch": "mejor_epoca",
        "best_validation_rmse": "mejor_val_rmse",
        "best_validation_mae": "mejor_val_mae",
        "training_seconds": "tiempo_entrenamiento_segundos",
        "total_seconds": "tiempo_total_segundos",
        "target_mode": "modo_target_entrenamiento",
    }
    current = current.rename(
        columns={column: rename_map.get(column, column) for column in current.columns}
    )
    return current


def standardize_context_columns(
    df: pd.DataFrame, file_path: Path, category: str, report: ProcessingReport
) -> pd.DataFrame:
    current = df.copy()

    current = normalize_columns(current)
    current = flatten_nested_dataframe(
        current, report=report, source_name=file_path.name
    )
    current = normalize_columns(current)
    current = trim_object_columns(current)
    current = coerce_numeric_columns(current)
    current = coerce_datetime_columns(current)

    model = infer_model_from_path(file_path, current)
    scenario = infer_scenario_from_path(file_path, current)
    split = infer_split_from_path(file_path, category, current)

    if "modelo" not in current.columns:
        current.insert(0, "modelo", model)
    else:
        current["modelo"] = current["modelo"].fillna(model).astype(object)
        current["modelo"] = current["modelo"].map(
            lambda value: normalize_text(value) if pd.notna(value) else value
        )
        current["modelo"] = current["modelo"].replace({"": np.nan})
        current["modelo"] = current["modelo"].fillna(model)

    if "escenario" not in current.columns:
        current.insert(1, "escenario", scenario)
    else:
        current["escenario"] = current["escenario"].fillna(scenario)
        current["escenario"] = current["escenario"].map(
            lambda value: normalize_text(value) if pd.notna(value) else value
        )

    if "particion" not in current.columns:
        current.insert(2, "particion", split)
    else:
        current["particion"] = current["particion"].fillna(split)

    current["particion"] = current["particion"].map(
        lambda value: normalize_partition_value(value) if pd.notna(value) else value
    )

    current["tipo_archivo"] = category
    current["nombre_archivo"] = file_path.name
    current["archivo_origen"] = str(file_path.as_posix())

    if "modelo" in current.columns:
        invalid_model_mask = current["modelo"].notna() & ~current["modelo"].astype(
            str
        ).isin(VALID_MODELS)
        if invalid_model_mask.any():
            current.loc[invalid_model_mask, "modelo"] = model

    if category in {"predictions_validation", "predictions_test"}:
        current = standardize_prediction_columns(current)
    elif category == "history":
        current = standardize_history_columns(current)
    elif category == "metrics":
        current = standardize_metrics_columns(current)
    elif category == "summary":
        current = standardize_summary_columns(current)

    if "metrica" in current.columns:
        current["metrica"] = current["metrica"].map(
            lambda value: normalize_metric_value(value) if pd.notna(value) else value
        )

    current = coerce_numeric_columns(current)
    current = coerce_datetime_columns(current)
    current = trim_object_columns(current)
    current = drop_empty_rows_and_columns(current)

    return current


def validate_required_context(
    df: pd.DataFrame, file_path: Path, report: ProcessingReport
) -> None:
    for column in ["modelo", "escenario", "particion", "archivo_origen"]:
        if column not in df.columns:
            report.warn(
                f"El archivo '{file_path.name}' quedó sin la columna obligatoria '{column}'."
            )

    if "modelo" in df.columns and df["modelo"].isna().all():
        report.warn(f"No se pudo inferir el modelo del archivo '{file_path.name}'.")

    if "escenario" in df.columns and df["escenario"].isna().all():
        report.warn(f"No se pudo inferir el escenario del archivo '{file_path.name}'.")


def load_category_tables(
    file_paths: list[Path], category: str, report: ProcessingReport
) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []

    for file_path in file_paths:
        try:
            raw_df = robust_read_table(file_path)
            if raw_df.empty:
                report.warn(f"El archivo '{file_path.name}' está vacío y se omitirá.")
                continue

            clean_df = standardize_context_columns(raw_df, file_path, category, report)
            validate_required_context(clean_df, file_path, report)
            tables.append(clean_df)
        except Exception as error:
            report.error(f"No se pudo procesar '{file_path}': {error}")

    return tables


def concatenate_tables(
    tables: list[pd.DataFrame], report: ProcessingReport, table_name: str
) -> pd.DataFrame:
    if not tables:
        report.warn(f"No se pudieron construir datos para la tabla '{table_name}'.")
        return pd.DataFrame()

    try:
        concatenated = pd.concat(tables, ignore_index=True, sort=False)
    except Exception as error:
        report.error(f"Error al concatenar la tabla '{table_name}': {error}")
        return pd.DataFrame()

    concatenated = normalize_columns(concatenated)
    concatenated = coerce_numeric_columns(concatenated)
    concatenated = coerce_datetime_columns(concatenated)
    concatenated = trim_object_columns(concatenated)
    concatenated = drop_empty_rows_and_columns(concatenated)
    concatenated = remove_duplicate_rows(
        concatenated, report=report, table_name=table_name
    )
    return concatenated


def build_history_best_epoch_summary(
    history_df: pd.DataFrame, report: ProcessingReport
) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    metric_column = choose_first_existing_column(
        history_df,
        [
            "val_loss",
            "validation_loss",
            "val_rmse",
            "validation_rmse",
            "val_mae",
            "validation_mae",
        ],
    )
    epoch_column = choose_first_existing_column(history_df, ["epoca", "epoch"])

    if metric_column is None or epoch_column is None:
        report.warn(
            "No se pudo derivar la mejor época desde history porque faltan columnas como 'epoca' o alguna métrica de validación."
        )
        return pd.DataFrame()

    key_columns = [
        column for column in ["modelo", "escenario"] if column in history_df.columns
    ]
    if not key_columns:
        return pd.DataFrame()

    work = history_df.copy()
    work[metric_column] = pd.to_numeric(work[metric_column], errors="coerce")
    work[epoch_column] = pd.to_numeric(work[epoch_column], errors="coerce")
    work = work.dropna(subset=[metric_column, epoch_column])
    if work.empty:
        return pd.DataFrame()

    best_idx = work.groupby(key_columns, dropna=False)[metric_column].idxmin()
    best_rows = work.loc[best_idx].copy()
    best_rows = best_rows[key_columns + [epoch_column, metric_column]]
    best_rows = best_rows.rename(
        columns={epoch_column: "mejor_epoca", metric_column: "mejor_valor_history"}
    )
    return best_rows.reset_index(drop=True)


def is_metric_column_for_long_table(column_name: str) -> bool:
    normalized = normalize_text(column_name)

    if normalized in TABLEAU_METRIC_COLUMN_HINTS:
        return True

    if normalized.startswith(("train_", "val_", "test_")) and any(
        hint in normalized for hint in ["mae", "rmse", "mse", "mape", "r2", "loss"]
    ):
        return True

    return False


def build_long_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    id_columns = [
        column
        for column in [
            "modelo",
            "escenario",
            "particion",
            "experimento",
            "mejor_epoca",
            "modo_target_entrenamiento",
            "archivo_origen",
            "nombre_archivo",
        ]
        if column in metrics_df.columns
    ]

    value_columns = [
        column
        for column in metrics_df.columns
        if column not in id_columns and is_metric_column_for_long_table(column)
    ]
    if not value_columns:
        return pd.DataFrame()

    long_df = metrics_df.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="metrica",
        value_name="valor_metrica",
    )
    long_df = long_df.dropna(subset=["valor_metrica"]).reset_index(drop=True)
    return long_df


def build_clean_long_metrics_table(
    metrics_df: pd.DataFrame, report: ProcessingReport
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    current = metrics_df.copy()

    if "metrica" in current.columns and "valor_metrica" in current.columns:
        current["metrica"] = current["metrica"].map(
            lambda value: normalize_metric_value(value) if pd.notna(value) else value
        )
        current["valor_metrica"] = pd.to_numeric(
            current["valor_metrica"].map(parse_mixed_numeric),
            errors="coerce",
        )

        if "particion" in current.columns:
            current["particion"] = current["particion"].map(
                lambda value: (
                    normalize_partition_value(value) if pd.notna(value) else value
                )
            )

        keep_columns = [
            column
            for column in [
                "modelo",
                "escenario",
                "particion",
                "mejor_epoca",
                "modo_target_entrenamiento",
                "archivo_origen",
                "nombre_archivo",
                "metrica",
                "valor_metrica",
            ]
            if column in current.columns
        ]

        current = current[keep_columns].copy()
        required = [
            column
            for column in [
                "modelo",
                "escenario",
                "particion",
                "metrica",
                "valor_metrica",
            ]
            if column in current.columns
        ]
        current = current.dropna(subset=required)
        current = remove_duplicate_rows(current, report, "tabla_metricas_larga")
        return current.reset_index(drop=True)

    report.warn(
        "La tabla de métricas no vino en formato largo (metrica/valor_metrica). Se reconstruirá desde columnas numéricas."
    )

    reconstructed = build_long_metrics_table(current)
    if reconstructed.empty:
        return reconstructed

    reconstructed["metrica"] = reconstructed["metrica"].map(
        lambda value: normalize_metric_value(value) if pd.notna(value) else value
    )
    reconstructed["valor_metrica"] = pd.to_numeric(
        reconstructed["valor_metrica"].map(parse_mixed_numeric),
        errors="coerce",
    )

    if "particion" in reconstructed.columns:
        reconstructed["particion"] = reconstructed["particion"].map(
            lambda value: normalize_partition_value(value) if pd.notna(value) else value
        )

    reconstructed = reconstructed.dropna(
        subset=["modelo", "escenario", "particion", "metrica", "valor_metrica"]
    )
    reconstructed = remove_duplicate_rows(reconstructed, report, "tabla_metricas_larga")
    return reconstructed.reset_index(drop=True)


def aggregate_long_metrics_to_wide(
    long_df: pd.DataFrame, report: ProcessingReport
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    key_columns = [
        column
        for column in ["modelo", "escenario", "particion"]
        if column in long_df.columns
    ]
    if not key_columns:
        report.warn(
            "No se pudo construir tabla_metricas_modelo_escenario porque faltan claves base."
        )
        return pd.DataFrame()

    context_columns = [
        column
        for column in [
            "mejor_epoca",
            "modo_target_entrenamiento",
            "archivo_origen",
            "nombre_archivo",
        ]
        if column in long_df.columns
    ]

    if context_columns:
        base_context = (
            long_df[key_columns + context_columns]
            .groupby(key_columns, dropna=False, as_index=False)
            .agg({column: "first" for column in context_columns})
        )
    else:
        base_context = long_df[key_columns].drop_duplicates().reset_index(drop=True)

    wide_metrics = long_df.pivot_table(
        index=key_columns,
        columns="metrica",
        values="valor_metrica",
        aggfunc="mean",
    ).reset_index()

    wide_metrics.columns = [
        normalize_text(column) if isinstance(column, str) else column
        for column in wide_metrics.columns
    ]

    result = base_context.merge(wide_metrics, on=key_columns, how="outer")
    result = normalize_columns(result)
    result = coerce_numeric_columns(result)
    result = trim_object_columns(result)
    result = remove_duplicate_rows(result, report, "tabla_metricas_modelo_escenario")
    return result.reset_index(drop=True)


def safe_left_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    keys: list[str],
    suffix_name: str,
) -> pd.DataFrame:
    if left_df.empty or right_df.empty:
        return left_df

    right_columns_to_keep = keys.copy()
    for column in right_df.columns:
        if column in keys:
            continue
        if column not in left_df.columns:
            right_columns_to_keep.append(column)
        else:
            right_columns_to_keep.append(column)

    right_selected = right_df[right_columns_to_keep].copy()

    overlapping_non_keys = [
        column
        for column in right_selected.columns
        if column in left_df.columns and column not in keys
    ]
    rename_map = {column: f"{column}_{suffix_name}" for column in overlapping_non_keys}
    right_selected = right_selected.rename(columns=rename_map)

    return left_df.merge(right_selected, on=keys, how="left")


def merge_summary_sources(
    metrics_agg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    history_best_df: pd.DataFrame,
    report: ProcessingReport,
) -> pd.DataFrame:
    base = metrics_agg_df.copy() if not metrics_agg_df.empty else pd.DataFrame()

    if base.empty and not summary_df.empty:
        base = summary_df.copy()

    if base.empty and not history_best_df.empty:
        base = history_best_df.copy()

    if base.empty:
        report.warn(
            "No se pudo construir la tabla resumen general porque no hubo métricas ni resúmenes utilizables."
        )
        return pd.DataFrame()

    key_columns = [
        column
        for column in ["modelo", "escenario", "particion"]
        if column in base.columns
    ]
    if not key_columns:
        key_columns = [
            column for column in ["modelo", "escenario"] if column in base.columns
        ]

    if not summary_df.empty and key_columns:
        summary_work = remove_duplicate_rows(
            summary_df.copy(), report, "summary_fuente"
        )
        available_keys = [
            column for column in key_columns if column in summary_work.columns
        ]
        if available_keys:
            summary_work = summary_work.drop_duplicates(
                subset=available_keys
            ).reset_index(drop=True)
            base = safe_left_merge(base, summary_work, available_keys, "resumen")

    if not history_best_df.empty:
        history_keys = [
            column
            for column in ["modelo", "escenario"]
            if column in base.columns and column in history_best_df.columns
        ]
        if history_keys:
            history_work = (
                history_best_df.copy()
                .drop_duplicates(subset=history_keys)
                .reset_index(drop=True)
            )
            base = safe_left_merge(base, history_work, history_keys, "history")

            if (
                "mejor_epoca" not in base.columns
                and "mejor_epoca_history" in base.columns
            ):
                base["mejor_epoca"] = base["mejor_epoca_history"]
            elif (
                "mejor_epoca" in base.columns and "mejor_epoca_history" in base.columns
            ):
                base["mejor_epoca"] = base["mejor_epoca"].fillna(
                    base["mejor_epoca_history"]
                )

    base = normalize_columns(base)
    base = coerce_numeric_columns(base)
    base = trim_object_columns(base)
    base = remove_duplicate_rows(base, report, "tabla_resumen_general")
    return base


def detect_metric_for_ranking(df: pd.DataFrame) -> tuple[str | None, str | None]:
    normalized_map = {normalize_text(column): column for column in df.columns}
    for candidate in METRIC_PRIORITY_FOR_RANKING:
        if candidate in normalized_map:
            return normalized_map[candidate], candidate
    return None, None


def build_ranking_table(
    summary_df: pd.DataFrame, report: ProcessingReport
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    metric_column, metric_name = detect_metric_for_ranking(summary_df)
    if metric_column is None:
        report.warn(
            "No se pudo construir el ranking porque no se encontró una métrica prioritaria como test_rmse, val_rmse, rmse o mae."
        )
        return pd.DataFrame()

    work = summary_df.copy()
    work[metric_column] = pd.to_numeric(work[metric_column], errors="coerce")
    work = work.dropna(subset=[metric_column])
    if work.empty:
        report.warn(
            "El ranking no se generó porque la métrica elegida quedó vacía después de convertir a número."
        )
        return pd.DataFrame()

    rank_group = [
        column for column in ["escenario", "particion"] if column in work.columns
    ]
    if not rank_group:
        rank_group = ["escenario"] if "escenario" in work.columns else []

    if not rank_group:
        report.warn(
            "No se pudo construir el ranking porque no existe la columna 'escenario'."
        )
        return pd.DataFrame()

    normalized_metric_name = normalize_text(metric_name)
    ascending = "r2" not in normalized_metric_name

    work["metrica_ranking"] = metric_name
    work["valor_metrica_ranking"] = work[metric_column]
    work["ranking_modelo"] = work.groupby(rank_group, dropna=False)[metric_column].rank(
        method="dense",
        ascending=ascending,
    )
    work = work.sort_values(
        rank_group + ["ranking_modelo", metric_column, "modelo"], na_position="last"
    )

    selected_columns = [
        column
        for column in [
            "modelo",
            "escenario",
            "particion",
            "metrica_ranking",
            "valor_metrica_ranking",
            "ranking_modelo",
            metric_column,
            "mejor_epoca",
            "archivo_origen",
        ]
        if column in work.columns
    ]
    return work[selected_columns].reset_index(drop=True)


def build_baseline_vs_models_table(
    summary_df: pd.DataFrame, report: ProcessingReport
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    work = summary_df.copy()
    if "modelo" not in work.columns or "escenario" not in work.columns:
        report.warn(
            "No se pudo construir baseline_vs_modelos porque faltan columnas como 'modelo' o 'escenario'."
        )
        return pd.DataFrame()

    baseline_df = work[work["modelo"] == "baseline_persistencia"].copy()
    others_df = work[work["modelo"] != "baseline_persistencia"].copy()

    if baseline_df.empty:
        report.warn(
            "No se encontró baseline_persistencia en los resultados, por eso no se generó baseline_vs_modelos."
        )
        return pd.DataFrame()

    if others_df.empty:
        report.warn(
            "Solo se encontró baseline_persistencia, por eso no se generó baseline_vs_modelos."
        )
        return pd.DataFrame()

    join_keys = [
        column for column in ["escenario", "particion"] if column in work.columns
    ]
    if not join_keys:
        join_keys = ["escenario"]

    selected_metric_pairs: list[tuple[str, str]] = []
    for metric_name in [
        "test_rmse",
        "test_mae",
        "val_rmse",
        "val_mae",
        "rmse",
        "mae",
        "mape",
        "r2",
    ]:
        if metric_name in work.columns:
            selected_metric_pairs.append((metric_name, f"baseline_{metric_name}"))

    if not selected_metric_pairs:
        report.warn(
            "No se encontraron métricas comparables para construir baseline_vs_modelos."
        )
        return pd.DataFrame()

    baseline_columns = join_keys + [metric for metric, _ in selected_metric_pairs]
    baseline_selected = baseline_df[baseline_columns].copy().drop_duplicates()
    baseline_selected = baseline_selected.rename(
        columns={source: target for source, target in selected_metric_pairs}
    )

    model_columns = [column for column in work.columns if column != "archivo_origen"]
    merged = others_df[model_columns].merge(baseline_selected, on=join_keys, how="left")

    for source, target in selected_metric_pairs:
        delta_abs = f"delta_{source}_abs"
        delta_pct = f"delta_{source}_pct"

        if source in merged.columns and target in merged.columns:
            merged[delta_abs] = merged[source] - merged[target]
            merged[delta_pct] = np.where(
                merged[target].replace(0, np.nan).notna(),
                (merged[source] - merged[target]) / merged[target] * 100.0,
                np.nan,
            )

    metric_column, metric_name = detect_metric_for_ranking(merged)
    if metric_column is not None:
        baseline_metric_column = f"baseline_{metric_column}"
        if baseline_metric_column in merged.columns:
            better_when_lower = "r2" not in normalize_text(metric_name)

            if better_when_lower:
                merged["mejora_frente_baseline"] = np.where(
                    merged[baseline_metric_column].replace(0, np.nan).notna(),
                    (merged[baseline_metric_column] - merged[metric_column])
                    / merged[baseline_metric_column]
                    * 100.0,
                    np.nan,
                )
                merged["supera_baseline"] = (
                    merged[metric_column] < merged[baseline_metric_column]
                )
            else:
                merged["mejora_frente_baseline"] = np.where(
                    merged[baseline_metric_column].replace(0, np.nan).notna(),
                    (merged[metric_column] - merged[baseline_metric_column])
                    / merged[baseline_metric_column].abs()
                    * 100.0,
                    np.nan,
                )
                merged["supera_baseline"] = (
                    merged[metric_column] > merged[baseline_metric_column]
                )

            merged["metrica_comparacion_baseline"] = metric_column

    merged = remove_duplicate_rows(merged, report, "tabla_baseline_vs_modelos")
    return merged.reset_index(drop=True)


def validate_tableau_compatibility(
    df: pd.DataFrame, table_name: str, report: ProcessingReport
) -> None:
    if df.empty:
        return

    nested_columns = []
    for column in df.columns:
        if df[column].map(is_nested_value).fillna(False).any():
            nested_columns.append(column)

    if nested_columns:
        report.warn(
            f"La tabla '{table_name}' todavía tiene columnas anidadas no compatibles con Tableau: {nested_columns}"
        )

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        report.warn(
            f"La tabla '{table_name}' tiene columnas duplicadas, revisa la normalización: {duplicated_columns}"
        )


def export_csv(df: pd.DataFrame, export_path: Path, encoding: str, sep: str) -> int:
    export_path.parent.mkdir(parents=True, exist_ok=True)

    current = df.copy()
    current = trim_object_columns(current)

    for column in current.columns:
        if current[column].map(is_nested_value).fillna(False).any():
            current[column] = current[column].map(
                lambda value: (
                    json.dumps(value, ensure_ascii=False)
                    if is_nested_value(value)
                    else value
                )
            )

    current.to_csv(
        export_path,
        index=False,
        encoding=encoding,
        sep=sep,
        quoting=csv.QUOTE_ALL,
    )
    return len(current)


def export_excel_optional(
    df: pd.DataFrame, export_path: Path, report: ProcessingReport
) -> None:
    try:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        current = df.copy()
        current = trim_object_columns(current)

        with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
            current.to_excel(writer, index=False, sheet_name="datos")

            worksheet = writer.sheets["datos"]

            excel_number_format_en_us = "[$-en-US]#,##0.################"

            for col_idx, column_name in enumerate(current.columns, start=1):
                if pd.api.types.is_numeric_dtype(current[column_name]):
                    for row_idx in range(2, len(current) + 2):
                        worksheet.cell(row=row_idx, column=col_idx).number_format = (
                            excel_number_format_en_us
                        )

    except Exception as error:
        report.warn(
            f"No se pudo exportar '{export_path.name}' en formato Excel. No es crítico. Detalle: {error}"
        )


def export_json(data: dict[str, Any], export_path: Path, encoding: str) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding=encoding) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def build_manifest(
    report: ProcessingReport, exported_tables: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    return {
        "archivos_encontrados": report.archivos_encontrados,
        "tablas_exportadas": report.tablas_exportadas,
        "advertencias": report.advertencias,
        "errores": report.errores,
        "resumen_filas": {name: int(len(df)) for name, df in exported_tables.items()},
        "columnas_por_tabla": {
            name: list(df.columns) for name, df in exported_tables.items()
        },
    }


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    outputs_dir = (
        (project_root / args.outputs_dir).resolve()
        if not Path(args.outputs_dir).is_absolute()
        else Path(args.outputs_dir).resolve()
    )
    dashboard_dir = (
        (project_root / args.dashboard_dir).resolve()
        if not Path(args.dashboard_dir).is_absolute()
        else Path(args.dashboard_dir).resolve()
    )

    report = ProcessingReport()

    print_title("SEGMENTACION DE RESULTADOS PARA DASHBOARD TABLEAU")
    print(f"project_root   = {project_root}")
    print(f"outputs_dir    = {outputs_dir}")
    print(f"dashboard_dir  = {dashboard_dir}")
    print(f"encoding_csv   = {args.encoding}")
    print(f"separador_csv  = {args.sep}")
    print(f"export_excel   = {args.export_excel}")

    print_title("1) BUSQUEDA DE ARCHIVOS")
    categories = scan_candidate_files(
        outputs_dir, report, excluded_dirs=[dashboard_dir]
    )
    for category, paths in categories.items():
        print(f"{category:24s}: {len(paths)} archivos")

    print_title("2) LECTURA Y NORMALIZACION")
    metrics_tables = load_category_tables(categories["metrics"], "metrics", report)
    validation_tables = load_category_tables(
        categories["predictions_validation"], "predictions_validation", report
    )
    test_tables = load_category_tables(
        categories["predictions_test"], "predictions_test", report
    )
    history_tables = load_category_tables(categories["history"], "history", report)
    summary_tables = load_category_tables(categories["summary"], "summary", report)

    metrics_df = concatenate_tables(metrics_tables, report, "metrics")
    validation_df = concatenate_tables(
        validation_tables, report, "predicciones_validation"
    )
    test_df = concatenate_tables(test_tables, report, "predicciones_test")
    history_df = concatenate_tables(history_tables, report, "history")
    summary_df = concatenate_tables(summary_tables, report, "summary")

    print(f"metricas consolidadas         : {len(metrics_df)} filas")
    print(f"predicciones validation      : {len(validation_df)} filas")
    print(f"predicciones test            : {len(test_df)} filas")
    print(f"history por epocas           : {len(history_df)} filas")
    print(f"resumenes consolidados       : {len(summary_df)} filas")

    print_title("3) CONSTRUCCION DE TABLAS FINALES")
    tabla_metricas_larga = build_clean_long_metrics_table(metrics_df, report)
    tabla_metricas_modelo_escenario = aggregate_long_metrics_to_wide(
        tabla_metricas_larga, report
    )

    if tabla_metricas_larga.empty and not tabla_metricas_modelo_escenario.empty:
        tabla_metricas_larga = build_long_metrics_table(tabla_metricas_modelo_escenario)

    tabla_history_mejor_epoca = build_history_best_epoch_summary(history_df, report)

    tabla_resumen_general = merge_summary_sources(
        metrics_agg_df=tabla_metricas_modelo_escenario,
        summary_df=summary_df,
        history_best_df=tabla_history_mejor_epoca,
        report=report,
    )

    tabla_ranking = build_ranking_table(tabla_resumen_general, report)
    tabla_baseline_vs_modelos = build_baseline_vs_models_table(
        tabla_resumen_general, report
    )

    exported_tables: dict[str, pd.DataFrame] = {
        "tabla_metricas_modelo_escenario": tabla_metricas_modelo_escenario,
        "tabla_metricas_larga": tabla_metricas_larga,
        "tabla_predicciones_validation": validation_df,
        "tabla_predicciones_test": test_df,
        "tabla_history_epocas": history_df,
        "tabla_history_mejor_epoca": tabla_history_mejor_epoca,
        "tabla_resumen_general": tabla_resumen_general,
        "tabla_baseline_vs_modelos": tabla_baseline_vs_modelos,
        "tabla_ranking_modelos_por_escenario": tabla_ranking,
    }

    for table_name, df in exported_tables.items():
        validate_tableau_compatibility(df, table_name, report)
        print(f"{table_name:40s}: {len(df)} filas")

    print_title("4) EXPORTACION FINAL")
    for table_name, df in exported_tables.items():
        if df.empty:
            report.warn(f"La tabla '{table_name}' quedó vacía y no se exportará.")
            continue

        export_path_csv = dashboard_dir / f"{table_name}.csv"
        rows = export_csv(df, export_path_csv, encoding=args.encoding, sep=args.sep)
        report.tablas_exportadas[table_name] = rows
        print(f"Exportado CSV : {export_path_csv}  | filas = {rows}")

        if args.export_excel:
            export_path_xlsx = dashboard_dir / f"{table_name}.xlsx"
            export_excel_optional(df, export_path_xlsx, report)

    manifest = build_manifest(report, exported_tables)
    manifest_path = dashboard_dir / "dashboard_manifest.json"
    export_json(manifest, manifest_path, encoding="utf-8")
    print(f"Exportado     : {manifest_path}")

    print_title("5) RESUMEN FINAL")
    print("Archivos encontrados por categoria:")
    for category, count in report.archivos_encontrados.items():
        print(f"- {category}: {count}")

    print("\nTablas exportadas:")
    if report.tablas_exportadas:
        for table_name, count in report.tablas_exportadas.items():
            print(f"- {table_name}: {count} filas")
    else:
        print("- No se exportó ninguna tabla.")

    print("\nAdvertencias detectadas:")
    if report.advertencias:
        for warning in report.advertencias:
            print(f"- {warning}")
    else:
        print("- No hubo advertencias.")

    print("\nErrores detectados:")
    if report.errores:
        for error in report.errores:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("- No hubo errores críticos.")

    print("\nProceso terminado correctamente.")
    print(f"Revisa la carpeta final: {dashboard_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR CRITICO] El script terminó con una excepción no controlada.")
        print(str(exc))
        print(traceback.format_exc())
        sys.exit(1)
