from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import hashlib
import importlib.util
import json
import math
import shutil
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
COMMON_PATH = SRC / "10_common_training_pipeline.py"

SOURCE_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "1h"
    / "global"
    / "btcusdt_spot_1h_model_input_2019_2026.parquet"
)

SENS_DATA_DIR = PROJECT_ROOT / "data" / "1h" / "logreturn_sensitivity"
SENS_DATASET = (
    SENS_DATA_DIR
    / "btcusdt_spot_1h_logreturn_features_2019_2026.parquet"
)
SENS_DATA_MANIFEST = SENS_DATA_DIR / "logreturn_feature_manifest.json"

SENS_ROOT = PROJECT_ROOT / "logreturn_sensitivity"
PILOT_ROOT = SENS_ROOT / "pilot_global_seed42"
FULL_ROOT = SENS_ROOT / "full"

STRICT_MARKER = "STRICT_HOURLY_CONTINUITY_V1"
EXPECTED_CUBLAS = ":4096:8"
EXPECTED_GLOBAL_TEST_N = 9358

MODEL_SCRIPT_NAMES = {
    "lstm": "12_entrenar_lstm.py",
    "gru": "13_entrenar_gru.py",
    "cnn1d": "14_entrenar_cnn1d.py",
    "cnn_lstm": "15_entrenar_cnn_lstm.py",
}

MODEL_ORDER = [
    "baseline_persistencia",
    "lstm",
    "gru",
    "cnn1d",
    "cnn_lstm",
]

NEURAL_MODELS = ["lstm", "gru", "cnn1d", "cnn_lstm"]

LOGRETURN_FEATURES_22 = [
    "lr_log_return_1h",
    "lr_log_high_low_range",
    "lr_log_close_open_body",
    "lr_abs_log_close_open_body",
    "lr_mean_log_return_24h",
    "lr_vol_log_return_24h",
    "lr_mean_abs_log_return_24h",
    "lr_volume_ratio_24h",
    "lr_quote_volume_ratio_24h",
    "lr_trades_ratio_24h",
    "lr_taker_buy_base_share",
    "lr_taker_buy_quote_share",
    "lr_volume_log_change_1h",
    "lr_quote_volume_log_change_1h",
    "lr_trades_log_change_1h",
    "lr_hour_sin",
    "lr_hour_cos",
    "lr_dow_sin",
    "lr_dow_cos",
    "lr_month_sin",
    "lr_month_cos",
    "lr_is_weekend",
]

assert len(LOGRETURN_FEATURES_22) == 22


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [
        "open_datetime_utc",
        "datetime_utc",
        "open_time_utc",
        "timestamp_utc",
        "datetime",
        "timestamp",
        "open_time",
        "date",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lower = c.lower()
        if "time" in lower or "date" in lower:
            return c
    raise KeyError("No se pudo detectar la columna temporal.")


def _safe_log_ratio(num: pd.Series, den: pd.Series, eps: float = 1e-12) -> pd.Series:
    a = pd.to_numeric(num, errors="raise").astype(float)
    b = pd.to_numeric(den, errors="raise").astype(float)
    if (a < 0).any() or (b < 0).any():
        raise ValueError("Se detectaron valores negativos en una razón logarítmica.")
    return np.log((a + eps) / (b + eps))


def build_logreturn_dataset(force: bool = False) -> Dict[str, Any]:
    if not SOURCE_PARQUET.exists():
        raise FileNotFoundError(SOURCE_PARQUET)

    if SENS_DATASET.exists() and SENS_DATA_MANIFEST.exists() and not force:
        manifest = json.loads(SENS_DATA_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") == "READY":
            return manifest

    SENS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SOURCE_PARQUET)
    source_rows = int(len(df))
    ts_col = detect_timestamp_column(df)

    required = [
        ts_col,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Hay timestamps inválidos/NaT.")

    order = np.argsort(ts.to_numpy())
    out = out.iloc[order].reset_index(drop=True)
    ts = pd.to_datetime(out[ts_col], utc=True, errors="raise")

    # Segmentos estrictamente horarios. Las estadísticas móviles nunca atraviesan gaps.
    step_hours = ts.diff().dt.total_seconds().div(3600.0)
    is_contiguous = step_hours.eq(1.0)
    segment_id = (~is_contiguous).cumsum()

    close = pd.to_numeric(out["close"], errors="raise").astype(float)
    open_ = pd.to_numeric(out["open"], errors="raise").astype(float)
    high = pd.to_numeric(out["high"], errors="raise").astype(float)
    low = pd.to_numeric(out["low"], errors="raise").astype(float)
    volume = pd.to_numeric(out["volume"], errors="raise").astype(float)
    quote_volume = pd.to_numeric(out["quote_asset_volume"], errors="raise").astype(float)
    trades = pd.to_numeric(out["number_of_trades"], errors="raise").astype(float)
    taker_base = pd.to_numeric(out["taker_buy_base_volume"], errors="raise").astype(float)
    taker_quote = pd.to_numeric(out["taker_buy_quote_volume"], errors="raise").astype(float)

    for label, series in {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }.items():
        if (series <= 0).any():
            raise ValueError(f"{label} contiene valores <= 0; no se puede usar logaritmo.")

    def previous_within_segment(series: pd.Series) -> pd.Series:
        prev = series.groupby(segment_id, sort=False).shift(1)
        return prev.fillna(series)

    prev_close = previous_within_segment(close)
    prev_volume = previous_within_segment(volume)
    prev_quote = previous_within_segment(quote_volume)
    prev_trades = previous_within_segment(trades)

    lr = np.log(close / prev_close)

    def rolling_mean(series: pd.Series, window: int = 24) -> pd.Series:
        return (
            series.groupby(segment_id, sort=False)
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
            .astype(float)
        )

    def rolling_std(series: pd.Series, window: int = 24) -> pd.Series:
        return (
            series.groupby(segment_id, sort=False)
            .transform(lambda s: s.rolling(window, min_periods=1).std(ddof=0))
            .fillna(0.0)
            .astype(float)
        )

    eps = 1e-12
    vol_ma24 = rolling_mean(volume)
    quote_ma24 = rolling_mean(quote_volume)
    trades_ma24 = rolling_mean(trades)

    out["lr_log_return_1h"] = lr
    out["lr_log_high_low_range"] = np.log(high / low)
    out["lr_log_close_open_body"] = np.log(close / open_)
    out["lr_abs_log_close_open_body"] = np.abs(out["lr_log_close_open_body"])
    out["lr_mean_log_return_24h"] = rolling_mean(pd.Series(lr, index=out.index))
    out["lr_vol_log_return_24h"] = rolling_std(pd.Series(lr, index=out.index))
    out["lr_mean_abs_log_return_24h"] = rolling_mean(
        pd.Series(np.abs(lr), index=out.index)
    )
    out["lr_volume_ratio_24h"] = volume / np.maximum(vol_ma24, eps) - 1.0
    out["lr_quote_volume_ratio_24h"] = quote_volume / np.maximum(quote_ma24, eps) - 1.0
    out["lr_trades_ratio_24h"] = trades / np.maximum(trades_ma24, eps) - 1.0
    out["lr_taker_buy_base_share"] = taker_base / np.maximum(volume, eps)
    out["lr_taker_buy_quote_share"] = taker_quote / np.maximum(quote_volume, eps)
    out["lr_volume_log_change_1h"] = _safe_log_ratio(volume, prev_volume)
    out["lr_quote_volume_log_change_1h"] = _safe_log_ratio(quote_volume, prev_quote)
    out["lr_trades_log_change_1h"] = _safe_log_ratio(trades, prev_trades)

    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    month0 = (ts.dt.month.astype(float) - 1.0)

    out["lr_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["lr_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["lr_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["lr_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["lr_month_sin"] = np.sin(2.0 * np.pi * month0 / 12.0)
    out["lr_month_cos"] = np.cos(2.0 * np.pi * month0 / 12.0)
    out["lr_is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.float32)

    feature_matrix = out[LOGRETURN_FEATURES_22].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_matrix).all():
        bad = np.argwhere(~np.isfinite(feature_matrix))
        raise ValueError(f"Features log-return contienen NaN/Inf; primeras posiciones: {bad[:10]}")

    if len(out) != source_rows:
        raise AssertionError("La vista log-return alteró el número de filas fuente.")

    out.to_parquet(SENS_DATASET, index=False)

    gaps = int((~is_contiguous.iloc[1:]).sum())
    manifest = {
        "status": "READY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet": str(SOURCE_PARQUET),
        "output_parquet": str(SENS_DATASET),
        "source_rows": source_rows,
        "output_rows": int(len(out)),
        "timestamp_column": ts_col,
        "strict_gap_boundaries_detected": gaps,
        "feature_count": len(LOGRETURN_FEATURES_22),
        "features": LOGRETURN_FEATURES_22,
        "feature_design": {
            "raw_price_levels_used_as_predictors": False,
            "year_used_as_predictor": False,
            "cyclic_calendar_encoding": True,
            "rolling_statistics_cross_gaps": False,
            "boundary_log_changes_are_zero_by_construction": True,
            "target_definition": "log(P_t+1 / P_t), applied later by in-memory target patch",
        },
        "source_sha256": sha256_file(SOURCE_PARQUET),
        "output_sha256": sha256_file(SENS_DATASET),
    }
    SENS_DATA_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def load_official_common() -> Any:
    if not COMMON_PATH.exists():
        raise FileNotFoundError(COMMON_PATH)
    text = COMMON_PATH.read_text(encoding="utf-8")
    if STRICT_MARKER not in text:
        raise RuntimeError(
            "El pipeline oficial no contiene el filtro estricto validado. "
            "No se ejecuta la sensibilidad log-return."
        )

    spec = importlib.util.spec_from_file_location(
        "common_logreturn_sensitivity_runtime",
        COMMON_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["common_logreturn_sensitivity_runtime"] = module
    spec.loader.exec_module(module)
    patch_common_for_logreturn(module)
    return module


def patch_common_for_logreturn(common: Any) -> None:
    def validate_target_mode(mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "logreturn": "log_return",
            "log_returns": "log_return",
            "return": "log_return",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"price", "delta", "log_return"}:
            raise ValueError(
                f"Modo de target no soportado: {mode}. "
                "Usa 'price', 'delta' o 'log_return'."
            )
        return normalized

    def build_train_target(
        future_price: float,
        baseline_price: float,
        target_mode: str,
    ) -> float:
        mode = validate_target_mode(target_mode)
        if mode == "price":
            return float(future_price)
        if mode == "delta":
            return float(future_price - baseline_price)
        if future_price <= 0 or baseline_price <= 0:
            raise ValueError("log_return requiere precios estrictamente positivos.")
        return float(np.log(future_price / baseline_price))

    def reconstruct_price_predictions(
        pred_target_raw: np.ndarray,
        baseline_raw: np.ndarray,
        target_mode: str,
    ) -> np.ndarray:
        pred = np.asarray(pred_target_raw, dtype=np.float64).reshape(-1)
        baseline = np.asarray(baseline_raw, dtype=np.float64).reshape(-1)
        mode = validate_target_mode(target_mode)
        if mode == "price":
            result = pred
        elif mode == "delta":
            result = baseline + pred
        else:
            result = baseline * np.exp(pred)
        if not np.isfinite(result).all():
            raise FloatingPointError(
                "La reconstrucción desde log-return produjo NaN/Inf."
            )
        return result

    def translate_target_mode_label(mode: str) -> str:
        m = validate_target_mode(mode)
        if m == "price":
            return "precio"
        if m == "delta":
            return "delta"
        return "log_return"

    def deterministic_get_device() -> torch.device:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            return torch.device("cuda")
        return torch.device("cpu")

    common.validate_target_mode = validate_target_mode
    common.build_train_target = build_train_target
    common.reconstruct_price_predictions = reconstruct_price_predictions
    common.translate_target_mode_label = translate_target_mode_label
    common.get_device = deterministic_get_device


def load_model_builders() -> Dict[str, Any]:
    builders: Dict[str, Any] = {}
    for model, filename in MODEL_SCRIPT_NAMES.items():
        path = SRC / filename
        if not path.exists():
            raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location(
            f"logreturn_builder_{model}",
            path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"logreturn_builder_{model}"] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, "model_builder"):
            raise AttributeError(f"{filename} no expone model_builder().")
        builders[model] = mod.model_builder
    return builders


def configure_determinism(require_cuda: bool = True) -> torch.device:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Se cancela para mantener comparabilidad "
            "con el benchmark definitivo."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scenario_catalog_with_logreturn_data(common: Any) -> Dict[str, Any]:
    catalog = common.build_default_scenarios(project_root=PROJECT_ROOT)
    patched: Dict[str, Any] = {}
    for key, scenario in catalog.items():
        sc = deepcopy(scenario)
        if getattr(sc, "source_path", None) is not None:
            sc.source_path = SENS_DATASET
        # Los escenarios actuales usan source_path global + filtros. Si en una
        # versión futura aparecen train_path/test_path directos, no se adivinan.
        if getattr(sc, "train_path", None) is not None or getattr(sc, "test_path", None) is not None:
            raise RuntimeError(
                f"{key}: usa train_path/test_path directos; requiere adaptación explícita."
            )
        patched[key] = sc
    return patched


def make_config(
    common: Any,
    experiment_root: Path,
    seed: int,
    model_subdir: str,
) -> Any:
    return common.ExperimentConfig(
        project_root=experiment_root,
        lookback=48,
        horizon=1,
        batch_size=64,
        epochs=50,
        patience=5,
        learning_rate=0.001,
        num_workers=0,
        random_seed=int(seed),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_column="close",
        timestamp_column=None,
        feature_columns=list(LOGRETURN_FEATURES_22),
        model_subdir=model_subdir,
        save_best_weights=True,
        verbose=True,
    )


def prepare_clean_root(path: Path, allow_backup: bool = False) -> Path | None:
    backup = None
    if path.exists():
        if not allow_backup:
            raise FileExistsError(
                f"Ya existe la carpeta experimental:\n{path}\n"
                "No se sobrescribe automáticamente."
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(path.name + f"__BACKUP_{stamp}")
        path.rename(backup)
    path.mkdir(parents=True, exist_ok=False)
    return backup


def run_persistence(
    common: Any,
    scenario: Any,
    experiment_root: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    cfg = make_config(
        common,
        experiment_root=experiment_root,
        seed=seed,
        model_subdir="baseline_persistencia",
    )
    common.set_global_seed(seed)
    prepared = common.prepare_scenario(
        scenario=scenario,
        config=cfg,
        device=common.get_device(),
        target_mode="log_return",
    )
    return common.export_baseline_predictions(
        prepared=prepared,
        model_name="baseline_persistencia",
        config=cfg,
    )


def run_neural(
    common: Any,
    builder: Any,
    model_name: str,
    scenario: Any,
    experiment_root: Path,
    seed: int,
) -> Dict[str, Any]:
    cfg = make_config(
        common,
        experiment_root=experiment_root,
        seed=seed,
        model_subdir=model_name,
    )
    result = common.run_training_for_scenario(
        scenario=scenario,
        config=cfg,
        model_name=model_name,
        model_builder=builder,
        train_target_mode="log_return",
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def prediction_return_diagnostics(prediction_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(prediction_csv)
    required = ["valor_real", "valor_predicho", "valor_baseline"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{prediction_csv.name}: faltan {missing}")

    actual = df["valor_real"].to_numpy(dtype=np.float64)
    pred = df["valor_predicho"].to_numpy(dtype=np.float64)
    base = df["valor_baseline"].to_numpy(dtype=np.float64)

    if (actual <= 0).any() or (pred <= 0).any() or (base <= 0).any():
        raise ValueError("Se requieren precios > 0 para métricas log-return.")

    r_true = np.log(actual / base)
    r_pred = np.log(pred / base)

    err = r_true - r_pred
    persistence_err = r_true

    rmse_r = float(np.sqrt(np.mean(err ** 2)))
    mae_r = float(np.mean(np.abs(err)))
    rmse_r_pers = float(np.sqrt(np.mean(persistence_err ** 2)))
    mae_r_pers = float(np.mean(np.abs(persistence_err)))

    sse_model = float(np.sum(err ** 2))
    sse_pers = float(np.sum(persistence_err ** 2))
    r2_oos = float(1.0 - sse_model / sse_pers) if sse_pers > 0 else float("nan")

    nonzero = np.abs(r_true) > 0
    if nonzero.any():
        directional = float(np.mean(np.sign(r_pred[nonzero]) == np.sign(r_true[nonzero])))
    else:
        directional = float("nan")

    corr = (
        float(np.corrcoef(r_pred, r_true)[0, 1])
        if np.std(r_pred) > 0 and np.std(r_true) > 0
        else float("nan")
    )

    return {
        "n": int(len(df)),
        "return_MAE": mae_r,
        "return_RMSE": rmse_r,
        "persistence_return_MAE": mae_r_pers,
        "persistence_return_RMSE": rmse_r_pers,
        "return_RMSE_ratio_vs_persistence": (
            rmse_r / rmse_r_pers if rmse_r_pers > 0 else float("nan")
        ),
        "return_skill_vs_persistence": (
            1.0 - rmse_r / rmse_r_pers if rmse_r_pers > 0 else float("nan")
        ),
        "R2_oos_vs_zero_return": r2_oos,
        "directional_accuracy": directional,
        "corr_predicted_vs_realized_log_return": corr,
        "mean_predicted_log_return": float(np.mean(r_pred)),
        "sd_predicted_log_return": float(np.std(r_pred, ddof=1)),
    }


def price_metrics_from_prediction(prediction_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(prediction_csv)
    y = df["valor_real"].to_numpy(dtype=np.float64)
    pred = df["valor_predicho"].to_numpy(dtype=np.float64)
    base = df["valor_baseline"].to_numpy(dtype=np.float64)

    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    rmse_base = float(np.sqrt(np.mean((y - base) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    mae_base = float(np.mean(np.abs(y - base)))

    return {
        "n": int(len(df)),
        "price_MAE": mae,
        "price_RMSE": rmse,
        "persistence_price_MAE": mae_base,
        "persistence_price_RMSE": rmse_base,
        "relative_RMSE": rmse / rmse_base if rmse_base > 0 else float("nan"),
        "RMSE_skill": 1.0 - rmse / rmse_base if rmse_base > 0 else float("nan"),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
