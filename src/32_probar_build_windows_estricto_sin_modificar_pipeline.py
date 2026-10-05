from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import torch


# ======================================================================================
# PRUEBA SEGURA DE build_windows() ESTRICTO
# ======================================================================================
#
# IMPORTANTE:
# - NO modifica src/10_common_training_pipeline.py
# - NO entrena modelos
# - NO sobrescribe métricas, predicciones ni checkpoints
# - Importa el pipeline oficial y sustituye build_windows SOLO EN MEMORIA
# - Ejecuta prepare_scenario() con los 13 escenarios
# - Compara ventanas originales vs estrictas
# - Audita que TODA ventana aceptada tenga:
#       * 48 filas horarias consecutivas
#       * target exactamente 1 hora después del último input
#
# Solo si este script termina con:
#
#     Estado estricto : PASS_STRICT_48H_TO_1H
#     Violaciones aceptadas : 0
#
# se debe considerar sustituir build_windows() en el pipeline oficial.
# ======================================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
COMMON_PATH = SRC_DIR / "10_common_training_pipeline.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "strict_window_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 48
HORIZON = 1
TARGET_MODE = "delta"
EXPECTED_STEP_NS = 3_600_000_000_000  # 1 hora en nanosegundos


def load_common():
    if not COMMON_PATH.exists():
        raise FileNotFoundError(
            f"No existe:\n{COMMON_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "common_strict_window_test",
        COMMON_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"No se pudo cargar:\n{COMMON_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["common_strict_window_test"] = module
    spec.loader.exec_module(module)
    return module


def timestamp_ns_array(
    df: pd.DataFrame,
    timestamp_column: str,
) -> np.ndarray:
    """
    Convierte timestamps UTC explícitamente a datetime64[ns] para evitar
    el problema de resolución temporal detectado con pandas 3.x.
    """
    ts = pd.to_datetime(
        df.loc[:, timestamp_column],
        utc=True,
        errors="coerce",
    )

    nat_count = int(ts.isna().sum())
    if nat_count:
        raise ValueError(
            f"Se detectaron {nat_count} timestamps NaT."
        )

    return (
        ts.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )


def strict_valid_start_mask(
    ts_ns: np.ndarray,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    """
    Devuelve una máscara booleana sobre TODOS los start_idx candidatos.

    Una muestra es válida únicamente si TODAS las transiciones desde la
    primera fila de input hasta la fila target son exactamente de +1 hora.

    Para lookback=48 y horizon=1:
      input: 48 filas -> 47 transiciones de 1 h
      target: 1 transición adicional de 1 h
      total: 48 transiciones exactas
    """
    n_rows = len(ts_ns)
    max_start = n_rows - lookback - horizon + 1

    if max_start <= 0:
        raise ValueError(
            f"No hay suficientes filas: {n_rows}"
        )

    step_ns = np.diff(ts_ns)
    good_step = step_ns == EXPECTED_STEP_NS

    # bad_prefix[k] = número de transiciones inválidas en step_ns[:k]
    bad_prefix = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum((~good_step).astype(np.int64)),
        ]
    )

    starts = np.arange(max_start, dtype=np.int64)
    target_idx = starts + lookback + horizon - 1

    # Las transiciones relevantes son:
    # step_ns[start_idx : target_idx]
    bad_count = (
        bad_prefix[target_idx]
        - bad_prefix[starts]
    )

    return bad_count == 0


def audit_accepted_samples(
    ts_ns: np.ndarray,
    valid_starts: np.ndarray,
    lookback: int,
    horizon: int,
) -> dict:
    """
    Verificación independiente posterior de las muestras aceptadas.
    """
    if valid_starts.size == 0:
        return {
            "accepted": 0,
            "violating_accepted": 0,
            "max_input_span_hours": None,
            "max_target_lag_hours": None,
            "max_total_span_hours": None,
        }

    last_input_idx = valid_starts + lookback - 1
    target_idx = valid_starts + lookback + horizon - 1

    input_span_h = (
        ts_ns[last_input_idx]
        - ts_ns[valid_starts]
    ) / EXPECTED_STEP_NS

    target_lag_h = (
        ts_ns[target_idx]
        - ts_ns[last_input_idx]
    ) / EXPECTED_STEP_NS

    total_span_h = (
        ts_ns[target_idx]
        - ts_ns[valid_starts]
    ) / EXPECTED_STEP_NS

    expected_input_span = lookback - 1
    expected_target_lag = horizon
    expected_total_span = lookback + horizon - 1

    strict_ok = (
        np.isclose(
            input_span_h,
            expected_input_span,
            rtol=0.0,
            atol=1e-12,
        )
        & np.isclose(
            target_lag_h,
            expected_target_lag,
            rtol=0.0,
            atol=1e-12,
        )
        & np.isclose(
            total_span_h,
            expected_total_span,
            rtol=0.0,
            atol=1e-12,
        )
    )

    return {
        "accepted": int(valid_starts.size),
        "violating_accepted": int((~strict_ok).sum()),
        "max_input_span_hours": float(
            np.max(input_span_h)
        ),
        "max_target_lag_hours": float(
            np.max(target_lag_h)
        ),
        "max_total_span_hours": float(
            np.max(total_span_h)
        ),
    }


def main() -> None:
    print("=" * 110)
    print("PRUEBA SEGURA - build_windows ESTRICTO - 13 ESCENARIOS")
    print("=" * 110)
    print("Entrenamiento: NO")
    print("Modificación del pipeline oficial: NO")
    print()

    common = load_common()

    config = common.ExperimentConfig(
        project_root=PROJECT_ROOT,
        lookback=LOOKBACK,
        horizon=HORIZON,
        random_seed=42,
        feature_columns=list(
            getattr(common, "CANONICAL_FEATURES_22")
        ),
    )

    if len(config.feature_columns or []) != 22:
        raise RuntimeError(
            "Se esperaban exactamente 22 features canónicas."
        )

    catalog = common.build_default_scenarios(
        project_root=PROJECT_ROOT
    )

    if len(catalog) != 13:
        raise RuntimeError(
            f"Se esperaban 13 escenarios; hay {len(catalog)}."
        )

    original_build_windows = common.build_windows

    # Se llena en cada llamada del build estricto.
    strict_call_records: list[dict] = []

    state = {
        "scenario_key": "",
        "scenario_name": "",
        "scenario_mode": "",
        "split_index": 0,
    }

    split_names = ["train", "val", "test"]

    def build_windows_strict_test(
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        timestamp_column: str,
        lookback: int,
        horizon: int,
        target_mode: str = "price",
    ):
        """
        COPIA SEGURA / EXPERIMENTAL de build_windows().

        Conserva exactamente:
        - features float32
        - target real
        - baseline = último close visible
        - target residual/price mediante common.build_train_target()
        - WindowedSplit oficial

        Único cambio:
        - DESCARTA cualquier start_idx cuya secuencia input→target
          contenga una transición temporal distinta de +1 hora.
        """

        split_index = int(state["split_index"])
        if split_index >= 3:
            raise RuntimeError(
                "prepare_scenario realizó más de tres llamadas "
                "a build_windows; revisar el pipeline."
            )

        split_name = split_names[split_index]
        state["split_index"] = split_index + 1

        validated_target_mode = common.validate_target_mode(
            target_mode
        )

        features = df.loc[:, feature_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        targets = df.loc[:, target_column].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        if np.isnan(features).any():
            raise ValueError(
                "NaN en features antes de construir ventanas."
            )
        if np.isnan(targets).any():
            raise ValueError(
                "NaN en target antes de construir ventanas."
            )
        if np.isinf(features).any():
            raise ValueError(
                "Inf en features antes de construir ventanas."
            )
        if np.isinf(targets).any():
            raise ValueError(
                "Inf en target antes de construir ventanas."
            )

        ts_series = pd.to_datetime(
            df.loc[:, timestamp_column],
            utc=True,
            errors="coerce",
        )
        if ts_series.isna().any():
            raise ValueError(
                "Timestamps inválidos/NaT antes de construir ventanas."
            )

        # Timestamps para WindowedSplit, igual propósito que el original.
        timestamps = (
            ts_series.dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )

        # Timestamps enteros ns para validación temporal estricta.
        ts_ns = timestamps.astype(np.int64)

        n_rows = len(df)
        max_start = (
            n_rows - lookback - horizon + 1
        )

        if max_start <= 0:
            raise ValueError(
                f"No hay suficientes filas para lookback={lookback}, "
                f"horizon={horizon}. Filas={n_rows}"
            )

        valid_mask = strict_valid_start_mask(
            ts_ns=ts_ns,
            lookback=lookback,
            horizon=horizon,
        )

        candidate_starts = np.arange(
            max_start,
            dtype=np.int64,
        )
        valid_starts = candidate_starts[valid_mask]
        rejected_starts = candidate_starts[~valid_mask]

        if valid_starts.size == 0:
            raise ValueError(
                f"{state['scenario_name']}/{split_name}: "
                "el filtro estricto eliminó todas las ventanas."
            )

        # Auditoría independiente de lo que se acaba de aceptar.
        accepted_audit = audit_accepted_samples(
            ts_ns=ts_ns,
            valid_starts=valid_starts,
            lookback=lookback,
            horizon=horizon,
        )

        if accepted_audit["violating_accepted"] != 0:
            raise RuntimeError(
                f"{state['scenario_name']}/{split_name}: "
                "el build estricto aceptó una ventana inválida."
            )

        x_windows: list[np.ndarray] = []
        y_values: list[float] = []
        y_timestamps: list[np.datetime64] = []
        baseline_values: list[float] = []
        train_target_values: list[float] = []

        for start_idx_np in valid_starts:
            start_idx = int(start_idx_np)
            end_idx = start_idx + lookback
            target_idx = end_idx + horizon - 1

            future_price = float(
                targets[target_idx]
            )
            baseline_price = float(
                targets[end_idx - 1]
            )

            x_windows.append(
                features[start_idx:end_idx]
            )
            y_values.append(
                future_price
            )
            y_timestamps.append(
                timestamps[target_idx]
            )
            baseline_values.append(
                baseline_price
            )
            train_target_values.append(
                common.build_train_target(
                    future_price,
                    baseline_price,
                    validated_target_mode,
                )
            )

        x_raw = np.stack(
            x_windows
        ).astype(np.float32)

        y_raw = np.asarray(
            y_values,
            dtype=np.float32,
        ).reshape(-1, 1)

        ts_array = np.asarray(
            y_timestamps
        )

        baseline_raw = np.asarray(
            baseline_values,
            dtype=np.float32,
        ).reshape(-1, 1)

        train_target_raw = np.asarray(
            train_target_values,
            dtype=np.float32,
        ).reshape(-1, 1)

        strict_call_records.append(
            {
                "scenario_key": state["scenario_key"],
                "scenario": state["scenario_name"],
                "scenario_mode": state["scenario_mode"],
                "split": split_name,
                "rows_in_split": int(n_rows),
                "candidate_windows_original_rule": int(
                    max_start
                ),
                "strict_windows_accepted": int(
                    valid_starts.size
                ),
                "strict_windows_rejected": int(
                    rejected_starts.size
                ),
                "strict_rejected_pct": (
                    100.0
                    * rejected_starts.size
                    / max_start
                ),
                "violations_among_accepted": int(
                    accepted_audit[
                        "violating_accepted"
                    ]
                ),
                "max_input_span_hours_accepted": (
                    accepted_audit[
                        "max_input_span_hours"
                    ]
                ),
                "max_target_lag_hours_accepted": (
                    accepted_audit[
                        "max_target_lag_hours"
                    ]
                ),
                "max_total_span_hours_accepted": (
                    accepted_audit[
                        "max_total_span_hours"
                    ]
                ),
            }
        )

        return common.WindowedSplit(
            x_raw=x_raw,
            y_raw=y_raw,
            timestamps=ts_array,
            baseline_raw=baseline_raw,
            train_target_raw=train_target_raw,
            target_mode=validated_target_mode,
        )

    # ------------------------------------------------------------------
    # FASE A: prepara cada escenario con build_windows ORIGINAL.
    # FASE B: prepara el mismo escenario con build_windows ESTRICTO EN MEMORIA.
    # ------------------------------------------------------------------

    comparison_rows: list[dict] = []

    for scenario_key, scenario in catalog.items():
        print()
        print("-" * 110)
        print(
            f"ESCENARIO: {scenario_key} | "
            f"{scenario.display_name} | mode={scenario.mode}"
        )
        print("-" * 110)

        # A) pipeline original
        common.build_windows = original_build_windows

        original_prepared = common.prepare_scenario(
            scenario=scenario,
            config=config,
            device=torch.device("cpu"),
            target_mode=TARGET_MODE,
        )

        # B) copia estricta solo en memoria
        state["scenario_key"] = scenario_key
        state["scenario_name"] = scenario.display_name
        state["scenario_mode"] = scenario.mode
        state["split_index"] = 0

        common.build_windows = build_windows_strict_test

        strict_prepared = common.prepare_scenario(
            scenario=scenario,
            config=config,
            device=torch.device("cpu"),
            target_mode=TARGET_MODE,
        )

        if state["split_index"] != 3:
            raise RuntimeError(
                f"{scenario_key}: se esperaban 3 llamadas "
                f"al build estricto; hubo {state['split_index']}."
            )

        records_this_scenario = [
            row
            for row in strict_call_records
            if row["scenario_key"] == scenario_key
        ]

        record_by_split = {
            row["split"]: row
            for row in records_this_scenario
        }

        for split_name in split_names:
            original_split = original_prepared.splits[
                split_name
            ]
            strict_split = strict_prepared.splits[
                split_name
            ]

            original_n = int(
                original_split.x_raw.shape[0]
            )
            strict_n = int(
                strict_split.x_raw.shape[0]
            )

            record = record_by_split[split_name]

            if (
                original_n
                != record[
                    "candidate_windows_original_rule"
                ]
            ):
                raise RuntimeError(
                    f"{scenario_key}/{split_name}: "
                    "conteo original no coincide con candidatos."
                )

            if (
                strict_n
                != record[
                    "strict_windows_accepted"
                ]
            ):
                raise RuntimeError(
                    f"{scenario_key}/{split_name}: "
                    "conteo estricto no coincide con auditoría."
                )

            # Los timestamps estrictos deben ser subconjunto de los originales.
            original_ts = pd.Index(
                pd.to_datetime(
                    original_split.timestamps,
                    utc=True,
                )
            )
            strict_ts = pd.Index(
                pd.to_datetime(
                    strict_split.timestamps,
                    utc=True,
                )
            )

            strict_is_subset = bool(
                strict_ts.isin(original_ts).all()
            )

            if not strict_is_subset:
                raise RuntimeError(
                    f"{scenario_key}/{split_name}: "
                    "los targets estrictos no son subconjunto "
                    "de los targets originales."
                )

            # Si no se rechazó nada, ambos raw outputs deben ser EXACTOS.
            exact_same_when_no_rejection = None
            if original_n == strict_n:
                exact_same_when_no_rejection = bool(
                    np.array_equal(
                        original_split.x_raw,
                        strict_split.x_raw,
                    )
                    and np.array_equal(
                        original_split.y_raw,
                        strict_split.y_raw,
                    )
                    and np.array_equal(
                        original_split.baseline_raw,
                        strict_split.baseline_raw,
                    )
                    and np.array_equal(
                        original_split.train_target_raw,
                        strict_split.train_target_raw,
                    )
                    and np.array_equal(
                        original_split.timestamps,
                        strict_split.timestamps,
                    )
                )

                if not exact_same_when_no_rejection:
                    raise RuntimeError(
                        f"{scenario_key}/{split_name}: "
                        "no había gaps rechazados pero la copia "
                        "estricta alteró los datos raw."
                    )

            comparison_rows.append(
                {
                    "scenario_key": scenario_key,
                    "scenario": scenario.display_name,
                    "scenario_mode": scenario.mode,
                    "split": split_name,
                    "original_windows": original_n,
                    "strict_windows": strict_n,
                    "rejected_windows": (
                        original_n - strict_n
                    ),
                    "rejected_pct": (
                        100.0
                        * (original_n - strict_n)
                        / original_n
                        if original_n
                        else 0.0
                    ),
                    "strict_targets_subset_of_original": (
                        strict_is_subset
                    ),
                    "exact_same_raw_when_no_rejection": (
                        exact_same_when_no_rejection
                    ),
                    "violations_among_strict_accepted": (
                        record[
                            "violations_among_accepted"
                        ]
                    ),
                    "max_input_span_hours_strict": (
                        record[
                            "max_input_span_hours_accepted"
                        ]
                    ),
                    "max_target_lag_hours_strict": (
                        record[
                            "max_target_lag_hours_accepted"
                        ]
                    ),
                    "max_total_span_hours_strict": (
                        record[
                            "max_total_span_hours_accepted"
                        ]
                    ),
                }
            )

        print(
            "split | original -> strict | rejected | "
            "violations among accepted | max target lag"
        )

        for split_name in split_names:
            row = next(
                r
                for r in comparison_rows
                if r["scenario_key"] == scenario_key
                and r["split"] == split_name
            )
            print(
                f"{split_name:5s} | "
                f"{row['original_windows']:6d} -> "
                f"{row['strict_windows']:6d} | "
                f"rej={row['rejected_windows']:4d} | "
                f"accepted_viol={row['violations_among_strict_accepted']:2d} | "
                f"max_target_lag={row['max_target_lag_hours_strict']:.2f} h"
            )

    # Restauración explícita aunque el script ya vaya a terminar.
    common.build_windows = original_build_windows

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    strict_records_df = pd.DataFrame(
        strict_call_records
    )

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------

    total_original = int(
        comparison_df["original_windows"].sum()
    )
    total_strict = int(
        comparison_df["strict_windows"].sum()
    )
    total_rejected = int(
        comparison_df["rejected_windows"].sum()
    )

    total_strict_violations = int(
        comparison_df[
            "violations_among_strict_accepted"
        ].sum()
    )

    test_df = comparison_df.loc[
        comparison_df["split"] == "test"
    ].copy()

    total_test_original = int(
        test_df["original_windows"].sum()
    )
    total_test_strict = int(
        test_df["strict_windows"].sum()
    )
    total_test_rejected = int(
        test_df["rejected_windows"].sum()
    )

    all_subset = bool(
        comparison_df[
            "strict_targets_subset_of_original"
        ].all()
    )

    no_rejection_rows = comparison_df.loc[
        comparison_df["rejected_windows"] == 0
    ]

    unchanged_clean_splits = bool(
        no_rejection_rows[
            "exact_same_raw_when_no_rejection"
        ].fillna(False).all()
    )

    strict_geometry_ok = bool(
        np.isclose(
            comparison_df[
                "max_input_span_hours_strict"
            ].astype(float),
            LOOKBACK - 1,
            rtol=0.0,
            atol=1e-12,
        ).all()
        and np.isclose(
            comparison_df[
                "max_target_lag_hours_strict"
            ].astype(float),
            HORIZON,
            rtol=0.0,
            atol=1e-12,
        ).all()
        and np.isclose(
            comparison_df[
                "max_total_span_hours_strict"
            ].astype(float),
            LOOKBACK + HORIZON - 1,
            rtol=0.0,
            atol=1e-12,
        ).all()
    )

    status = (
        "PASS_STRICT_48H_TO_1H"
        if (
            total_strict_violations == 0
            and all_subset
            and unchanged_clean_splits
            and strict_geometry_ok
        )
        else "FAIL_REVIEW_REQUIRED"
    )

    # ------------------------------------------------------------------
    # GUARDADO
    # ------------------------------------------------------------------

    comparison_path = (
        OUTPUT_DIR
        / "strict_vs_original_by_split.csv"
    )

    strict_calls_path = (
        OUTPUT_DIR
        / "strict_build_internal_audit.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "strict_vs_original_test_only.csv"
    )

    manifest_path = (
        OUTPUT_DIR
        / "strict_window_test_manifest.json"
    )

    report_path = (
        OUTPUT_DIR
        / "strict_window_test_report.txt"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    strict_records_df.to_csv(
        strict_calls_path,
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
        "training_performed": False,
        "official_pipeline_modified": False,
        "scenario_count": len(catalog),
        "feature_count": len(
            config.feature_columns or []
        ),
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "strict_rule": (
            "Every transition from first input row through target "
            "must be exactly +1 hour."
        ),
        "original_total_windows_train_val_test": (
            total_original
        ),
        "strict_total_windows_train_val_test": (
            total_strict
        ),
        "rejected_total_windows": total_rejected,
        "rejected_total_pct": (
            100.0
            * total_rejected
            / total_original
        ),
        "original_total_test_windows": (
            total_test_original
        ),
        "strict_total_test_windows": (
            total_test_strict
        ),
        "rejected_test_windows": (
            total_test_rejected
        ),
        "rejected_test_pct": (
            100.0
            * total_test_rejected
            / total_test_original
        ),
        "violations_among_strict_accepted": (
            total_strict_violations
        ),
        "strict_targets_subset_of_original": (
            all_subset
        ),
        "clean_splits_raw_exactly_unchanged": (
            unchanged_clean_splits
        ),
        "strict_geometry_verified": (
            strict_geometry_ok
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 100,
        "PRUEBA SEGURA build_windows ESTRICTO - RESULTADO FINAL",
        "=" * 100,
        f"Estado estricto                    : {status}",
        "Entrenamiento realizado            : NO",
        "Pipeline oficial modificado         : NO",
        f"Escenarios                         : {len(catalog)} / 13",
        f"Features                           : {len(config.feature_columns or [])}",
        f"Lookback                           : {LOOKBACK}",
        f"Horizon                            : {HORIZON}",
        "",
        f"Ventanas originales total          : {total_original:,}",
        f"Ventanas estrictas total           : {total_strict:,}",
        f"Ventanas rechazadas total          : {total_rejected:,}",
        f"Porcentaje rechazado total         : {manifest['rejected_total_pct']:.6f}%",
        "",
        f"Ventanas TEST originales           : {total_test_original:,}",
        f"Ventanas TEST estrictas            : {total_test_strict:,}",
        f"Ventanas TEST rechazadas           : {total_test_rejected:,}",
        f"Porcentaje TEST rechazado          : {manifest['rejected_test_pct']:.6f}%",
        "",
        f"Violaciones ENTRE aceptadas        : {total_strict_violations}",
        f"Targets estrictos ⊂ originales     : {all_subset}",
        f"Splits limpios sin cambios raw     : {unchanged_clean_splits}",
        f"Geometría 47h + 1h = 48h           : {strict_geometry_ok}",
        "",
        "TEST POR ESCENARIO",
        "-" * 100,
    ]

    for _, row in test_df.iterrows():
        lines.append(
            f"{str(row['scenario_key']):30s} "
            f"{int(row['original_windows']):6d} -> "
            f"{int(row['strict_windows']):6d} | "
            f"rechazadas={int(row['rejected_windows']):4d} | "
            f"aceptadas_invalidas="
            f"{int(row['violations_among_strict_accepted']):2d} | "
            f"max target lag="
            f"{float(row['max_target_lag_hours_strict']):.2f} h"
        )

    lines.extend(
        [
            "",
            "ARCHIVOS",
            "-" * 100,
            str(comparison_path),
            str(strict_calls_path),
            str(test_path),
            str(manifest_path),
            "",
            "IMPORTANTE:",
            "Este script NO modifica 10_common_training_pipeline.py.",
            (
                "Solo si Estado estricto = PASS_STRICT_48H_TO_1H "
                "se recomienda preparar el reemplazo oficial."
            ),
        ]
    )

    report = "\n".join(lines)

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(report)


if __name__ == "__main__":
    main()
