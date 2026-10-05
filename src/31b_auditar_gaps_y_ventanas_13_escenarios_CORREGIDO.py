from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch


# ======================================================================================
# AUDITORIA DE CONTINUIDAD TEMPORAL DE VENTANAS
# ======================================================================================
#
# Objetivo:
#   Verificar, SIN ENTRENAR MODELOS, si las ventanas que usa el pipeline actual
#   cumplen estrictamente:
#
#       lookback = 48 filas horarias consecutivas
#       horizon  = 1 hora exacta entre la ultima fila del input y el target
#
# El script reutiliza prepare_scenario() y build_windows() del pipeline oficial.
# Intercepta cada DataFrame justo antes de construir las ventanas, audita sus timestamps
# y luego deja que build_windows() original produzca exactamente las mismas muestras.
#
# No modifica datos, no modifica el pipeline y no entrena ninguna red.
# ======================================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
COMMON_PATH = SRC_DIR / "10_common_training_pipeline.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gap_window_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 48
HORIZON = 1
EXPECTED_STEP_HOURS = 1.0
TARGET_MODE = "delta"

EXPECTED_TEST_N = {
    "global": 9358,
    "pre_covid": 1476,
    "covid": 4220,
    "post_covid": 3567,
    "trimestre_mas_bajo": 275,
    "trimestre_mas_alto": 284,
    "intra_2024_Q1": 280,
    "intra_2024_Q3": 284,
    "intra_2025_Q1": 276,
    "cross_2024_Q1_to_2024_Q3": 2160,
    "cross_2024_Q1_to_2025_Q1": 2112,
    "cross_pre_to_post": 24048,
    "cross_bajo_to_alto": 2160,
}


def load_common_module():
    if not COMMON_PATH.exists():
        raise FileNotFoundError(
            f"No existe el pipeline común:\n{COMMON_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "common_gap_window_audit",
        COMMON_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"No se pudo cargar:\n{COMMON_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["common_gap_window_audit"] = module
    spec.loader.exec_module(module)
    return module


def hours_between(a_ns: np.ndarray, b_ns: np.ndarray) -> np.ndarray:
    """Devuelve (b-a) en horas usando enteros nanosegundos."""
    return (b_ns - a_ns) / 3_600_000_000_000.0


def audit_frame(
    *,
    df: pd.DataFrame,
    scenario_key: str,
    scenario_name: str,
    scenario_mode: str,
    split_name: str,
    timestamp_column: str,
    lookback: int,
    horizon: int,
) -> tuple[dict, list[dict], list[dict]]:
    """
    Audita exactamente las muestras que build_windows() generaria sobre este DataFrame.
    """

    if timestamp_column not in df.columns:
        raise KeyError(
            f"{scenario_name}/{split_name}: no existe timestamp {timestamp_column!r}"
        )

    ts = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce",
    )

    nat_count = int(ts.isna().sum())

    if nat_count:
        raise ValueError(
            f"{scenario_name}/{split_name}: existen {nat_count} timestamps NaT. "
            "La auditoria se detiene porque no puede probar continuidad."
        )

    # IMPORTANT:
    # pandas 3.x may preserve datetime64[us] resolution internally. Converting
    # directly with astype("int64") can therefore return microseconds rather
    # than nanoseconds. The previous audit assumed nanoseconds and produced
    # a false 100% violation rate. Force UTC timestamps explicitly to
    # datetime64[ns] before integer arithmetic.
    ts_ns = (
        ts.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    n_rows = len(df)
    max_start = n_rows - lookback - horizon + 1

    if max_start <= 0:
        raise ValueError(
            f"{scenario_name}/{split_name}: insuficientes filas para "
            f"lookback={lookback}, horizon={horizon}. Filas={n_rows}"
        )

    # Delta entre cada par de filas consecutivas del DataFrame.
    delta_hours = np.diff(ts_ns) / 3_600_000_000_000.0

    # Sanity check: in a nominal hourly dataset, the dominant/median step
    # should be approximately 1 hour. We do not force it to 1 here because
    # genuine gaps are precisely what this audit must detect.
    median_step_hours = (
        float(np.median(delta_hours))
        if len(delta_hours)
        else float("nan")
    )

    # Una transición válida debe ser exactamente +1 hora.
    good_step = np.isclose(
        delta_hours,
        EXPECTED_STEP_HOURS,
        rtol=0.0,
        atol=1e-9,
    )
    bad_step = ~good_step

    # Prefix sum para contar gaps dentro de cada ventana sin bucles costosos.
    bad_prefix = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(bad_step.astype(np.int64)),
        ]
    )

    starts = np.arange(max_start, dtype=np.int64)
    end_exclusive = starts + lookback
    last_input_idx = end_exclusive - 1
    target_idx = end_exclusive + horizon - 1

    # Transiciones internas del input: start ... last_input
    input_bad_count = (
        bad_prefix[last_input_idx] - bad_prefix[starts]
    )

    # Transiciones desde la ultima observacion visible hasta el target.
    target_bad_count = (
        bad_prefix[target_idx] - bad_prefix[last_input_idx]
    )

    # Todo el camino desde la primera observacion de input hasta target.
    path_bad_count = (
        bad_prefix[target_idx] - bad_prefix[starts]
    )

    input_start_ns = ts_ns[starts]
    input_end_ns = ts_ns[last_input_idx]
    target_ns = ts_ns[target_idx]

    input_span_hours = hours_between(
        input_start_ns,
        input_end_ns,
    )
    target_lag_hours = hours_between(
        input_end_ns,
        target_ns,
    )
    total_span_hours = hours_between(
        input_start_ns,
        target_ns,
    )

    expected_input_span = float(lookback - 1)
    expected_target_lag = float(horizon)
    expected_total_span = float(lookback + horizon - 1)

    input_span_ok = np.isclose(
        input_span_hours,
        expected_input_span,
        rtol=0.0,
        atol=1e-9,
    )
    target_lag_ok = np.isclose(
        target_lag_hours,
        expected_target_lag,
        rtol=0.0,
        atol=1e-9,
    )
    total_span_ok = np.isclose(
        total_span_hours,
        expected_total_span,
        rtol=0.0,
        atol=1e-9,
    )

    input_consecutive = (
        (input_bad_count == 0)
        & input_span_ok
    )

    target_exact = (
        (target_bad_count == 0)
        & target_lag_ok
    )

    strict_valid = (
        (path_bad_count == 0)
        & input_consecutive
        & target_exact
        & total_span_ok
    )

    violating_idx = np.flatnonzero(~strict_valid)

    # Eventos de discontinuidad unicos dentro del split.
    gap_rows: list[dict] = []
    for transition_idx in np.flatnonzero(bad_step):
        before = ts.iloc[int(transition_idx)]
        after = ts.iloc[int(transition_idx + 1)]
        gap_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario": scenario_name,
                "scenario_mode": scenario_mode,
                "split": split_name,
                "transition_index": int(transition_idx),
                "timestamp_before": before.isoformat(),
                "timestamp_after": after.isoformat(),
                "delta_hours": float(delta_hours[transition_idx]),
                "missing_hour_equivalents": (
                    float(delta_hours[transition_idx] - 1.0)
                    if delta_hours[transition_idx] > 1.0
                    else 0.0
                ),
                "transition_type": (
                    "gap"
                    if delta_hours[transition_idx] > 1.0
                    else (
                        "duplicate"
                        if np.isclose(delta_hours[transition_idx], 0.0)
                        else "backward_or_subhour"
                    )
                ),
            }
        )

    # Detalle de cada muestra que viola el supuesto.
    violation_rows: list[dict] = []

    for sample_index in violating_idx:
        s = int(starts[sample_index])
        li = int(last_input_idx[sample_index])
        ti = int(target_idx[sample_index])

        window_deltas = delta_hours[s:ti]
        max_step = (
            float(np.max(window_deltas))
            if len(window_deltas)
            else float("nan")
        )
        min_step = (
            float(np.min(window_deltas))
            if len(window_deltas)
            else float("nan")
        )

        reasons = []
        if input_bad_count[sample_index] > 0:
            reasons.append("input_crosses_non_1h_step")
        if not input_span_ok[sample_index]:
            reasons.append("input_span_not_47h")
        if target_bad_count[sample_index] > 0:
            reasons.append("target_path_crosses_non_1h_step")
        if not target_lag_ok[sample_index]:
            reasons.append("target_not_exact_horizon")
        if not total_span_ok[sample_index]:
            reasons.append("total_span_not_expected")

        violation_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario": scenario_name,
                "scenario_mode": scenario_mode,
                "split": split_name,
                "sample_index": int(sample_index),
                "start_row": s,
                "last_input_row": li,
                "target_row": ti,
                "input_start": ts.iloc[s].isoformat(),
                "input_end": ts.iloc[li].isoformat(),
                "target_timestamp": ts.iloc[ti].isoformat(),
                "input_span_hours": float(
                    input_span_hours[sample_index]
                ),
                "expected_input_span_hours": expected_input_span,
                "target_lag_hours": float(
                    target_lag_hours[sample_index]
                ),
                "expected_target_lag_hours": expected_target_lag,
                "total_span_hours": float(
                    total_span_hours[sample_index]
                ),
                "expected_total_span_hours": expected_total_span,
                "bad_steps_inside_input": int(
                    input_bad_count[sample_index]
                ),
                "bad_steps_input_to_target": int(
                    path_bad_count[sample_index]
                ),
                "min_step_hours": min_step,
                "max_step_hours": max_step,
                "reason": ";".join(reasons),
            }
        )

    valid_count = int(strict_valid.sum())
    invalid_count = int((~strict_valid).sum())

    summary = {
        "scenario_key": scenario_key,
        "scenario": scenario_name,
        "scenario_mode": scenario_mode,
        "split": split_name,
        "rows_in_split": int(n_rows),
        "candidate_windows": int(max_start),
        "strict_valid_windows": valid_count,
        "violating_windows": invalid_count,
        "violating_windows_pct": (
            100.0 * invalid_count / max_start
        ),
        "input_crossing_gap_windows": int(
            (input_bad_count > 0).sum()
        ),
        "target_horizon_violation_windows": int(
            (~target_exact).sum()
        ),
        "any_non_1h_transition_windows": int(
            (path_bad_count > 0).sum()
        ),
        "unique_non_1h_transitions_in_split": int(
            bad_step.sum()
        ),
        "min_step_hours_in_split": (
            float(np.min(delta_hours))
            if len(delta_hours)
            else float("nan")
        ),
        "median_step_hours_in_split": median_step_hours,
        "max_step_hours_in_split": (
            float(np.max(delta_hours))
            if len(delta_hours)
            else float("nan")
        ),
        "max_input_span_hours": float(
            np.max(input_span_hours)
        ),
        "max_target_lag_hours": float(
            np.max(target_lag_hours)
        ),
        "max_total_span_hours": float(
            np.max(total_span_hours)
        ),
        "expected_input_span_hours": expected_input_span,
        "expected_target_lag_hours": expected_target_lag,
        "expected_total_span_hours": expected_total_span,
        "nat_timestamps": nat_count,
    }

    return summary, violation_rows, gap_rows


def main() -> None:
    print("=" * 110)
    print("AUDITORIA DE GAPS Y VENTANAS - 13 ESCENARIOS - SIN ENTRENAMIENTO")
    print("=" * 110)

    common = load_common_module()

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
            "El pipeline cargado no contiene exactamente 22 features canonicas."
        )

    catalog = common.build_default_scenarios(
        project_root=PROJECT_ROOT
    )

    if len(catalog) != 13:
        raise RuntimeError(
            f"Se esperaban 13 escenarios; se encontraron {len(catalog)}."
        )

    original_build_windows = common.build_windows

    all_summaries: list[dict] = []
    all_violations: list[dict] = []
    all_gaps: list[dict] = []
    pipeline_count_checks: list[dict] = []

    state = {
        "scenario_key": None,
        "scenario_name": None,
        "scenario_mode": None,
        "call_index": 0,
    }

    split_order = ["train", "val", "test"]

    def audited_build_windows(
        df,
        feature_columns,
        target_column,
        timestamp_column,
        lookback,
        horizon,
        target_mode="price",
    ):
        call_index = int(state["call_index"])

        if call_index >= 3:
            raise RuntimeError(
                "prepare_scenario llamo build_windows mas de tres veces. "
                "El pipeline cambio y esta auditoria debe actualizarse."
            )

        split_name = split_order[call_index]
        state["call_index"] = call_index + 1

        summary, violations, gaps = audit_frame(
            df=df,
            scenario_key=str(state["scenario_key"]),
            scenario_name=str(state["scenario_name"]),
            scenario_mode=str(state["scenario_mode"]),
            split_name=split_name,
            timestamp_column=timestamp_column,
            lookback=lookback,
            horizon=horizon,
        )

        # Ejecuta build_windows oficial para verificar que contamos exactamente
        # las mismas muestras que usa el experimento actual.
        result = original_build_windows(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            timestamp_column=timestamp_column,
            lookback=lookback,
            horizon=horizon,
            target_mode=target_mode,
        )

        pipeline_windows = int(result.x_raw.shape[0])

        if pipeline_windows != summary["candidate_windows"]:
            raise RuntimeError(
                f"{state['scenario_name']}/{split_name}: "
                f"auditoria={summary['candidate_windows']} ventanas, "
                f"pipeline={pipeline_windows}."
            )

        summary["pipeline_windows"] = pipeline_windows
        summary["pipeline_count_match"] = True

        all_summaries.append(summary)
        all_violations.extend(violations)
        all_gaps.extend(gaps)

        pipeline_count_checks.append(
            {
                "scenario_key": state["scenario_key"],
                "scenario": state["scenario_name"],
                "split": split_name,
                "audit_windows": summary["candidate_windows"],
                "pipeline_windows": pipeline_windows,
                "match": True,
            }
        )

        return result

    common.build_windows = audited_build_windows

    try:
        for scenario_key, scenario in catalog.items():
            print()
            print("-" * 110)
            print(
                f"ESCENARIO: {scenario_key} | "
                f"display={scenario.display_name} | mode={scenario.mode}"
            )
            print("-" * 110)

            state["scenario_key"] = scenario_key
            state["scenario_name"] = scenario.display_name
            state["scenario_mode"] = scenario.mode
            state["call_index"] = 0

            # CPU: la auditoria no necesita GPU y no entrena nada.
            prepared = common.prepare_scenario(
                scenario=scenario,
                config=config,
                device=torch.device("cpu"),
                target_mode=TARGET_MODE,
            )

            if state["call_index"] != 3:
                raise RuntimeError(
                    f"{scenario_key}: se esperaban 3 llamadas a build_windows "
                    f"(train/val/test); se observaron {state['call_index']}."
                )

            # Verifica que los splits preparados sigan teniendo los mismos conteos.
            prepared_counts = {
                "train": int(
                    prepared.splits["train"].x_raw.shape[0]
                ),
                "val": int(
                    prepared.splits["val"].x_raw.shape[0]
                ),
                "test": int(
                    prepared.splits["test"].x_raw.shape[0]
                ),
            }

            scenario_rows = [
                row
                for row in all_summaries
                if row["scenario_key"] == scenario_key
            ]

            for row in scenario_rows:
                split_name = row["split"]
                if (
                    prepared_counts[split_name]
                    != row["candidate_windows"]
                ):
                    raise RuntimeError(
                        f"{scenario_key}/{split_name}: el PreparedScenario "
                        "no coincide con el conteo auditado."
                    )

                print(
                    f"{split_name:5s} | "
                    f"windows={row['candidate_windows']:6d} | "
                    f"valid={row['strict_valid_windows']:6d} | "
                    f"violations={row['violating_windows']:4d} | "
                    f"gaps_in_split={row['unique_non_1h_transitions_in_split']:3d} | "
                    f"median_step={row['median_step_hours_in_split']:.2f} h | "
                    f"max_target_lag={row['max_target_lag_hours']:.2f} h"
                )

    finally:
        common.build_windows = original_build_windows

    summary_df = pd.DataFrame(all_summaries)
    violations_df = pd.DataFrame(all_violations)
    gaps_df = pd.DataFrame(all_gaps)
    count_checks_df = pd.DataFrame(pipeline_count_checks)

    # ----------------------------------------------------------------------------------
    # Resumen TEST por escenario
    # ----------------------------------------------------------------------------------
    test_df = summary_df.loc[
        summary_df["split"] == "test"
    ].copy()

    test_df["expected_test_n_manuscript"] = (
        test_df["scenario_key"]
        .map(EXPECTED_TEST_N)
        .astype("Int64")
    )

    test_df["test_n_matches_manuscript"] = (
        test_df["candidate_windows"].astype("Int64")
        == test_df["expected_test_n_manuscript"]
    )

    # ----------------------------------------------------------------------------------
    # Resumen por escenario sobre train + val + test
    # ----------------------------------------------------------------------------------
    scenario_df = (
        summary_df
        .groupby(
            ["scenario_key", "scenario", "scenario_mode"],
            as_index=False,
        )
        .agg(
            total_windows=("candidate_windows", "sum"),
            total_strict_valid_windows=(
                "strict_valid_windows",
                "sum",
            ),
            total_violating_windows=(
                "violating_windows",
                "sum",
            ),
            train_val_test_non_1h_transitions=(
                "unique_non_1h_transitions_in_split",
                "sum",
            ),
            max_input_span_hours=(
                "max_input_span_hours",
                "max",
            ),
            max_target_lag_hours=(
                "max_target_lag_hours",
                "max",
            ),
            max_total_span_hours=(
                "max_total_span_hours",
                "max",
            ),
        )
    )

    scenario_df["total_violating_pct"] = (
        100.0
        * scenario_df["total_violating_windows"]
        / scenario_df["total_windows"]
    )

    # ----------------------------------------------------------------------------------
    # Resumen total
    # ----------------------------------------------------------------------------------
    total_windows = int(summary_df["candidate_windows"].sum())
    total_violations = int(summary_df["violating_windows"].sum())

    test_windows = int(test_df["candidate_windows"].sum())
    test_violations = int(test_df["violating_windows"].sum())

    all_pipeline_matches = bool(
        summary_df["pipeline_count_match"].all()
    )

    all_test_counts_match = bool(
        test_df["test_n_matches_manuscript"].fillna(False).all()
    )

    strict_status = (
        "PASS_STRICT_48H_TO_1H"
        if total_violations == 0
        else "TEMPORAL_VIOLATIONS_DETECTED"
    )

    # ----------------------------------------------------------------------------------
    # Guardado
    # ----------------------------------------------------------------------------------
    summary_path = OUTPUT_DIR / "gap_window_audit_by_split.csv"
    scenario_path = OUTPUT_DIR / "gap_window_audit_by_scenario.csv"
    test_path = OUTPUT_DIR / "gap_window_audit_test_only.csv"
    violations_path = OUTPUT_DIR / "gap_window_violations_detail.csv"
    gaps_path = OUTPUT_DIR / "gap_events_by_split.csv"
    checks_path = OUTPUT_DIR / "pipeline_window_count_checks.csv"
    manifest_path = OUTPUT_DIR / "gap_window_audit_manifest.json"
    report_path = OUTPUT_DIR / "gap_window_audit_report.txt"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig",
    )

    violations_df.to_csv(
        violations_path,
        index=False,
        encoding="utf-8-sig",
    )

    gaps_df.to_csv(
        gaps_path,
        index=False,
        encoding="utf-8-sig",
    )

    count_checks_df.to_csv(
        checks_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": strict_status,
        "training_performed": False,
        "project_root": str(PROJECT_ROOT),
        "common_pipeline": str(COMMON_PATH),
        "scenario_count": int(len(catalog)),
        "lookback_rows": LOOKBACK,
        "horizon_rows": HORIZON,
        "expected_frequency_hours": EXPECTED_STEP_HOURS,
        "strict_definition": {
            "input_rows": 48,
            "input_expected_elapsed_hours": 47,
            "target_expected_lag_hours_from_last_input": 1,
            "expected_elapsed_hours_first_input_to_target": 48,
            "all_row_to_row_transitions_must_equal_hours": 1,
        },
        "total_windows_train_val_test": total_windows,
        "total_violating_windows_train_val_test": total_violations,
        "total_violation_pct": (
            100.0 * total_violations / total_windows
            if total_windows
            else 0.0
        ),
        "total_test_windows": test_windows,
        "total_test_violating_windows": test_violations,
        "test_violation_pct": (
            100.0 * test_violations / test_windows
            if test_windows
            else 0.0
        ),
        "all_pipeline_window_counts_match": all_pipeline_matches,
        "all_manuscript_test_counts_match": all_test_counts_match,
        "canonical_feature_count": len(
            config.feature_columns or []
        ),
        "interpretation": (
            "PASS: every candidate window uses strictly consecutive hourly timestamps."
            if total_violations == 0
            else (
                "REVIEW REQUIRED: at least one window used by the current pipeline "
                "crosses a non-1h timestamp transition or has a target lag different "
                "from the nominal horizon."
            )
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "=" * 90,
        "AUDITORIA DE GAPS Y VENTANAS - RESULTADO FINAL",
        "=" * 90,
        f"Estado estricto                    : {strict_status}",
        f"Entrenamiento realizado            : NO",
        f"Escenarios auditados               : {len(catalog)} / 13",
        f"Features configuradas              : {len(config.feature_columns or [])}",
        f"Lookback                           : {LOOKBACK} filas",
        f"Horizon                            : {HORIZON} fila",
        f"Frecuencia esperada                : 1 hora",
        f"Conversión temporal                : forzada a datetime64[ns]",
        "",
        f"Ventanas totales train+val+test    : {total_windows:,}",
        f"Ventanas violatorias totales       : {total_violations:,}",
        f"Porcentaje violatorio total        : {manifest['total_violation_pct']:.6f}%",
        "",
        f"Ventanas TEST totales              : {test_windows:,}",
        f"Ventanas TEST violatorias          : {test_violations:,}",
        f"Porcentaje TEST violatorio         : {manifest['test_violation_pct']:.6f}%",
        "",
        f"Conteos auditoria = pipeline       : {all_pipeline_matches}",
        f"Test n = manuscrito final           : {all_test_counts_match}",
        "",
        "TEST POR ESCENARIO",
        "-" * 90,
    ]

    for _, row in test_df.iterrows():
        report_lines.append(
            f"{row['scenario_key']:30s} "
            f"n={int(row['candidate_windows']):6d} | "
            f"viol={int(row['violating_windows']):4d} | "
            f"pct={float(row['violating_windows_pct']):8.5f}% | "
            f"max target lag={float(row['max_target_lag_hours']):7.2f} h | "
            f"n manuscript match={bool(row['test_n_matches_manuscript'])}"
        )

    report_lines.extend(
        [
            "",
            "INTERPRETACION",
            "-" * 90,
            manifest["interpretation"],
            "",
            "Archivos generados:",
            str(summary_path),
            str(scenario_path),
            str(test_path),
            str(violations_path),
            str(gaps_path),
            str(checks_path),
            str(manifest_path),
        ]
    )

    report_text = "\n".join(report_lines)
    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print(report_text)


if __name__ == "__main__":
    main()
