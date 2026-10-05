from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = PROJECT_ROOT / "src" / "10_common_training_pipeline.py"
OUT_DIR = PROJECT_ROOT / "outputs" / "m3_temporal_scenario_audit"
STRICT_MARKER = "STRICT_HOURLY_CONTINUITY_V1"


def load_common():
    text = COMMON_PATH.read_text(encoding="utf-8")
    if STRICT_MARKER not in text:
        raise RuntimeError(
            "El pipeline oficial no contiene el filtro temporal estricto. "
            "No se realiza la auditoría M3 sobre una versión antigua."
        )

    spec = importlib.util.spec_from_file_location(
        "common_m3_temporal_audit",
        COMMON_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)

    module = importlib.util.module_from_spec(spec)
    sys.modules["common_m3_temporal_audit"] = module
    spec.loader.exec_module(module)
    return module


def to_utc_index(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))


def fmt_ts(ts) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")


def frame_range(df: pd.DataFrame, timestamp_column: str):
    ts = to_utc_index(df[timestamp_column])
    return ts.min(), ts.max(), int(len(df))


def split_target_range(split):
    ts = to_utc_index(split.timestamps)
    return ts.min(), ts.max(), int(len(ts)), ts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common = load_common()

    config = common.ExperimentConfig(
        project_root=PROJECT_ROOT,
        lookback=48,
        horizon=1,
        batch_size=64,
        epochs=50,
        patience=5,
        learning_rate=0.001,
        num_workers=0,
        random_seed=42,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_column="close",
        timestamp_column=None,
        feature_columns=list(common.CANONICAL_FEATURES_22),
        model_subdir="m3_audit_no_training",
        save_best_weights=False,
        verbose=False,
    )

    catalog = common.build_default_scenarios(project_root=PROJECT_ROOT)
    if len(catalog) != 13:
        raise RuntimeError(f"Escenarios encontrados={len(catalog)}; esperado=13.")

    # ------------------------------------------------------------------
    # A) Exact source-level regime and quarter ranges after canonical
    #    pre-model cleaning, before scenario-specific train/val/test split.
    # ------------------------------------------------------------------
    global_path = common.DEFAULT_GLOBAL_PARQUET
    source = common.load_parquet_frame(global_path, timestamp_column=None)
    ts_col = common.infer_timestamp_column(source)
    target_col = common.infer_target_column(source, forced_target="close")
    clean = common.clean_frame_for_modeling(
        source,
        feature_columns=list(common.CANONICAL_FEATURES_22),
        target_column=target_col,
        timestamp_column=ts_col,
    )

    regime_rows = []
    for regime in ["pre_covid", "covid", "post_covid"]:
        block, meta = common.apply_filter_spec(
            clean,
            {"type": "regime", "value": regime},
            target_column=target_col,
            timestamp_column=ts_col,
        )
        start, end, n = frame_range(block, ts_col)
        regime_rows.append(
            {
                "regime": regime,
                "source_start_utc": fmt_ts(start),
                "source_end_utc": fmt_ts(end),
                "eligible_rows_before_split": n,
            }
        )
    regimes = pd.DataFrame(regime_rows)
    regimes.to_csv(
        OUT_DIR / "m3_regime_exact_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fixed_quarters = ["2024-Q1", "2024-Q3", "2025-Q1"]
    quarter_rows = []
    for quarter in fixed_quarters:
        block, _ = common.apply_filter_spec(
            clean,
            {"type": "quarter", "value": quarter},
            target_column=target_col,
            timestamp_column=ts_col,
        )
        start, end, n = frame_range(block, ts_col)
        quarter_rows.append(
            {
                "quarter": quarter,
                "source_start_utc": fmt_ts(start),
                "source_end_utc": fmt_ts(end),
                "eligible_rows_before_split": n,
                "design_role": (
                    "anchor quarter"
                    if quarter == "2024-Q1"
                    else (
                        "same-year later quarter relative to 2024-Q1"
                        if quarter == "2024-Q3"
                        else "same calendar quarter one year after 2024-Q1"
                    )
                ),
            }
        )

    low_q = common.resolve_extreme_quarter_label(
        clean, target_column=target_col, extreme_value="lowest_mean_close"
    )
    high_q = common.resolve_extreme_quarter_label(
        clean, target_column=target_col, extreme_value="highest_mean_close"
    )

    for label, quarter in [
        ("lowest_mean_close", low_q),
        ("highest_mean_close", high_q),
    ]:
        block, _ = common.apply_filter_spec(
            clean,
            {"type": "quarter", "value": quarter},
            target_column=target_col,
            timestamp_column=ts_col,
        )
        start, end, n = frame_range(block, ts_col)
        quarter_rows.append(
            {
                "quarter": quarter,
                "source_start_utc": fmt_ts(start),
                "source_end_utc": fmt_ts(end),
                "eligible_rows_before_split": n,
                "design_role": f"data-resolved extreme quarter: {label}",
            }
        )

    quarters = pd.DataFrame(quarter_rows)
    quarters.to_csv(
        OUT_DIR / "m3_quarter_exact_ranges_and_roles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # B) Exact raw split ranges and effective TARGET timestamp ranges
    #    for all 13 scenarios under the definitive strict window builder.
    # ------------------------------------------------------------------
    rows = []
    test_sets = {}

    for key, scenario in catalog.items():
        prepared = common.prepare_scenario(
            scenario=scenario,
            config=config,
            device=torch.device("cpu"),
            target_mode="delta",
        )

        for split_name in ["train", "val", "test"]:
            raw = prepared.raw_frames[split_name]
            raw_start, raw_end, raw_n = frame_range(
                raw, prepared.timestamp_column
            )
            target_start, target_end, target_n, target_ts = split_target_range(
                prepared.splits[split_name]
            )

            rows.append(
                {
                    "scenario_key": key,
                    "scenario_display_name": prepared.scenario.display_name,
                    "mode": prepared.scenario.mode,
                    "split": split_name,
                    "raw_start_utc": fmt_ts(raw_start),
                    "raw_end_utc": fmt_ts(raw_end),
                    "raw_rows": raw_n,
                    "effective_target_start_utc": fmt_ts(target_start),
                    "effective_target_end_utc": fmt_ts(target_end),
                    "strict_windows": target_n,
                    "resolved_metadata_json": json.dumps(
                        prepared.scenario.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                }
            )

            if split_name == "test":
                # Integer ns keys are robust for exact set overlap.
                ns = target_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                test_sets[key] = set(ns.tolist())

    periods = pd.DataFrame(rows)
    periods.to_csv(
        OUT_DIR / "m3_all_scenario_split_date_ranges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # C) Pairwise overlap of effective TEST targets.
    # ------------------------------------------------------------------
    keys = list(catalog.keys())
    count_matrix = pd.DataFrame(index=keys, columns=keys, dtype=int)
    pct_row_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)

    overlap_rows = []
    for a in keys:
        for b in keys:
            inter = len(test_sets[a].intersection(test_sets[b]))
            count_matrix.loc[a, b] = inter
            pct = 100.0 * inter / len(test_sets[a]) if test_sets[a] else np.nan
            pct_row_matrix.loc[a, b] = pct

            if a < b and inter > 0:
                overlap_rows.append(
                    {
                        "scenario_a": a,
                        "scenario_b": b,
                        "overlap_test_targets": inter,
                        "pct_of_a_test": 100.0 * inter / len(test_sets[a]),
                        "pct_of_b_test": 100.0 * inter / len(test_sets[b]),
                    }
                )

    count_matrix.to_csv(
        OUT_DIR / "m3_test_overlap_counts_matrix.csv",
        encoding="utf-8-sig",
    )
    pct_row_matrix.to_csv(
        OUT_DIR / "m3_test_overlap_pct_of_row_matrix.csv",
        encoding="utf-8-sig",
    )

    overlap_long = pd.DataFrame(overlap_rows)
    overlap_long = overlap_long.sort_values(
        ["overlap_test_targets", "scenario_a", "scenario_b"],
        ascending=[False, True, True],
    )
    overlap_long.to_csv(
        OUT_DIR / "m3_nonzero_test_overlaps.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # D) Compact publication-ready audit summary.
    # ------------------------------------------------------------------
    global_rows = periods[
        (periods["scenario_key"] == "global")
    ][
        [
            "split",
            "raw_start_utc",
            "raw_end_utc",
            "effective_target_start_utc",
            "effective_target_end_utc",
            "strict_windows",
        ]
    ].copy()

    global_test = periods[
        (periods["scenario_key"] == "global")
        & (periods["split"] == "test")
    ].iloc[0]

    global_overlaps = []
    for other in keys:
        if other == "global":
            continue
        inter = len(test_sets["global"].intersection(test_sets[other]))
        if inter > 0:
            global_overlaps.append(
                {
                    "other_scenario": other,
                    "overlap_test_targets": inter,
                    "pct_of_global_test": 100.0 * inter / len(test_sets["global"]),
                    "pct_of_other_test": 100.0 * inter / len(test_sets[other]),
                }
            )
    global_overlap_df = pd.DataFrame(global_overlaps)
    if len(global_overlap_df):
        global_overlap_df = global_overlap_df.sort_values(
            "overlap_test_targets", ascending=False
        )
    global_overlap_df.to_csv(
        OUT_DIR / "m3_global_test_overlaps.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = OUT_DIR / "m3_temporal_scenario_audit_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("M3 TEMPORAL SCENARIO AUDIT - NO TRAINING\n")
        f.write("=" * 110 + "\n\n")
        f.write("REGIME EXACT BOUNDARIES\n")
        f.write(regimes.to_string(index=False))
        f.write("\n\nFIXED / EXTREME QUARTERS\n")
        f.write(quarters.to_string(index=False))
        f.write("\n\nGLOBAL CHRONOLOGICAL SPLIT\n")
        f.write(global_rows.to_string(index=False))
        f.write("\n\nGLOBAL TEST OVERLAPS WITH OTHER TEST SCENARIOS\n")
        if len(global_overlap_df):
            f.write(global_overlap_df.to_string(index=False))
        else:
            f.write("No nonzero overlaps.")
        f.write("\n\nALL NONZERO TEST OVERLAPS\n")
        if len(overlap_long):
            f.write(overlap_long.to_string(index=False))
        else:
            f.write("No nonzero overlaps.")
        f.write("\n")

    print("=" * 110)
    print("M3 TEMPORAL SCENARIO AUDIT - COMPLETE")
    print("=" * 110)
    print("Entrenamiento realizado: NO")
    print()
    print("REGIME EXACT BOUNDARIES")
    print(regimes.to_string(index=False))
    print()
    print("FIXED / EXTREME QUARTERS")
    print(quarters.to_string(index=False))
    print()
    print("GLOBAL CHRONOLOGICAL SPLIT")
    print(global_rows.to_string(index=False))
    print()
    print("GLOBAL TEST OVERLAPS")
    if len(global_overlap_df):
        print(global_overlap_df.to_string(index=False))
    else:
        print("No nonzero overlaps.")
    print()
    print("Files:")
    for p in sorted(OUT_DIR.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
