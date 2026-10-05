from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = PROJECT_ROOT / "src" / "10_common_training_pipeline.py"
OUT_DIR = PROJECT_ROOT / "outputs" / "strict_window_migration"
MARKER = "STRICT_HOURLY_CONTINUITY_V1"

LOOKBACK = 48
HORIZON = 1
TARGET_MODE = "delta"
EXPECTED_STEP_NS = 3_600_000_000_000

EXPECTED_TEST_COUNTS = {
    "global": 9358,
    "pre_covid": 1380,
    "covid": 4172,
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

EXPECTED_TOTAL_WINDOWS = 179_304
EXPECTED_TOTAL_TEST_WINDOWS = 50_356


def load_common():
    text = COMMON_PATH.read_text(encoding="utf-8")
    if MARKER not in text:
        raise RuntimeError(
            "El pipeline oficial NO contiene el filtro estricto."
        )

    spec = importlib.util.spec_from_file_location(
        "common_strict_official_verify",
        COMMON_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["common_strict_official_verify"] = module
    spec.loader.exec_module(module)
    return module


def expected_valid_targets(
    df: pd.DataFrame,
    timestamp_column: str,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    ts = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce",
    )
    if ts.isna().any():
        raise ValueError("NaT detectado.")

    raw_ts = ts.to_numpy()
    ns = (
        ts.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    max_start = len(df) - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError("Split demasiado corto.")

    good = np.diff(ns) == EXPECTED_STEP_NS
    bad_prefix = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum((~good).astype(np.int64)),
        ]
    )

    starts = np.arange(max_start, dtype=np.int64)
    target_idx = starts + lookback + horizon - 1
    bad_count = bad_prefix[target_idx] - bad_prefix[starts]
    return np.asarray(raw_ts[target_idx[bad_count == 0]])


def normalized_ns(values) -> np.ndarray:
    idx = pd.to_datetime(values, utc=True)
    return (
        idx.tz_convert("UTC")
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common = load_common()

    config = common.ExperimentConfig(
        project_root=PROJECT_ROOT,
        lookback=LOOKBACK,
        horizon=HORIZON,
        random_seed=42,
        feature_columns=list(common.CANONICAL_FEATURES_22),
    )

    catalog = common.build_default_scenarios(
        project_root=PROJECT_ROOT
    )
    if len(catalog) != 13:
        raise RuntimeError(f"Escenarios={len(catalog)}; esperado=13")

    official = common.build_windows
    rows = []
    state = {"key": "", "name": "", "mode": "", "i": 0}
    split_names = ["train", "val", "test"]

    def verifying_wrapper(
        df,
        feature_columns,
        target_column,
        timestamp_column,
        lookback,
        horizon,
        target_mode="price",
    ):
        i = int(state["i"])
        if i >= 3:
            raise RuntimeError("Más de 3 llamadas a build_windows.")
        split = split_names[i]
        state["i"] = i + 1

        expected_ts = expected_valid_targets(
            df,
            timestamp_column,
            lookback,
            horizon,
        )

        result = official(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            timestamp_column=timestamp_column,
            lookback=lookback,
            horizon=horizon,
            target_mode=target_mode,
        )

        exp_ns = normalized_ns(expected_ts)
        act_ns = normalized_ns(result.timestamps)
        timestamps_exact = bool(np.array_equal(exp_ns, act_ns))

        if not timestamps_exact:
            raise RuntimeError(
                f"{state['key']}/{split}: targets no coinciden "
                "con la máscara estricta."
            )

        rows.append(
            {
                "scenario_key": state["key"],
                "scenario": state["name"],
                "mode": state["mode"],
                "split": split,
                "strict_windows": int(result.x_raw.shape[0]),
                "expected_windows": int(len(expected_ts)),
                "timestamps_exact": timestamps_exact,
                "accepted_violations": 0,
            }
        )
        return result

    common.build_windows = verifying_wrapper

    try:
        for key, scenario in catalog.items():
            state["key"] = key
            state["name"] = scenario.display_name
            state["mode"] = scenario.mode
            state["i"] = 0

            common.prepare_scenario(
                scenario=scenario,
                config=config,
                device=torch.device("cpu"),
                target_mode=TARGET_MODE,
            )

            if state["i"] != 3:
                raise RuntimeError(
                    f"{key}: llamadas build_windows={state['i']}; esperado=3"
                )
    finally:
        common.build_windows = official

    df = pd.DataFrame(rows)
    total = int(df["strict_windows"].sum())
    test = df.loc[df["split"] == "test"].copy()
    total_test = int(test["strict_windows"].sum())

    test["expected_validated_test_n"] = (
        test["scenario_key"].map(EXPECTED_TEST_COUNTS).astype(int)
    )
    test["count_match"] = (
        test["strict_windows"]
        == test["expected_validated_test_n"]
    )

    pass_all = bool(
        (df["accepted_violations"] == 0).all()
        and df["timestamps_exact"].all()
        and test["count_match"].all()
        and total == EXPECTED_TOTAL_WINDOWS
        and total_test == EXPECTED_TOTAL_TEST_WINDOWS
    )

    status = (
        "PASS_STRICT_48H_TO_1H_OFFICIAL"
        if pass_all
        else "FAIL_REVIEW_REQUIRED"
    )

    split_csv = OUT_DIR / "official_strict_audit_by_split.csv"
    test_csv = OUT_DIR / "official_strict_audit_test.csv"
    manifest_path = OUT_DIR / "official_strict_audit_manifest.json"

    df.to_csv(split_csv, index=False, encoding="utf-8-sig")
    test.to_csv(test_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "official_pipeline_modified": True,
        "training_performed": False,
        "scenario_count": 13,
        "feature_count": 22,
        "lookback": 48,
        "horizon": 1,
        "total_windows": total,
        "total_test_windows": total_test,
        "accepted_violations": int(df["accepted_violations"].sum()),
        "all_target_timestamps_exact": bool(df["timestamps_exact"].all()),
        "all_test_counts_match_validated_safe_run": bool(
            test["count_match"].all()
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("AUDITORIA DEL PIPELINE OFICIAL ESTRICTO")
    print("=" * 100)
    print(f"Estado                           : {status}")
    print(f"Ventanas train+val+test          : {total:,}")
    print(f"Ventanas TEST                    : {total_test:,}")
    print("Violaciones entre aceptadas      : 0")
    print(f"Timestamps target exactos        : {df['timestamps_exact'].all()}")
    print(f"Conteos test esperados           : {test['count_match'].all()}")
    print()
    print(
        test[
            [
                "scenario_key",
                "strict_windows",
                "expected_validated_test_n",
                "count_match",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Manifest: {manifest_path}")

    if not pass_all:
        raise RuntimeError(
            "La auditoría oficial no pasó. NO regeneres el benchmark."
        )


if __name__ == "__main__":
    main()
