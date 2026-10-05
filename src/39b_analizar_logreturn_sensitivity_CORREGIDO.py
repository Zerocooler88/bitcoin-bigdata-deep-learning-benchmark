from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import logreturn_sensitivity_common_CORREGIDO as lr


def geometric_mean_positive(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="raise").to_numpy(dtype=float)
    if (arr <= 0).any():
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def main() -> None:
    analysis_root = lr.FULL_ROOT / "analysis"
    manifest_path = analysis_root / "logreturn_full_manifest.json"
    price_path = analysis_root / "logreturn_13_scenarios_seed42_price_metrics.csv"
    return_path = analysis_root / "logreturn_13_scenarios_seed42_return_metrics.csv"
    multi_path = analysis_root / "logreturn_global_multiseed_price_detail.csv"
    multi_return_path = analysis_root / "logreturn_global_multiseed_return_detail.csv"

    required = [manifest_path, price_path, return_path, multi_path, multi_return_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan outputs full:\n- " + "\n- ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_LOGRETURN_SENSITIVITY":
        raise RuntimeError(f"Estado full no completo: {manifest.get('status')}")

    price = pd.read_csv(price_path)
    ret = pd.read_csv(return_path)
    multi = pd.read_csv(multi_path)
    multi_ret = pd.read_csv(multi_return_path)

    # 65-result supplementary table: price + return diagnostics.
    key = ["scenario_key", "scenario", "model", "seed"]
    supplement = price.merge(
        ret,
        on=key,
        how="inner",
        suffixes=("_price", "_return"),
    )
    supplement.to_csv(
        analysis_root / "logreturn_supplement_65_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Scenario-normalized aggregation to address cross-scale RMSE averaging.
    neural = price.loc[price["model"] != "baseline_persistencia"].copy()
    scenario_summary = (
        neural.groupby("model", as_index=False)
        .agg(
            scenarios=("scenario_key", "nunique"),
            geometric_mean_relative_RMSE=("relative_RMSE", geometric_mean_positive),
            mean_relative_RMSE=("relative_RMSE", "mean"),
            median_relative_RMSE=("relative_RMSE", "median"),
            mean_RMSE_skill=("RMSE_skill", "mean"),
            wins_vs_persistence=("RMSE_skill", lambda x: int((x > 0).sum())),
        )
        .sort_values("geometric_mean_relative_RMSE")
        .reset_index(drop=True)
    )

    neural_ret = ret.loc[ret["model"] != "baseline_persistencia"].copy()
    ret_summary = (
        neural_ret.groupby("model", as_index=False)
        .agg(
            mean_directional_accuracy=("directional_accuracy", "mean"),
            mean_R2_oos_vs_zero_return=("R2_oos_vs_zero_return", "mean"),
            geometric_mean_return_RMSE_ratio=(
                "return_RMSE_ratio_vs_persistence",
                geometric_mean_positive,
            ),
            mean_return_skill=("return_skill_vs_persistence", "mean"),
        )
    )

    scenario_summary = scenario_summary.merge(ret_summary, on="model", how="left")
    scenario_summary.to_csv(
        analysis_root / "logreturn_scenario_normalized_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Five-seed normalized summary.
    multi_summary = (
        multi.groupby("model", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_price_RMSE=("price_RMSE", "mean"),
            sd_price_RMSE=("price_RMSE", "std"),
            geometric_mean_relative_RMSE=("relative_RMSE", geometric_mean_positive),
            mean_relative_RMSE=("relative_RMSE", "mean"),
            mean_RMSE_skill=("RMSE_skill", "mean"),
            wins_seed_vs_persistence=("RMSE_skill", lambda x: int((x > 0).sum())),
        )
    )

    multi_ret_summary = (
        multi_ret.groupby("model", as_index=False)
        .agg(
            mean_return_RMSE=("return_RMSE", "mean"),
            sd_return_RMSE=("return_RMSE", "std"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            mean_R2_oos_vs_zero_return=("R2_oos_vs_zero_return", "mean"),
            mean_corr_pred_realized=("corr_predicted_vs_realized_log_return", "mean"),
        )
    )

    multi_summary = multi_summary.merge(multi_ret_summary, on="model", how="left")
    multi_summary = multi_summary.sort_values("mean_price_RMSE").reset_index(drop=True)
    multi_summary.to_csv(
        analysis_root / "logreturn_global_multiseed_normalized_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 110)
    print("LOG-RETURN SENSITIVITY - ANALYSIS")
    print("=" * 110)
    print()
    print("13-SCENARIO NORMALIZED SUMMARY")
    print(scenario_summary.to_string(index=False))
    print()
    print("GLOBAL FIVE-SEED SUMMARY")
    print(multi_summary.to_string(index=False))
    print()
    print("Files:")
    print(analysis_root / "logreturn_supplement_65_results.csv")
    print(analysis_root / "logreturn_scenario_normalized_summary.csv")
    print(analysis_root / "logreturn_global_multiseed_normalized_summary.csv")


if __name__ == "__main__":
    main()
