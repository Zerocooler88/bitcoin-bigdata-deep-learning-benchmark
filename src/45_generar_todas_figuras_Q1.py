from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys


EXPECTED = [
    "Fig03_bronze_overview_CORRECTED.png",
    "Fig04_monthly_price_contrasts.png",
    "Fig05_high_frequency_returns_CORRECTED.png",
    "Fig06_volatility_extreme_observations.png",
    "Fig07_bronze_relationships.png",
    "Fig08_gold_monthly_close.png",
    "Fig09_global_multiseed_rmse.png",
    "Fig10_global_validation_rmse.png",
    "Fig11_scenario_test_rmse.png",
    "Fig12_delta_rmse_vs_persistence.png",
    "Fig13_global_residual_corrections_14d.png",
    "Fig14_global_delta_rmse_bootstrap_ci.png",
]


def main():
    parser = argparse.ArgumentParser(description="Generate all data-based manuscript figures 3-14.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    src = Path(__file__).resolve().parent
    outdir = root / "outputs" / "figures_q1_final_complete"

    subprocess.run(
        [
            sys.executable,
            str(src / "43_generar_figuras_03_08_EDA_CORREGIDO.py"),
            "--project-root",
            str(root),
            "--output-dir",
            str(outdir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(src / "44_generar_figuras_09_14_BENCHMARK_ACTUAL.py"),
            "--project-root",
            str(root),
            "--output-dir",
            str(outdir),
        ],
        check=True,
    )

    missing = [name for name in EXPECTED if not (outdir / name).exists()]
    if missing:
        raise RuntimeError("Missing generated figures:\n" + "\n".join(missing))

    print("=" * 100)
    print("ALL DATA-BASED MANUSCRIPT FIGURES 3-14 GENERATED")
    print("=" * 100)
    print(outdir)
    for name in EXPECTED:
        print(outdir / name)


if __name__ == "__main__":
    main()
