from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import logreturn_sensitivity_common_CORREGIDO_v3 as lr


def main() -> None:
    pilot_root = lr.PILOT_ROOT
    analysis = pilot_root / "analysis"
    manifest_path = analysis / "pilot_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_LOGRETURN_GLOBAL_SEED42":
        raise RuntimeError(
            f"Piloto no aprobado: {manifest.get('status')}"
        )

    candidates = list(
        pilot_root.rglob(
            "baseline_persistencia__global_2019_2026__test.csv"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "No se encontró exactamente un CSV baseline del piloto. "
            f"Encontrados={len(candidates)}"
        )

    pred = candidates[0]
    price = lr.price_metrics_from_prediction(
        pred,
        as_persistence=True,
    )
    ret = lr.prediction_return_diagnostics(
        pred,
        as_persistence=True,
    )

    # Update only the baseline row in existing pilot tables.
    price_path = analysis / "pilot_relative_price_metrics.csv"
    ret_path = analysis / "pilot_return_metrics.csv"

    price_df = pd.read_csv(price_path)
    ret_df = pd.read_csv(ret_path)

    mask_p = price_df["model"].eq("baseline_persistencia")
    mask_r = ret_df["model"].eq("baseline_persistencia")

    for k, v in price.items():
        if k in price_df.columns:
            price_df.loc[mask_p, k] = v

    for k, v in ret.items():
        if k in ret_df.columns:
            ret_df.loc[mask_r, k] = v

    price_df.to_csv(price_path, index=False, encoding="utf-8-sig")
    ret_df.to_csv(ret_path, index=False, encoding="utf-8-sig")

    manifest["persistence_metrics_canonicalized"] = True
    manifest["persistence_return_definition"] = "predicted log-return = 0 exactly"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("PILOTO LOG-RETURN - BASELINE CANÓNICO CORREGIDO SIN REENTRENAR")
    print("=" * 100)
    print("Persistence price relative_RMSE : 1.0")
    print("Persistence price RMSE_skill    : 0.0")
    print("Persistence return RMSE ratio   : 1.0")
    print("Persistence return skill        : 0.0")
    print("Persistence OOS R2              : 0.0")
    print("Directional accuracy persistence: N/A (zero-return forecast)")
    print()
    print("No se reentrenó ningún modelo.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
