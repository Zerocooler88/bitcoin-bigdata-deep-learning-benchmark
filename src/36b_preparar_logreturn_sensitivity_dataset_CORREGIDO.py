from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import logreturn_sensitivity_common_CORREGIDO as lr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recrear el parquet de sensibilidad aunque ya exista.",
    )
    args = parser.parse_args()

    manifest = lr.build_logreturn_dataset(force=args.force)

    print("=" * 100)
    print("DATASET LOG-RETURN DE SENSIBILIDAD")
    print("=" * 100)
    print(f"Estado            : {manifest['status']}")
    print(f"Filas fuente      : {manifest['source_rows']:,}")
    print(f"Filas salida      : {manifest['output_rows']:,}")
    print(f"Filas excluidas por universo oficial: {manifest.get('official_ineligible_rows_removed', 'N/A')}")
    print(f"Universo oficial igualado            : {manifest.get('official_row_universe_matched', False)}")
    print(f"Features          : {manifest['feature_count']}")
    print(f"Gap boundaries    : {manifest['strict_gap_boundaries_detected']}")
    print("Raw price features: NO")
    print("Year feature      : NO")
    print("Calendario cíclico: SÍ")
    print("Rolling cruza gaps: NO")
    print()
    print(f"Parquet : {manifest['output_parquet']}")
    print(f"Manifest: {lr.SENS_DATA_MANIFEST}")


if __name__ == "__main__":
    main()
