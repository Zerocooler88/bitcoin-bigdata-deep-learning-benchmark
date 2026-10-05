from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import py_compile
import re
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = PROJECT_ROOT / "src" / "10_common_training_pipeline.py"
MIGRATION_DIR = PROJECT_ROOT / "outputs" / "strict_window_migration"
MARKER = "STRICT_HOURLY_CONTINUITY_V1"


STRICT_FUNCTION = """
def build_windows(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    timestamp_column: str,
    lookback: int,
    horizon: int,
    target_mode: str = "price",
) -> WindowedSplit:
    # STRICT_HOURLY_CONTINUITY_V1
    # A candidate sample is retained only when every timestamp transition
    # from the first input row through the target row is exactly +1 hour.
    validated_target_mode = validate_target_mode(target_mode)

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
            "Se detectaron NaN en las columnas predictoras antes de construir ventanas."
        )
    if np.isnan(targets).any():
        raise ValueError(
            "Se detectaron NaN en la columna objetivo antes de construir ventanas."
        )
    if np.isinf(features).any():
        raise ValueError(
            "Se detectaron valores infinitos en las columnas predictoras antes de construir ventanas."
        )
    if np.isinf(targets).any():
        raise ValueError(
            "Se detectaron valores infinitos en la columna objetivo antes de construir ventanas."
        )

    ts_series = pd.to_datetime(
        df.loc[:, timestamp_column],
        utc=True,
        errors="coerce",
    )

    if ts_series.isna().any():
        raise ValueError(
            "Se detectaron timestamps inválidos/NaT antes de construir ventanas."
        )

    # Keep the same output timestamp representation. Use a second
    # nanosecond representation only for temporal continuity arithmetic.
    timestamps = ts_series.to_numpy()

    ts_ns = (
        ts_series.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    n_rows = len(df)
    max_start = n_rows - lookback - horizon + 1

    if max_start <= 0:
        raise ValueError(
            f"No hay suficientes filas para generar ventanas con "
            f"lookback={lookback} y horizon={horizon}. "
            f"Filas disponibles: {n_rows}"
        )

    expected_step_ns = 3_600_000_000_000
    step_ns = np.diff(ts_ns)
    good_step = step_ns == expected_step_ns

    bad_prefix = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum((~good_step).astype(np.int64)),
        ]
    )

    candidate_starts = np.arange(max_start, dtype=np.int64)
    target_indices = candidate_starts + lookback + horizon - 1
    bad_counts = (
        bad_prefix[target_indices]
        - bad_prefix[candidate_starts]
    )
    valid_starts = candidate_starts[bad_counts == 0]

    if valid_starts.size == 0:
        raise ValueError(
            "El control estricto de continuidad temporal eliminó todas las ventanas."
        )

    x_windows: List[np.ndarray] = []
    y_values: List[float] = []
    y_timestamps: List[np.datetime64] = []
    baseline_values: List[float] = []
    train_target_values: List[float] = []

    for start_idx_np in valid_starts:
        start_idx = int(start_idx_np)
        end_idx = start_idx + lookback
        target_idx = end_idx + horizon - 1

        future_price = float(targets[target_idx])
        baseline_price = float(targets[end_idx - 1])

        x_windows.append(features[start_idx:end_idx])
        y_values.append(future_price)
        y_timestamps.append(timestamps[target_idx])
        baseline_values.append(baseline_price)
        train_target_values.append(
            build_train_target(
                future_price,
                baseline_price,
                validated_target_mode,
            )
        )

    x_raw = np.stack(x_windows).astype(np.float32)
    y_raw = np.asarray(
        y_values,
        dtype=np.float32,
    ).reshape(-1, 1)
    ts_array = np.asarray(y_timestamps)
    baseline_raw = np.asarray(
        baseline_values,
        dtype=np.float32,
    ).reshape(-1, 1)
    train_target_raw = np.asarray(
        train_target_values,
        dtype=np.float32,
    ).reshape(-1, 1)

    return WindowedSplit(
        x_raw=x_raw,
        y_raw=y_raw,
        timestamps=ts_array,
        baseline_raw=baseline_raw,
        train_target_raw=train_target_raw,
        target_mode=validated_target_mode,
    )


"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not COMMON_PATH.exists():
        raise FileNotFoundError(COMMON_PATH)

    MIGRATION_DIR.mkdir(parents=True, exist_ok=True)
    original = COMMON_PATH.read_text(encoding="utf-8")

    if MARKER in original:
        py_compile.compile(str(COMMON_PATH), doraise=True)
        print("El pipeline oficial ya contiene el filtro estricto.")
        print(f"Archivo: {COMMON_PATH}")
        print("py_compile: OK")
        return

    start_match = re.search(r"(?m)^def build_windows\(", original)
    end_match = re.search(r"(?m)^def fit_scalers\(", original)

    if start_match is None or end_match is None:
        raise RuntimeError(
            "No se pudo localizar build_windows() / fit_scalers(). "
            "No se modifica ningún archivo."
        )

    if end_match.start() <= start_match.start():
        raise RuntimeError("Orden de funciones inesperado.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = (
        MIGRATION_DIR
        / f"10_common_training_pipeline__PRE_STRICT__{timestamp}.py"
    )
    shutil.copy2(COMMON_PATH, backup)

    before_hash = sha256(COMMON_PATH)

    patched = (
        original[: start_match.start()]
        + STRICT_FUNCTION
        + original[end_match.start():]
    )

    temp = COMMON_PATH.with_suffix(".STRICT_TMP.py")
    temp.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(temp), doraise=True)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    temp.replace(COMMON_PATH)
    after_hash = sha256(COMMON_PATH)

    py_compile.compile(str(COMMON_PATH), doraise=True)

    final_text = COMMON_PATH.read_text(encoding="utf-8")
    if MARKER not in final_text:
        raise RuntimeError("El marcador estricto no quedó instalado.")

    manifest = {
        "status": "PATCH_APPLIED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_pipeline": str(COMMON_PATH),
        "backup": str(backup),
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "marker": MARKER,
        "rule": (
            "Every timestamp transition from the first input observation "
            "through the target observation must be exactly +1 hour."
        ),
        "lookback_interpretation": (
            "For lookback=48 and horizon=1: 48 observations span 47 hours "
            "and target occurs exactly 1 hour after the last input."
        ),
    }

    manifest_path = MIGRATION_DIR / "strict_patch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 96)
    print("FILTRO ESTRICTO INSTALADO EN EL PIPELINE OFICIAL")
    print("=" * 96)
    print(f"Pipeline : {COMMON_PATH}")
    print(f"Backup   : {backup}")
    print(f"SHA antes: {before_hash}")
    print(f"SHA nuevo: {after_hash}")
    print("py_compile: OK")
    print(f"Manifest : {manifest_path}")
    print()
    print("SIGUIENTE PASO:")
    print("Ejecuta 34_verificar_pipeline_estricto_oficial.py antes de reentrenar.")


if __name__ == "__main__":
    main()
