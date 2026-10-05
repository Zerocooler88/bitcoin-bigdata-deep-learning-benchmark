from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

COMMON_PATH = SRC_DIR / "10_common_training_pipeline.py"
ORCHESTRATOR_PATH = SRC_DIR / "16_run_experimentos.py"

FINAL_DIR = PROJECT_ROOT / "outputs" / "scenarios_final_seed42"
MANIFEST_DIR = FINAL_DIR / "manifest"
ARCHIVE_METRICS_DIR = FINAL_DIR / "metrics"
ARCHIVE_PREDICTIONS_DIR = FINAL_DIR / "predictions"
ARCHIVE_HISTORY_DIR = FINAL_DIR / "history"
ARCHIVE_MODELS_DIR = FINAL_DIR / "models"
ARCHIVE_CONSOLIDATED_DIR = FINAL_DIR / "consolidated"

EXPECTED_CUBLAS = ":4096:8"
FINAL_SEED = 42

MODEL_KEYS = [
    "baseline_persistencia",
    "lstm",
    "gru",
    "cnn1d",
    "cnn_lstm",
]

NEURAL_MODELS = [
    "lstm",
    "gru",
    "cnn1d",
    "cnn_lstm",
]

EXPECTED_CONFIG = {
    "lookback": 48,
    "horizon": 1,
    "batch_size": 64,
    "epochs": 50,
    "patience": 5,
    "learning_rate": 0.001,
    "num_workers": 0,
    "random_seed": FINAL_SEED,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "max_grad_norm": 1.0,
}

MTIME_TOLERANCE_SECONDS = 5.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_common():
    spec = importlib.util.spec_from_file_location(
        "common_final13_seed42",
        COMMON_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["common_final13_seed42"] = module
    spec.loader.exec_module(module)
    return module


def probe_cuda() -> dict:
    code = (
        "import json, torch; "
        "print(json.dumps({"
        "'cuda_available': bool(torch.cuda.is_available()),"
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'torch': torch.__version__,"
        "'cuda': torch.version.cuda"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "No se pudo comprobar CUDA.\n"
            + result.stdout
            + "\n"
            + result.stderr
        )
    return json.loads(result.stdout.strip())


def backup_previous_final_dir() -> Path | None:
    if not FINAL_DIR.exists():
        return None

    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = FINAL_DIR.with_name(
        f"{FINAL_DIR.name}__backup_{token}"
    )
    FINAL_DIR.rename(backup)
    return backup


def ensure_dirs() -> None:
    for path in [
        FINAL_DIR,
        MANIFEST_DIR,
        ARCHIVE_METRICS_DIR,
        ARCHIVE_PREDICTIONS_DIR,
        ARCHIVE_HISTORY_DIR,
        ARCHIVE_MODELS_DIR,
        ARCHIVE_CONSOLIDATED_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def copy_with_manifest(
    source: Path,
    destination: Path,
    manifest_rows: list[dict],
    role: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manifest_rows.append(
        {
            "role": role,
            "source_relative": source.relative_to(PROJECT_ROOT).as_posix(),
            "archive_relative": destination.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": int(source.stat().st_size),
            "sha256": sha256_file(source),
        }
    )


def validate_numeric(actual, expected, label: str, tol: float = 1e-12) -> None:
    if actual is None:
        raise AssertionError(f"Falta config.{label}")
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(
            f"config.{label}={actual!r}; esperado={expected!r}"
        )


def find_latest_orchestrator_summary(run_start: float, suffix: str) -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "outputs" / "metrics").glob(
            f"run_experimentos__resumen_consolidado__*.{suffix}"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.stat().st_mtime >= run_start - MTIME_TOLERANCE_SECONDS:
            return path
    raise FileNotFoundError(
        f"No se encontró resumen del orquestador .{suffix} generado en esta corrida."
    )


def optional_compare_archived_global_seed42(
    scenario_display: str,
) -> list[dict]:
    """
    Compara, si existe, la predicción global recién generada con el archivo
    seed_42 de la corrida multiseed final. No es requisito para completar
    la generación si el archivo archivado no existe.
    """
    base = PROJECT_ROOT / "outputs" / "multiseed_final_deterministic"
    rows = []

    if not base.exists():
        return rows

    for model in NEURAL_MODELS:
        fresh = (
            PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{model}__{scenario_display}__test.csv"
        )

        candidates = [
            p for p in base.rglob(f"{model}__{scenario_display}__test.csv")
            if "seed_42" in p.as_posix().lower()
            or "seed42" in p.as_posix().lower()
        ]

        if len(candidates) != 1 or not fresh.exists():
            rows.append(
                {
                    "model": model,
                    "status": "not_uniquely_found",
                    "archived_path": (
                        "; ".join(str(p) for p in candidates)
                        if candidates
                        else ""
                    ),
                    "same_shape": None,
                    "max_abs_prediction_diff": None,
                }
            )
            continue

        archived = candidates[0]
        a = pd.read_csv(fresh)
        b = pd.read_csv(archived)

        same_shape = a.shape == b.shape

        if same_shape and "valor_predicho" in a.columns and "valor_predicho" in b.columns:
            max_diff = float(
                (
                    pd.to_numeric(a["valor_predicho"], errors="coerce")
                    - pd.to_numeric(b["valor_predicho"], errors="coerce")
                ).abs().max()
            )
        else:
            max_diff = None

        status = (
            "match"
            if same_shape and max_diff is not None and max_diff <= 1e-9
            else "different"
        )

        rows.append(
            {
                "model": model,
                "status": status,
                "archived_path": str(archived),
                "same_shape": same_shape,
                "max_abs_prediction_diff": max_diff,
            }
        )

    return rows


def main() -> None:
    print("=" * 108)
    print("GENERACION DEFINITIVA - 13 ESCENARIOS - SEED 42 - BENCHMARK Q1")
    print("=" * 108)

    # Fijar antes de cargar el pipeline/PyTorch de los procesos hijos.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS
    os.environ["PYTHONHASHSEED"] = str(FINAL_SEED)

    if not COMMON_PATH.exists():
        raise FileNotFoundError(COMMON_PATH)
    if not ORCHESTRATOR_PATH.exists():
        raise FileNotFoundError(ORCHESTRATOR_PATH)

    common = load_common()
    config = common.ExperimentConfig()

    canonical_features = list(config.feature_columns or [])
    if len(canonical_features) != 22:
        raise AssertionError(
            f"Se esperaban 22 features canónicas; se encontraron {len(canonical_features)}."
        )

    catalog = common.build_default_scenarios(project_root=PROJECT_ROOT)
    if len(catalog) != 13:
        raise AssertionError(
            f"Se esperaban 13 escenarios; el catálogo contiene {len(catalog)}."
        )

    scenarios = list(catalog.values())
    scenario_names = [s.display_name for s in scenarios]

    global_path = (
        PROJECT_ROOT
        / "data"
        / "1h"
        / "global"
        / "btcusdt_spot_1h_model_input_2019_2026.parquet"
    )
    if not global_path.exists():
        raise FileNotFoundError(
            f"No existe el parquet global:\n{global_path}"
        )

    cuda = probe_cuda()
    if not cuda["cuda_available"]:
        raise RuntimeError(
            "CUDA no está disponible. Se cancela para no generar una versión "
            "definitiva con un entorno distinto del benchmark Q1."
        )

    print(f"Features canónicas : {len(canonical_features)}")
    print(f"Escenarios         : {len(scenarios)}")
    print(f"Seed final         : {FINAL_SEED}")
    print(f"CUBLAS             : {os.environ['CUBLAS_WORKSPACE_CONFIG']}")
    print(f"CUDA               : {cuda['cuda_available']}")
    print(f"GPU                : {cuda['device']}")
    print(f"PyTorch            : {cuda['torch']}")
    print(f"CUDA PyTorch       : {cuda['cuda']}")
    print()
    print("Escenarios oficiales:")
    for index, scenario in enumerate(scenarios, start=1):
        print(f"{index:02d}. {scenario.display_name}")

    previous_backup = backup_previous_final_dir()
    ensure_dirs()

    if previous_backup is not None:
        print()
        print("Backup de final anterior:")
        print(previous_backup)

    command = [
        sys.executable,
        str(ORCHESTRATOR_PATH),
        "--models",
        "all",
        "--scenarios",
        "all",
        "--lookback",
        "48",
        "--horizon",
        "1",
        "--batch-size",
        "64",
        "--epochs",
        "50",
        "--patience",
        "5",
        "--learning-rate",
        "0.001",
        "--target-mode",
        "delta",
        "--num-workers",
        "0",
        "--seed",
        str(FINAL_SEED),
    ]

    run_start_wall = time.time()
    run_start_utc = datetime.now(timezone.utc).isoformat()

    print()
    print("=" * 108)
    print("INICIANDO ORQUESTADOR OFICIAL")
    print("=" * 108)
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print()

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=False,
    )

    run_end_wall = time.time()
    run_end_utc = datetime.now(timezone.utc).isoformat()
    total_duration = run_end_wall - run_start_wall

    if completed.returncode != 0:
        raise RuntimeError(
            "El orquestador terminó con código distinto de cero. "
            "NO se declara cerrada la generación definitiva. "
            "Revisa el error mostrado arriba."
        )

    print()
    print("=" * 108)
    print("AUDITORIA POST-EJECUCION")
    print("=" * 108)

    manifest_rows: list[dict] = []
    metric_rows: list[dict] = []

    expected_metrics_json = 0
    expected_metrics_csv = 0
    expected_predictions = 0
    expected_histories = 0
    expected_weights = 0

    for model in MODEL_KEYS:
        for scenario in scenarios:
            prefix = f"{model}__{scenario.display_name}"

            metrics_json = (
                PROJECT_ROOT / "outputs" / "metrics" / f"{prefix}__metrics.json"
            )
            metrics_csv = (
                PROJECT_ROOT / "outputs" / "metrics" / f"{prefix}__metrics.csv"
            )
            pred_val = (
                PROJECT_ROOT
                / "outputs"
                / "predictions"
                / f"{prefix}__validation.csv"
            )
            pred_test = (
                PROJECT_ROOT
                / "outputs"
                / "predictions"
                / f"{prefix}__test.csv"
            )

            for required in [metrics_json, metrics_csv, pred_val, pred_test]:
                if not required.exists():
                    raise FileNotFoundError(
                        f"Falta artefacto obligatorio:\n{required}"
                    )
                if required.stat().st_mtime < (
                    run_start_wall - MTIME_TOLERANCE_SECONDS
                ):
                    raise RuntimeError(
                        "El artefacto existe pero no parece pertenecer a esta corrida:\n"
                        f"{required}"
                    )

            expected_metrics_json += 1
            expected_metrics_csv += 1
            expected_predictions += 2

            payload = json.loads(
                metrics_json.read_text(encoding="utf-8")
            )

            features = payload.get("feature_columns", [])
            if features != canonical_features:
                raise AssertionError(
                    f"{prefix}: feature_columns no coincide con las 22 canónicas."
                )

            target_mode = payload.get("training_target_mode")
            expected_target = "price" if model == "baseline_persistencia" else "delta"
            if target_mode != expected_target:
                raise AssertionError(
                    f"{prefix}: target_mode={target_mode!r}; esperado={expected_target!r}"
                )

            cfg = payload.get("config", {})
            for key, expected in EXPECTED_CONFIG.items():
                validate_numeric(cfg.get(key), expected, f"{prefix}.{key}")

            if cfg.get("device") != "cuda":
                raise AssertionError(
                    f"{prefix}: device={cfg.get('device')!r}; esperado='cuda'."
                )

            validation = payload.get("validation", {})
            test = payload.get("test", {})
            baseline_test = payload.get("baseline_test", {})

            baseline_rmse = float(baseline_test["RMSE"])
            model_rmse = float(test["RMSE"])
            delta_rmse = model_rmse - baseline_rmse
            delta_pct = (
                100.0 * delta_rmse / baseline_rmse
                if baseline_rmse != 0
                else float("nan")
            )

            metric_rows.append(
                {
                    "model": model,
                    "scenario": scenario.display_name,
                    "scenario_key": scenario.key,
                    "scenario_mode": scenario.mode,
                    "seed": FINAL_SEED,
                    "feature_count": len(features),
                    "target_mode": target_mode,
                    "best_epoch": int(payload.get("best_epoch", 0)),
                    "device": cfg.get("device"),
                    "val_MAE": validation.get("MAE"),
                    "val_MSE": validation.get("MSE"),
                    "val_RMSE": validation.get("RMSE"),
                    "val_MAPE": validation.get("MAPE"),
                    "val_R2": validation.get("R2"),
                    "test_MAE": test.get("MAE"),
                    "test_MSE": test.get("MSE"),
                    "test_RMSE": test.get("RMSE"),
                    "test_MAPE": test.get("MAPE"),
                    "test_R2": test.get("R2"),
                    "baseline_test_MAE": baseline_test.get("MAE"),
                    "baseline_test_MSE": baseline_test.get("MSE"),
                    "baseline_test_RMSE": baseline_test.get("RMSE"),
                    "baseline_test_MAPE": baseline_test.get("MAPE"),
                    "baseline_test_R2": baseline_test.get("R2"),
                    "delta_RMSE_vs_persistence": delta_rmse,
                    "delta_RMSE_pct_vs_persistence": delta_pct,
                }
            )

            copy_with_manifest(
                metrics_json,
                ARCHIVE_METRICS_DIR / metrics_json.name,
                manifest_rows,
                "metrics_json",
            )
            copy_with_manifest(
                metrics_csv,
                ARCHIVE_METRICS_DIR / metrics_csv.name,
                manifest_rows,
                "metrics_csv",
            )
            copy_with_manifest(
                pred_val,
                ARCHIVE_PREDICTIONS_DIR / pred_val.name,
                manifest_rows,
                "prediction_validation",
            )
            copy_with_manifest(
                pred_test,
                ARCHIVE_PREDICTIONS_DIR / pred_test.name,
                manifest_rows,
                "prediction_test",
            )

            if model in NEURAL_MODELS:
                history = (
                    PROJECT_ROOT
                    / "outputs"
                    / "history"
                    / f"{prefix}__history.csv"
                )
                weight = (
                    PROJECT_ROOT
                    / "models"
                    / model
                    / f"{prefix}__best.pt"
                )

                for required in [history, weight]:
                    if not required.exists():
                        raise FileNotFoundError(
                            f"Falta artefacto neural obligatorio:\n{required}"
                        )
                    if required.stat().st_mtime < (
                        run_start_wall - MTIME_TOLERANCE_SECONDS
                    ):
                        raise RuntimeError(
                            "El artefacto neural existe pero no parece pertenecer "
                            f"a esta corrida:\n{required}"
                        )

                expected_histories += 1
                expected_weights += 1

                copy_with_manifest(
                    history,
                    ARCHIVE_HISTORY_DIR / history.name,
                    manifest_rows,
                    "training_history",
                )
                copy_with_manifest(
                    weight,
                    ARCHIVE_MODELS_DIR / model / weight.name,
                    manifest_rows,
                    "best_weights",
                )

    # Consolidated summaries per model.
    for model in MODEL_KEYS:
        for suffix in ["json", "csv"]:
            source = (
                PROJECT_ROOT
                / "outputs"
                / "metrics"
                / f"{model}__resumen_consolidado.{suffix}"
            )
            if not source.exists():
                raise FileNotFoundError(
                    f"Falta resumen consolidado de modelo:\n{source}"
                )
            if source.stat().st_mtime < (
                run_start_wall - MTIME_TOLERANCE_SECONDS
            ):
                raise RuntimeError(
                    f"Resumen consolidado antiguo:\n{source}"
                )
            copy_with_manifest(
                source,
                ARCHIVE_CONSOLIDATED_DIR / source.name,
                manifest_rows,
                "model_consolidated_summary",
            )

    latest_orch_json = find_latest_orchestrator_summary(
        run_start_wall, "json"
    )
    latest_orch_csv = find_latest_orchestrator_summary(
        run_start_wall, "csv"
    )

    copy_with_manifest(
        latest_orch_json,
        ARCHIVE_CONSOLIDATED_DIR / latest_orch_json.name,
        manifest_rows,
        "orchestrator_summary",
    )
    copy_with_manifest(
        latest_orch_csv,
        ARCHIVE_CONSOLIDATED_DIR / latest_orch_csv.name,
        manifest_rows,
        "orchestrator_summary",
    )

    metrics_df = pd.DataFrame(metric_rows)

    expected_rows = len(MODEL_KEYS) * len(scenarios)
    if len(metrics_df) != expected_rows:
        raise AssertionError(
            f"Tabla final tiene {len(metrics_df)} filas; esperado={expected_rows}."
        )

    # Ganador por RMSE de test en cada escenario.
    winner_rows = []
    for scenario_name, group in metrics_df.groupby(
        "scenario", sort=False
    ):
        ranked = group.sort_values(
            ["test_RMSE", "model"],
            ascending=[True, True],
        ).reset_index(drop=True)

        winner = ranked.iloc[0]
        runner_up = ranked.iloc[1]

        winner_rows.append(
            {
                "scenario": scenario_name,
                "winner_model": winner["model"],
                "winner_RMSE": winner["test_RMSE"],
                "runner_up_model": runner_up["model"],
                "runner_up_RMSE": runner_up["test_RMSE"],
                "margin_RMSE": (
                    float(runner_up["test_RMSE"])
                    - float(winner["test_RMSE"])
                ),
                "winner_delta_RMSE_vs_persistence": winner[
                    "delta_RMSE_vs_persistence"
                ],
                "winner_delta_RMSE_pct_vs_persistence": winner[
                    "delta_RMSE_pct_vs_persistence"
                ],
            }
        )

    winners_df = pd.DataFrame(winner_rows)

    metric_table_path = (
        MANIFEST_DIR / "final_13_scenarios_seed42_metrics.csv"
    )
    winners_path = (
        MANIFEST_DIR / "final_13_scenarios_seed42_winners.csv"
    )
    metrics_df.to_csv(
        metric_table_path,
        index=False,
        encoding="utf-8-sig",
    )
    winners_df.to_csv(
        winners_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Comparación opcional con el seed42 global del bloque multiseed.
    global_display = catalog["global"].display_name
    global_compare_rows = optional_compare_archived_global_seed42(
        global_display
    )
    global_compare_path = (
        MANIFEST_DIR
        / "global_seed42_vs_multiseed_archive.csv"
    )
    pd.DataFrame(global_compare_rows).to_csv(
        global_compare_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Manifest SHA de los artefactos archivados.
    sha_manifest_path = MANIFEST_DIR / "sha256_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(
        sha_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    winner_counts = (
        winners_df["winner_model"]
        .value_counts()
        .reindex(MODEL_KEYS, fill_value=0)
        .to_dict()
    )

    audit_payload = {
        "status": "COMPLETE",
        "run_start_utc": run_start_utc,
        "run_end_utc": run_end_utc,
        "duration_seconds": total_duration,
        "seed": FINAL_SEED,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "python_executable": sys.executable,
        "cuda_probe": cuda,
        "canonical_feature_count": len(canonical_features),
        "canonical_features": canonical_features,
        "scenario_count": len(scenarios),
        "scenarios": [
            {
                "key": s.key,
                "display_name": s.display_name,
                "mode": s.mode,
                "metadata": s.metadata,
            }
            for s in scenarios
        ],
        "model_count": len(MODEL_KEYS),
        "models": MODEL_KEYS,
        "expected_combinations": expected_rows,
        "metrics_json_count": expected_metrics_json,
        "metrics_csv_count": expected_metrics_csv,
        "prediction_csv_count": expected_predictions,
        "history_csv_count": expected_histories,
        "weights_pt_count": expected_weights,
        "winner_counts_by_test_rmse": winner_counts,
        "orchestrator_return_code": completed.returncode,
        "global_seed42_multiseed_comparison": global_compare_rows,
        "final_archive": str(FINAL_DIR),
    }

    audit_path = (
        MANIFEST_DIR
        / "final_13_scenarios_seed42_audit.json"
    )
    audit_path.write_text(
        json.dumps(
            audit_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Hash de las tablas/manifiestos generados por el propio cierre.
    closure_files = [
        metric_table_path,
        winners_path,
        global_compare_path,
        sha_manifest_path,
        audit_path,
    ]
    closure_hashes = []
    for path in closure_files:
        closure_hashes.append(
            {
                "file": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(closure_hashes).to_csv(
        MANIFEST_DIR / "closure_files_sha256.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 108)
    print("CIERRE DEFINITIVO - RESULTADO")
    print("=" * 108)
    print(f"Estado                         : COMPLETE")
    print(f"Escenarios                     : {len(scenarios)} / 13")
    print(f"Métodos                        : {len(MODEL_KEYS)} / 5")
    print(f"Combinaciones auditadas        : {len(metrics_df)} / 65")
    print(f"JSON métricas                  : {expected_metrics_json} / 65")
    print(f"CSV métricas                   : {expected_metrics_csv} / 65")
    print(f"CSV predicciones               : {expected_predictions} / 130")
    print(f"Historias neurales             : {expected_histories} / 52")
    print(f"Pesos neurales                 : {expected_weights} / 52")
    print(f"Features en cada combinación   : 22")
    print(f"Seed                           : {FINAL_SEED}")
    print(f"Device                         : cuda")
    print(f"CUBLAS                         : {EXPECTED_CUBLAS}")
    print(f"Duración total                 : {total_duration:.2f} s")
    print()
    print("Ganadores por RMSE de test:")
    for model in MODEL_KEYS:
        print(f"- {model:22s}: {winner_counts.get(model, 0)}")
    print()
    print("Archivo oficial final:")
    print(FINAL_DIR)
    print()
    print("Tabla de métricas:")
    print(metric_table_path)
    print()
    print("Tabla de ganadores:")
    print(winners_path)
    print()
    print("Auditoría:")
    print(audit_path)
    print()
    print("IMPORTANTE:")
    print(
        "Solo si el estado anterior es COMPLETE deben usarse estos "
        "13 escenarios para actualizar el manuscrito."
    )


if __name__ == "__main__":
    main()
