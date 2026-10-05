from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUT_ROOT = PROJECT_ROOT / "outputs"
PACKAGE_DIR = OUT_ROOT / "reproducibility_package_q1"
ZIP_PATH = OUT_ROOT / "Bitcoin_Benchmark_Q1_reproducibility_package.zip"

SEEDS = [11, 23, 42, 84, 168]

KEY_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scikit-learn",
    "pyspark",
    "boto3",
    "matplotlib",
    "openpyxl",
    "tqdm",
]

INCLUDE_GLOBS = [
    "src/*.py",
    "requirements_q1_frozen.txt",
    "README_PRIMERO.txt",
    ".gitignore",
    "outputs/reproducibility/*.csv",
    "outputs/reproducibility/*.json",
    "outputs/reproducibility/*.txt",
    "outputs/statistical_analysis/*.csv",
    "outputs/statistical_analysis/*.json",
    "outputs/performance/*.csv",
    "outputs/performance/*.json",
]

HASH_ONLY_CANDIDATES = [
    "data/1h/global/btcusdt_spot_1h_model_input_2019_2026.parquet",
    "data/gold_lab/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_lab_2019_2026.parquet",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_common_pipeline():
    path = SRC_DIR / "10_common_training_pipeline.py"
    spec = importlib.util.spec_from_file_location("common_pipeline_repro", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["common_pipeline_repro"] = module
    spec.loader.exec_module(module)
    return module


def safe_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            return value if value else None
    except Exception:
        pass
    return None


def package_versions() -> dict[str, str]:
    versions = {}
    for pkg in KEY_PACKAGES:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = "NOT_INSTALLED"
    return versions


def gpu_info() -> dict:
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(props.total_memory),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
    else:
        info.update(
            {
                "gpu_name": None,
                "gpu_total_memory_bytes": None,
                "gpu_compute_capability": None,
            }
        )

    return info


def collect_included_files() -> list[Path]:
    files: set[Path] = set()

    for pattern in INCLUDE_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file():
                files.add(path.resolve())

    return sorted(
        p for p in files
        if p.resolve() != ZIP_PATH.resolve()
    )


def write_manifest_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "relative_path",
        "sha256",
        "size_bytes",
        "included_in_zip",
        "role",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 100)
    print("GENERADOR DE PAQUETE DE REPRODUCIBILIDAD Q1")
    print("=" * 100)

    common = load_common_pipeline()
    config = common.ExperimentConfig()

    features = list(config.feature_columns or [])
    if len(features) != 22:
        raise RuntimeError(
            f"Se esperaban 22 features canonicas y se encontraron {len(features)}."
        )

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    included_files = collect_included_files()

    manifest_rows = []

    for path in included_files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        manifest_rows.append(
            {
                "relative_path": rel,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "included_in_zip": True,
                "role": "source_or_small_result",
            }
        )

    for rel_str in HASH_ONLY_CANDIDATES:
        path = PROJECT_ROOT / rel_str
        if path.exists() and path.is_file():
            manifest_rows.append(
                {
                    "relative_path": rel_str,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "included_in_zip": False,
                    "role": "large_data_hash_only",
                }
            )

    git_commit = safe_git(["rev-parse", "HEAD"])
    git_branch = safe_git(["rev-parse", "--abbrev-ref", "HEAD"])
    git_status = safe_git(["status", "--porcelain"])

    environment = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "gpu": gpu_info(),
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "working_tree_clean": (
                None if git_status is None else git_status == ""
            ),
        },
    }

    config_dict = asdict(config)
    config_dict["project_root"] = "<PROJECT_ROOT>"
    config_dict["feature_columns"] = features

    protocol = {
        "benchmark_title": (
            "A reproducible Big Data benchmark for Bitcoin price forecasting "
            "using deep learning models and a persistence baseline"
        ),
        "symbol": "BTCUSDT",
        "market": "Binance Spot",
        "frequency": "1h",
        "models": [
            "Persistence",
            "LSTM",
            "GRU",
            "CNN-1D",
            "CNN-LSTM",
        ],
        "seeds_global_robustness": SEEDS,
        "canonical_feature_count": len(features),
        "canonical_features": features,
        "experiment_config_defaults": config_dict,
        "training_target_mode_neural_models": "delta",
        "persistence_target": "price",
        "loss_function": "MSELoss",
        "optimizer": "Adam",
        "deterministic_execution": {
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config_runtime": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
        "metrics": ["MAE", "MSE", "RMSE", "MAPE", "R2"],
        "statistical_analysis": {
            "comparison_baseline": "Persistence",
            "diebold_mariano_loss": "squared error",
            "hac_lag_hours": 168,
            "hierarchical_block_bootstrap_repetitions": 5000,
            "block_length_hours": 168,
            "multiple_testing_correction": "Holm",
        },
        "bigdata_lab_scope": {
            "warmup_excluded": True,
            "measured_repetitions": 3,
            "horizontal_scalability_claimed": False,
            "interpretation": (
                "steady-state Spark/MinIO laboratory performance and throughput"
            ),
        },
    }

    env_path = PACKAGE_DIR / "environment_manifest.json"
    protocol_path = PACKAGE_DIR / "experimental_protocol.json"
    manifest_path = PACKAGE_DIR / "sha256_manifest.csv"
    readme_path = PACKAGE_DIR / "README_REPRODUCIBILITY.md"

    env_path.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_manifest_csv(manifest_rows, manifest_path)

    readme = f"""# Bitcoin Benchmark Q1 — Reproducibility Package

## Scope

BTCUSDT Spot, hourly frequency, Persistence, LSTM, GRU, CNN-1D and CNN-LSTM.

## Canonical predictors

Exactly 22 predictor variables are enforced by the common training pipeline.

## Global multi-seed robustness

Seeds: {SEEDS}

## Core training configuration

- Lookback: {config.lookback}
- Forecast horizon: {config.horizon}
- Batch size: {config.batch_size}
- Maximum epochs: {config.epochs}
- Early-stopping patience: {config.patience}
- Learning rate: {config.learning_rate}
- Gradient clipping max norm: {config.max_grad_norm}
- Train/validation/test ratios: {config.train_ratio}/{config.val_ratio}/{config.test_ratio}
- Optimizer: Adam
- Loss: MSE

## Deterministic execution

The runtime environment, CUDA/cuDNN versions, GPU model, package versions,
and `CUBLAS_WORKSPACE_CONFIG` are recorded in `environment_manifest.json`.

## Statistical analysis

The global comparison against persistence uses squared-error Diebold-Mariano
tests with HAC adjustment, Holm correction, and a hierarchical block bootstrap.

## Big Data benchmark scope

The Spark/MinIO performance benchmark excludes one warm-up repetition and reports
three steady-state repetitions. It measures the laboratory configuration and does
not demonstrate horizontal scalability.

## Large data

Large parquet datasets are not bundled in the ZIP. When present locally, their
SHA-256 hashes are recorded in `sha256_manifest.csv`.

## Public-release warning

Before publication, inspect the source files and replace any local credentials,
host-specific paths, or secrets with environment variables. Do not publish secrets.
"""
    readme_path.write_text(readme, encoding="utf-8")

    project_files_dir = PACKAGE_DIR / "project_files"

    for path in included_files:
        rel = path.relative_to(PROJECT_ROOT)
        destination = project_files_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for path in PACKAGE_DIR.rglob("*"):
            if path.is_file():
                arcname = (
                    Path("Bitcoin_Benchmark_Q1_reproducibility_package")
                    / path.relative_to(PACKAGE_DIR)
                )
                zf.write(path, arcname.as_posix())

    zip_sha256 = sha256_file(ZIP_PATH)

    zip_hash_path = (
        OUT_ROOT
        / "Bitcoin_Benchmark_Q1_reproducibility_package.sha256.txt"
    )
    zip_hash_path.write_text(
        f"{zip_sha256}  {ZIP_PATH.name}\n",
        encoding="utf-8",
    )

    print()
    print(f"Features canonicas : {len(features)}")
    print(f"Seeds              : {SEEDS}")
    print(f"max_grad_norm      : {config.max_grad_norm}")
    print(
        "CUBLAS config      : "
        f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
    )
    print(f"Git commit         : {git_commit}")
    print(f"Git branch         : {git_branch}")
    print(f"Archivos auditados : {len(manifest_rows)}")
    print()
    print("Paquete generado:")
    print(ZIP_PATH)
    print()
    print("SHA-256 del ZIP:")
    print(zip_sha256)
    print()
    print("Manifiestos:")
    print(env_path)
    print(protocol_path)
    print(manifest_path)
    print(zip_hash_path)
    print()
    print("IMPORTANTE:")
    print("Revisa el ZIP antes de publicarlo.")
    print("No publiques credenciales ni rutas personales sin sanitizarlas.")


if __name__ == "__main__":
    main()
