from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs"

SCENARIO_FINAL = OUT_ROOT / "scenarios_final_seed42"
SCENARIO_MANIFEST = SCENARIO_FINAL / "manifest"
MULTISEED_FINAL = OUT_ROOT / "multiseed_final_deterministic"

PACKAGE_DIR = OUT_ROOT / "reproducibility_package_q1_FINAL"
ZIP_PATH = OUT_ROOT / "Bitcoin_Benchmark_Q1_reproducibility_package_FINAL.zip"
ZIP_HASH_PATH = OUT_ROOT / "Bitcoin_Benchmark_Q1_reproducibility_package_FINAL.sha256.txt"

EXPECTED_MODELS = ["lstm", "gru", "cnn1d", "cnn_lstm"]
EXPECTED_ALL_METHODS = [
    "baseline_persistencia", "lstm", "gru", "cnn1d", "cnn_lstm"
]
EXPECTED_SCENARIOS = 13
EXPECTED_COMBINATIONS = 65
EXPECTED_SEED = 42
EXPECTED_FEATURES = 22
EXPECTED_CUBLAS = ":4096:8"

KEY_PACKAGES = [
    "torch", "numpy", "pandas", "polars", "pyarrow",
    "scikit-learn", "pyspark", "boto3", "matplotlib",
    "openpyxl", "tqdm",
]

# Archivos que deben entrar al paquete final.
INCLUDE_PATTERNS = [
    "src/*.py",
    "requirements_q1_frozen.txt",
    "README_PRIMERO.txt",
    ".gitignore",

    "outputs/reproducibility/*.csv",
    "outputs/reproducibility/*.json",
    "outputs/reproducibility/*.txt",

    "outputs/statistical_analysis/*.csv",
    "outputs/statistical_analysis/*.json",
    "outputs/statistical_analysis/*.txt",

    "outputs/performance/*.csv",
    "outputs/performance/*.json",
    "outputs/performance/*.txt",

    "outputs/scenarios_final_seed42/manifest/*.csv",
    "outputs/scenarios_final_seed42/manifest/*.json",
    "outputs/scenarios_final_seed42/manifest/*.txt",

    "outputs/scenarios_final_seed42/metrics/*.csv",
    "outputs/scenarios_final_seed42/metrics/*.json",

    "outputs/scenarios_final_seed42/consolidated/*.csv",
    "outputs/scenarios_final_seed42/consolidated/*.json",

    # Las predicciones/history/pesos definitivos por escenario
    # permiten regenerar tablas y figuras sin reentrenar.
    "outputs/scenarios_final_seed42/predictions/*.csv",
    "outputs/scenarios_final_seed42/history/*.csv",
    "outputs/scenarios_final_seed42/models/**/*.pt",

    # Robustez global de cinco seeds: resultados y predicciones.
    "outputs/multiseed_final_deterministic/**/*.csv",
    "outputs/multiseed_final_deterministic/**/*.json",
    "outputs/multiseed_final_deterministic/**/*.txt",
]

# El parquet global es la fuente local de los escenarios finales.
DATA_CANDIDATES = [
    "data/1h/global/btcusdt_spot_1h_model_input_2019_2026.parquet",
]

# Detector conservador: NO imprime secretos, solo el archivo y la categoría.
SENSITIVE_PATTERNS = {
    "hardcoded_minio_default_credential": re.compile(
        r"\bminioadmin\b", re.IGNORECASE
    ),
    "windows_personal_path": re.compile(
        r"[A-Za-z]:\\Users\\[^\\\r\n]+", re.IGNORECASE
    ),
    "linux_home_path": re.compile(
        r"/home/[^/\s]+", re.IGNORECASE
    ),
    "password_assignment": re.compile(
        r"(PASSWORD|PASSWD|SECRET)\s*[:=]", re.IGNORECASE
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_git(args: list[str]) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if r.returncode == 0:
            value = r.stdout.strip()
            return value or None
    except Exception:
        pass
    return None


def package_versions() -> dict[str, str]:
    out = {}
    for pkg in KEY_PACKAGES:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "NOT_INSTALLED"
    return out


def gpu_manifest() -> dict:
    result = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(props.total_memory),
                "gpu_compute_capability": (
                    f"{props.major}.{props.minor}"
                ),
            }
        )
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_alignment() -> pd.DataFrame:
    alignment_path = (
        SCENARIO_MANIFEST / "global_seed42_vs_multiseed_archive.csv"
    )
    require(
        alignment_path.exists(),
        f"Falta el archivo de alineación:\n{alignment_path}",
    )

    df = pd.read_csv(alignment_path)

    require(
        set(df["model"]) == set(EXPECTED_MODELS),
        "La alineación global no contiene exactamente los cuatro modelos.",
    )
    require(
        (df["status"].astype(str).str.lower() == "match").all(),
        "Existe al menos un modelo global seed42 con status distinto de match.",
    )
    require(
        df["same_shape"].astype(str).str.lower().eq("true").all(),
        "Existe al menos un modelo con forma distinta.",
    )
    require(
        pd.to_numeric(
            df["max_abs_prediction_diff"], errors="raise"
        ).eq(0.0).all(),
        "Existe al menos un modelo con diferencias numéricas de predicción.",
    )

    # Auditoría adicional: compara archivo archivado de escenarios
    # vs archivo multiseed indicado en el CSV.
    detailed_rows = []

    for _, row in df.iterrows():
        model = str(row["model"])

        scenario_file = (
            SCENARIO_FINAL
            / "predictions"
            / f"{model}__global_2019_2026__test.csv"
        )
        archived_file = Path(str(row["archived_path"]))

        require(
            scenario_file.exists(),
            f"Falta predicción global final:\n{scenario_file}",
        )
        require(
            archived_file.exists(),
            f"Falta predicción multiseed seed42:\n{archived_file}",
        )

        a = pd.read_csv(scenario_file)
        b = pd.read_csv(archived_file)

        require(
            a.shape == b.shape,
            f"{model}: las tablas globales no tienen la misma forma.",
        )
        require(
            list(a.columns) == list(b.columns),
            f"{model}: las columnas globales no son idénticas.",
        )

        content_equal = a.equals(b)
        sha_a = sha256_file(scenario_file)
        sha_b = sha256_file(archived_file)

        detailed_rows.append(
            {
                "model": model,
                "scenario_file": scenario_file.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "multiseed_seed42_file": (
                    archived_file.relative_to(PROJECT_ROOT).as_posix()
                    if archived_file.is_relative_to(PROJECT_ROOT)
                    else str(archived_file)
                ),
                "same_shape": True,
                "same_columns": True,
                "dataframe_exact_equal": bool(content_equal),
                "scenario_sha256": sha_a,
                "multiseed_sha256": sha_b,
                "byte_identical": sha_a == sha_b,
                "max_abs_prediction_diff": 0.0,
            }
        )

        require(
            content_equal,
            f"{model}: el contenido CSV global no es exactamente igual.",
        )

    return pd.DataFrame(detailed_rows)


def verify_final_13() -> dict:
    audit_path = (
        SCENARIO_MANIFEST / "final_13_scenarios_seed42_audit.json"
    )
    metrics_path = (
        SCENARIO_MANIFEST / "final_13_scenarios_seed42_metrics.csv"
    )

    require(audit_path.exists(), f"Falta:\n{audit_path}")
    require(metrics_path.exists(), f"Falta:\n{metrics_path}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    require(audit.get("status") == "COMPLETE", "Audit status != COMPLETE")
    require(
        int(audit.get("scenario_count", -1)) == EXPECTED_SCENARIOS,
        "scenario_count != 13",
    )
    require(
        int(audit.get("expected_combinations", -1))
        == EXPECTED_COMBINATIONS,
        "expected_combinations != 65",
    )
    require(
        int(audit.get("canonical_feature_count", -1))
        == EXPECTED_FEATURES,
        "canonical_feature_count != 22",
    )
    require(
        int(audit.get("seed", -1)) == EXPECTED_SEED,
        "seed != 42",
    )
    require(
        audit.get("cublas_workspace_config") == EXPECTED_CUBLAS,
        "CUBLAS config final no coincide con :4096:8",
    )

    metrics = pd.read_csv(metrics_path)

    require(len(metrics) == EXPECTED_COMBINATIONS, "La tabla final no tiene 65 filas.")
    require(
        metrics["scenario"].nunique() == EXPECTED_SCENARIOS,
        "La tabla final no contiene 13 escenarios únicos.",
    )
    require(
        set(metrics["model"]) == set(EXPECTED_ALL_METHODS),
        "La tabla final no contiene exactamente los cinco métodos.",
    )
    require(
        pd.to_numeric(metrics["feature_count"]).eq(22).all(),
        "Existe una combinación con feature_count != 22.",
    )
    require(
        pd.to_numeric(metrics["seed"]).eq(42).all(),
        "Existe una combinación con seed != 42.",
    )
    require(
        metrics["device"].astype(str).str.lower().eq("cuda").all(),
        "Existe una combinación con device != cuda.",
    )

    return {
        "audit_path": audit_path,
        "metrics_path": metrics_path,
        "audit": audit,
        "metrics": metrics,
    }


def collect_files() -> list[Path]:
    files: set[Path] = set()

    for pattern in INCLUDE_PATTERNS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file():
                files.add(path.resolve())

    for rel in DATA_CANDIDATES:
        path = PROJECT_ROOT / rel
        if path.exists() and path.is_file():
            files.add(path.resolve())

    return sorted(files)


def scan_sensitive_text(files: list[Path]) -> list[dict]:
    findings = []
    text_suffixes = {
        ".py", ".txt", ".md", ".json", ".csv", ".yml", ".yaml", ".env"
    }

    for path in files:
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for category, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    {
                        "relative_path": path.relative_to(
                            PROJECT_ROOT
                        ).as_posix(),
                        "category": category,
                        "action": (
                            "Review/sanitize before public GitHub or Zenodo release"
                        ),
                    }
                )
    return findings


def main() -> None:
    print("=" * 108)
    print("VERIFICACION FINAL + REFRESCO DEL PAQUETE DE REPRODUCIBILIDAD Q1")
    print("=" * 108)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS

    final13 = verify_final_13()
    alignment = verify_alignment()

    print()
    print("ALINEACION GLOBAL SEED 42")
    print("-" * 108)
    for _, row in alignment.iterrows():
        print(
            f"{row['model']:8s} | dataframe_exact_equal="
            f"{row['dataframe_exact_equal']} | byte_identical="
            f"{row['byte_identical']} | max_abs_diff=0.0"
        )

    # Preservar un paquete FINAL anterior en caso de rerun.
    if PACKAGE_DIR.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = PACKAGE_DIR.with_name(
            f"{PACKAGE_DIR.name}__backup_{stamp}"
        )
        PACKAGE_DIR.rename(backup)
        print()
        print(f"Backup paquete FINAL anterior: {backup}")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    files = collect_files()

    manifest_rows = []
    for path in files:
        manifest_rows.append(
            {
                "relative_path": path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "included_in_zip": True,
            }
        )

    # Verificación de entorno.
    environment = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "gpu": gpu_manifest(),
        "git": {
            "commit": safe_git(["rev-parse", "HEAD"]),
            "branch": safe_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "working_tree_porcelain": safe_git(["status", "--porcelain"]),
        },
    }

    protocol = {
        "article_positioning": (
            "A reproducible Big Data benchmark for Bitcoin price forecasting "
            "using deep learning models and a persistence baseline"
        ),
        "final_13_scenario_benchmark": {
            "status": "COMPLETE",
            "seed": 42,
            "scenario_count": 13,
            "method_count": 5,
            "combination_count": 65,
            "feature_count": 22,
            "device": "cuda",
            "cublas_workspace_config": EXPECTED_CUBLAS,
        },
        "global_multiseed_robustness": {
            "seeds": [11, 23, 42, 84, 168],
            "global_seed42_exactly_aligned_with_13_scenario_run": True,
            "models_verified": EXPECTED_MODELS,
        },
        "statistical_analysis": {
            "baseline": "Persistence",
            "diebold_mariano_loss": "squared error",
            "hac_lag_hours": 168,
            "hierarchical_block_bootstrap_repetitions": 5000,
            "block_length_hours": 168,
            "multiple_testing_correction": "Holm",
        },
        "bigdata_performance": {
            "scope": "steady-state Spark/MinIO laboratory throughput",
            "warmup_excluded": True,
            "measured_repetitions": 3,
            "horizontal_scalability_claimed": False,
        },
    }

    (PACKAGE_DIR / "environment_manifest.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (PACKAGE_DIR / "experimental_protocol_FINAL.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    alignment.to_csv(
        PACKAGE_DIR / "global_seed42_alignment_verified.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(manifest_rows).to_csv(
        PACKAGE_DIR / "sha256_manifest_FINAL.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Escaneo de revisión pública.
    findings = scan_sensitive_text(files)
    findings_df = pd.DataFrame(
        findings,
        columns=["relative_path", "category", "action"],
    )
    findings_df.to_csv(
        PACKAGE_DIR / "public_release_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    readme = f"""# Bitcoin Benchmark Q1 — FINAL Reproducibility Package

## Final experimental status

- 13/13 scenarios completed.
- 5/5 methods completed.
- 65/65 scenario-method combinations audited.
- Exactly 22 predictors in every final combination.
- Final scenario benchmark seed: 42.
- Device: CUDA.
- CUBLAS_WORKSPACE_CONFIG: {EXPECTED_CUBLAS}.

## Global robustness alignment

The global seed-42 predictions from the final 13-scenario run were verified
against the archived seed-42 predictions from the five-seed robustness experiment.
All four neural models have exact DataFrame equality and zero absolute prediction
difference. `global_seed42_alignment_verified.csv` records the checks and SHA-256 values.

## Global robustness seeds

11, 23, 42, 84, 168.

## Statistical inference

The global neural-model comparisons against persistence use squared-error
Diebold-Mariano tests with HAC adjustment, Holm correction, and a hierarchical
block bootstrap with 5,000 replications and 168-hour blocks.

## Big Data performance

The Spark/MinIO benchmark reports steady-state laboratory throughput after one
excluded warm-up and three measured repetitions. It is not a horizontal
scalability experiment.

## Package contents

The package includes source scripts, frozen environment information, the final
13-scenario metrics/predictions/histories/weights, multi-seed result artifacts,
statistical-analysis outputs, performance outputs, manifests, and the global
model-input parquet when present.

## Public release

This ZIP is the internal final reproducibility snapshot. Review
`public_release_review.csv` before publishing it. Any hard-coded local credentials,
personal machine paths, or host-specific settings must be sanitized for a public
GitHub/Zenodo release.
"""

    (PACKAGE_DIR / "README_REPRODUCIBILITY_FINAL.md").write_text(
        readme,
        encoding="utf-8",
    )

    # Copia exacta de artefactos.
    project_files = PACKAGE_DIR / "project_files"
    for path in files:
        rel = path.relative_to(PROJECT_ROOT)
        destination = project_files / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    # Copiar también auditoría final original de forma destacada.
    shutil.copy2(
        final13["audit_path"],
        PACKAGE_DIR / "final_13_scenarios_seed42_audit.json",
    )
    shutil.copy2(
        final13["metrics_path"],
        PACKAGE_DIR / "final_13_scenarios_seed42_metrics.csv",
    )

    # Reemplazar ZIP final previo sin tocar el ZIP histórico.
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as zf:
        for path in PACKAGE_DIR.rglob("*"):
            if path.is_file():
                arcname = (
                    Path("Bitcoin_Benchmark_Q1_reproducibility_package_FINAL")
                    / path.relative_to(PACKAGE_DIR)
                ).as_posix()
                zf.write(path, arcname)

    zip_hash = sha256_file(ZIP_PATH)
    ZIP_HASH_PATH.write_text(
        f"{zip_hash}  {ZIP_PATH.name}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("PAQUETE FINAL REFRESCADO")
    print("=" * 108)
    print("Alineación global seed42         : VERIFIED")
    print("Modelos alineados                : 4 / 4")
    print("13 escenarios                    : COMPLETE")
    print("Combinaciones finales            : 65 / 65")
    print("Features                          : 22")
    print("Seed escenarios                   : 42")
    print("CUBLAS                            : :4096:8")
    print(f"Archivos incluidos/auditados     : {len(files)}")
    print(f"Hallazgos para revisión pública  : {len(findings)}")
    print()
    print("ZIP FINAL:")
    print(ZIP_PATH)
    print()
    print("SHA-256 FINAL:")
    print(zip_hash)
    print()
    print("Hash file:")
    print(ZIP_HASH_PATH)
    print()
    print("IMPORTANTE:")
    print(
        "El paquete queda cerrado como snapshot interno reproducible. "
        "Antes de publicarlo, revisa public_release_review.csv."
    )


if __name__ == "__main__":
    main()
