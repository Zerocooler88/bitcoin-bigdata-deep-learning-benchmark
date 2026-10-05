from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
OUT = PROJECT_ROOT / "outputs"

STRICT_MARKER = "STRICT_HOURLY_CONTINUITY_V1"
COMMON = SRC / "10_common_training_pipeline.py"
AUDIT_MANIFEST = (
    OUT / "strict_window_migration" / "official_strict_audit_manifest.json"
)

EXPECTED_AUDIT_STATUS = "PASS_STRICT_48H_TO_1H_OFFICIAL"
EXPECTED_CUBLAS = ":4096:8"


# -----------------------------------------------------------------------------
# Candidate names observed across the project versions.
# The first existing candidate is selected.
# -----------------------------------------------------------------------------
SCRIPT_CANDIDATES = {
    "scenarios": [
        "28_run_13_escenarios_final_seed42.py",
        "28_run_13_escenarios_final_seed42_CORREGIDO.py",
    ],
    "multiseed": [
        "21_run_multiseed.py",
        "06_run_multiseed_final_deterministic.py",
        "21_run_multiseed_final_deterministic.py",
    ],
    "multiseed_summary": [
        "22_resumir_multiseed_global.py",
        "07_resumir_multiseed_final.py",
        "22_resumir_multiseed_final.py",
    ],
    "statistics": [
        "23_estadistica_global_final_vs_persistencia_CORREGIDO.py",
        "23_estadistica_global_final_vs_persistencia.py",
        "23_estadistica_global_vs_persistencia.py",
    ],
    "efficiency": [
        "24_perfil_eficiencia_modelos.py",
        "24_perfil_eficiencia_modelos_CORREGIDO.py",
    ],
    "package": [
        "29_verificar_alineacion_y_refrescar_paquete_final.py",
        "29_verificar_alineacion_y_refrescar_paquete_final_CORREGIDO.py",
    ],
    "figures": [
        "30_generar_figuras_cientificas_q1.py",
        "30_generar_figuras_cientificas_q1_PYLANCE_CLEAN.py",
        "30_generar_figuras_cientificas_q1_SIN_ERRORES_PYLANCE.py",
        "30_generar_figuras_cientificas_Q1_FINAL_CORREGIDO.py",
        "30_generar_figuras_cientificas_Q1_FINAL.py",
    ],
}


FALLBACK_GLOBS = {
    "scenarios": [
        "28_run_13_escenarios*seed42*.py",
    ],
    "multiseed": [
        "21_run_multiseed*.py",
        "*run_multiseed*deterministic*.py",
    ],
    "multiseed_summary": [
        "22_resumir_multiseed*.py",
        "*resumir_multiseed*.py",
    ],
    "statistics": [
        "23_estadistica_global*persistencia*.py",
        "23_*persistencia*.py",
    ],
    "efficiency": [
        "24_perfil_eficiencia*.py",
    ],
    "package": [
        "29_verificar_alineacion*y_refrescar_paquete*.py",
        "29_verificar_alineacion*.py",
    ],
    "figures": [
        "30_generar_figuras_cientificas*.py",
    ],
}


def resolve_script(role: str) -> Path:
    # 1) Exact preferred names.
    for name in SCRIPT_CANDIDATES[role]:
        p = SRC / name
        if p.exists() and p.is_file():
            return p

    # 2) Fallback glob discovery.
    discovered: list[Path] = []
    for pattern in FALLBACK_GLOBS[role]:
        discovered.extend(
            p for p in SRC.glob(pattern)
            if p.is_file()
            and not p.name.startswith("35_")
            and not p.name.startswith("35b_")
        )

    # Remove duplicates preserving sorted deterministic order.
    unique = sorted(
        {p.resolve(): p for p in discovered}.values(),
        key=lambda p: p.name.lower(),
    )

    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        names = "\n  - ".join(p.name for p in unique)
        raise RuntimeError(
            f"Hay varios candidatos para '{role}' y no es seguro elegir "
            f"automáticamente:\n  - {names}\n"
            "Añade el nombre correcto a SCRIPT_CANDIDATES."
        )

    raise FileNotFoundError(
        f"No se encontró script para '{role}'.\n"
        f"Carpeta examinada: {SRC}\n"
        f"Candidatos esperados: {SCRIPT_CANDIDATES[role]}\n"
        f"Patrones alternativos: {FALLBACK_GLOBS[role]}"
    )


def run(label: str, script: Path) -> None:
    print()
    print("=" * 108)
    print(label)
    print("=" * 108)
    print(f"Script: {script.name}")
    print(f"Comando: {sys.executable} {script}")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=True,
    )


def move_if_exists(path: Path, snapshot: Path) -> None:
    if not path.exists():
        return
    destination = snapshot / path.name
    if destination.exists():
        raise FileExistsError(destination)
    shutil.move(str(path), str(destination))
    print(f"Backup: {path} -> {destination}")


def restore_unaffected_bigdata_files(snapshot: Path) -> None:
    old_perf = snapshot / "performance"
    new_perf = OUT / "performance"
    if not old_perf.exists():
        return

    new_perf.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in old_perf.glob("bigdata_pipeline_lab*"):
        if src.is_file():
            shutil.copy2(src, new_perf / src.name)
            copied += 1

    print(
        f"Archivos de profiling Spark/MinIO restaurados (no afectados): {copied}"
    )


def ensure_statistics_reporting_alias() -> None:
    """
    Algunas versiones del script 23 escriben:
      global_dm_hierarchical_bootstrap.csv
    y otras:
      global_final_dm_hierarchical_bootstrap.csv

    Si existe el primero pero no el segundo, crea una copia de compatibilidad.
    No inventa datos ni modifica el archivo fuente.
    """
    stats_dir = OUT / "statistical_analysis"
    final_stats = stats_dir / "global_final_dm_hierarchical_bootstrap.csv"
    raw_stats = stats_dir / "global_dm_hierarchical_bootstrap.csv"

    if final_stats.exists():
        print(f"Estadística final encontrada: {final_stats.name}")
        return

    if raw_stats.exists():
        shutil.copy2(raw_stats, final_stats)
        print(
            "Alias de compatibilidad creado: "
            f"{raw_stats.name} -> {final_stats.name}"
        )
        return

    # Last-resort diagnostic: show what the script actually produced.
    candidates = sorted(stats_dir.glob("*bootstrap*.csv")) if stats_dir.exists() else []
    found = ", ".join(p.name for p in candidates) or "(ninguno)"
    raise FileNotFoundError(
        "No se encontró el CSV estadístico esperado después de ejecutar "
        f"el script 23. Encontrados: {found}"
    )


def validate_preconditions(scripts: dict[str, Path]) -> dict:
    if not COMMON.exists():
        raise FileNotFoundError(COMMON)

    common_text = COMMON.read_text(encoding="utf-8")
    if STRICT_MARKER not in common_text:
        raise RuntimeError(
            "El pipeline oficial todavía no contiene el filtro estricto."
        )

    if not AUDIT_MANIFEST.exists():
        raise FileNotFoundError(
            "Falta la auditoría oficial estricta. Ejecuta primero "
            "34_verificar_pipeline_estricto_oficial.py"
        )

    audit = json.loads(AUDIT_MANIFEST.read_text(encoding="utf-8"))

    if audit.get("status") != EXPECTED_AUDIT_STATUS:
        raise RuntimeError(
            f"Auditoría estricta no aprobada: {audit.get('status')}"
        )

    if int(audit.get("accepted_violations", -1)) != 0:
        raise RuntimeError(
            "La auditoría registra violaciones aceptadas."
        )

    # The validated strict counts from the safe run.
    if int(audit.get("total_windows", -1)) != 179_304:
        raise RuntimeError(
            f"total_windows inesperado: {audit.get('total_windows')}"
        )

    if int(audit.get("total_test_windows", -1)) != 50_356:
        raise RuntimeError(
            f"total_test_windows inesperado: {audit.get('total_test_windows')}"
        )

    print("=" * 108)
    print("PRECONDICIONES ESTRICTAS: OK")
    print("=" * 108)
    print(f"Auditoría: {audit.get('status')}")
    print(f"Ventanas totales estrictas: {audit.get('total_windows'):,}")
    print(f"Ventanas test estrictas   : {audit.get('total_test_windows'):,}")
    print(f"Violaciones aceptadas     : {audit.get('accepted_violations')}")
    print()
    print("SCRIPTS RESUELTOS:")
    for role, path in scripts.items():
        print(f"  {role:18s} -> {path.name}")

    return audit


def main() -> None:
    # Resolve every dependency BEFORE touching any output folder.
    scripts = {
        role: resolve_script(role)
        for role in [
            "scenarios",
            "multiseed",
            "multiseed_summary",
            "statistics",
            "efficiency",
            "figures",
            "package",
        ]
    }

    audit = validate_preconditions(scripts)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS

    # Only now, after every dependency and precondition is verified, make a snapshot.
    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot = OUT / f"PRE_STRICT_RESULTS__{token}"
    snapshot.mkdir(parents=True, exist_ok=False)

    print()
    print("=" * 108)
    print("BACKUP DE RESULTADOS DEFINITIVOS ANTERIORES")
    print("=" * 108)
    print(snapshot)

    for name in [
        "scenarios_final_seed42",
        "multiseed_final_deterministic",
        "reproducibility",
        "statistical_analysis",
        "performance",
        "figures_q1_final",
        "reproducibility_package_q1_FINAL",
    ]:
        move_if_exists(OUT / name, snapshot)

    for name in [
        "Bitcoin_Benchmark_Q1_reproducibility_package_FINAL.zip",
        "Bitcoin_Benchmark_Q1_reproducibility_package_FINAL.sha256.txt",
    ]:
        move_if_exists(OUT / name, snapshot)

    restore_unaffected_bigdata_files(snapshot)

    # Regeneration sequence.
    run(
        "1/7 REGENERAR 13 ESCENARIOS SEED 42",
        scripts["scenarios"],
    )
    run(
        "2/7 REGENERAR GLOBAL MULTI-SEED",
        scripts["multiseed"],
    )
    run(
        "3/7 RESUMIR MULTI-SEED",
        scripts["multiseed_summary"],
    )
    run(
        "4/7 RECALCULAR DM + HAC + BOOTSTRAP + HOLM",
        scripts["statistics"],
    )

    ensure_statistics_reporting_alias()

    run(
        "5/7 RECALCULAR EFICIENCIA DE MODELOS",
        scripts["efficiency"],
    )
    run(
        "6/7 REGENERAR FIGURAS CIENTÍFICAS",
        scripts["figures"],
    )
    run(
        "7/7 VERIFICAR ALINEACIÓN Y REFRESCAR PAQUETE",
        scripts["package"],
    )

    final_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STRICT_REGENERATION_COMPLETE",
        "strict_audit_status": audit["status"],
        "strict_total_windows": int(audit["total_windows"]),
        "strict_total_test_windows": int(audit["total_test_windows"]),
        "accepted_violations": int(audit["accepted_violations"]),
        "previous_results_snapshot": str(snapshot),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "scripts_used": {
            role: str(path)
            for role, path in scripts.items()
        },
        "steps_completed": [
            "13_scenarios_seed42",
            "global_multiseed_5_seeds",
            "multiseed_summary",
            "dm_hac_bootstrap_holm",
            "model_efficiency",
            "publication_figures",
            "final_alignment_and_reproducibility_package",
        ],
    }

    final_path = (
        OUT
        / "strict_window_migration"
        / "strict_regeneration_complete.json"
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(
        json.dumps(
            final_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("REGENERACIÓN ESTRICTA COMPLETADA")
    print("=" * 108)
    print(f"Snapshot anterior : {snapshot}")
    print(f"Manifest final    : {final_path}")
    print()
    print(
        "No actualices el manuscrito con cifras antiguas. "
        "Usa únicamente los outputs nuevos de esta corrida."
    )


if __name__ == "__main__":
    main()
