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

SCRIPTS = {
    "scenarios": SRC / "28_run_13_escenarios_final_seed42.py",
    "multiseed": SRC / "06_run_multiseed_final_deterministic.py",
    "multiseed_summary": SRC / "07_resumir_multiseed_final.py",
    "statistics": SRC / "23_estadistica_global_vs_persistencia.py",
    "efficiency": SRC / "24_perfil_eficiencia_modelos.py",
    "package": SRC / "29_verificar_alineacion_y_refrescar_paquete_final.py",
}

FIGURE_CANDIDATES = [
    SRC / "30_generar_figuras_cientificas_q1_PYLANCE_CLEAN.py",
    SRC / "30_generar_figuras_cientificas_q1_SIN_ERRORES_PYLANCE.py",
    SRC / "30_generar_figuras_cientificas_Q1_FINAL_CORREGIDO.py",
    SRC / "30_generar_figuras_cientificas_Q1_FINAL.py",
]


def run(label: str, script: Path) -> None:
    print()
    print("=" * 108)
    print(label)
    print("=" * 108)
    print(f"{sys.executable} {script}")
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

    for src in old_perf.glob("bigdata_pipeline_lab*"):
        if src.is_file():
            shutil.copy2(src, new_perf / src.name)


def main() -> None:
    if not COMMON.exists():
        raise FileNotFoundError(COMMON)
    if STRICT_MARKER not in COMMON.read_text(encoding="utf-8"):
        raise RuntimeError(
            "El pipeline oficial todavía no contiene el filtro estricto."
        )

    if not AUDIT_MANIFEST.exists():
        raise FileNotFoundError(
            "Falta la auditoría oficial estricta. "
            "Ejecuta primero 34_verificar_pipeline_estricto_oficial.py"
        )

    audit = json.loads(AUDIT_MANIFEST.read_text(encoding="utf-8"))
    if audit.get("status") != EXPECTED_AUDIT_STATUS:
        raise RuntimeError(
            f"Auditoría estricta no aprobada: {audit.get('status')}"
        )
    if int(audit.get("accepted_violations", -1)) != 0:
        raise RuntimeError("La auditoría registra violaciones aceptadas.")

    for label, path in SCRIPTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Falta script requerido ({label}): {path}"
            )

    figure_script = next(
        (p for p in FIGURE_CANDIDATES if p.exists()),
        None,
    )
    if figure_script is None:
        raise FileNotFoundError(
            "No se encontró ningún script final de figuras 10–15."
        )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS

    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot = OUT / f"PRE_STRICT_RESULTS__{token}"
    snapshot.mkdir(parents=True, exist_ok=False)

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

    run(
        "1/7 REGENERAR 13 ESCENARIOS SEED 42",
        SCRIPTS["scenarios"],
    )
    run(
        "2/7 REGENERAR GLOBAL MULTI-SEED",
        SCRIPTS["multiseed"],
    )
    run(
        "3/7 RESUMIR MULTI-SEED",
        SCRIPTS["multiseed_summary"],
    )
    run(
        "4/7 RECALCULAR DM + HAC + BOOTSTRAP + HOLM",
        SCRIPTS["statistics"],
    )

    stats_dir = OUT / "statistical_analysis"
    raw_stats = stats_dir / "global_dm_hierarchical_bootstrap.csv"
    final_stats = stats_dir / "global_final_dm_hierarchical_bootstrap.csv"
    if not raw_stats.exists():
        raise FileNotFoundError(raw_stats)
    shutil.copy2(raw_stats, final_stats)

    run(
        "5/7 RECALCULAR EFICIENCIA DE MODELOS",
        SCRIPTS["efficiency"],
    )
    run(
        "6/7 REGENERAR FIGURAS CIENTÍFICAS",
        figure_script,
    )
    run(
        "7/7 VERIFICAR ALINEACIÓN Y REFRESCAR PAQUETE",
        SCRIPTS["package"],
    )

    final_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STRICT_REGENERATION_COMPLETE",
        "strict_audit_status": audit["status"],
        "strict_total_windows": audit["total_windows"],
        "strict_total_test_windows": audit["total_test_windows"],
        "accepted_violations": audit["accepted_violations"],
        "previous_results_snapshot": str(snapshot),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
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
        json.dumps(final_manifest, indent=2, ensure_ascii=False),
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
        "Usa únicamente los nuevos outputs de esta corrida."
    )


if __name__ == "__main__":
    main()
