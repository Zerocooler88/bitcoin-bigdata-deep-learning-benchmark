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

COMMON = SRC / "10_common_training_pipeline.py"
STRICT_MARKER = "STRICT_HOURLY_CONTINUITY_V1"

STRICT_AUDIT = (
    OUT
    / "strict_window_migration"
    / "official_strict_audit_manifest.json"
)

SCENARIO_AUDIT = (
    OUT
    / "scenarios_final_seed42"
    / "manifest"
    / "final_13_scenarios_seed42_audit.json"
)

MULTISEED_FINAL = OUT / "multiseed_final_deterministic"

SCRIPTS = {
    "multiseed": SRC / "06_run_multiseed_final_deterministic.py",
    "summary": SRC / "07_resumir_multiseed_final.py",
    "statistics": SRC / "23_estadistica_global_final_vs_persistencia_CORREGIDO.py",
    "efficiency": SRC / "24_perfil_eficiencia_modelos.py",
    "figures": SRC / "30_generar_figuras_cientificas_q1.py",
    "package": SRC / "29_verificar_alineacion_y_refrescar_paquete_final.py",
}

EXPECTED_CUBLAS = ":4096:8"
EXPECTED_STRICT_STATUS = "PASS_STRICT_48H_TO_1H_OFFICIAL"


def run_step(label: str, script: Path) -> None:
    print()
    print("=" * 108)
    print(label)
    print("=" * 108)
    print(f"Script : {script.name}")
    print(f"Python : {sys.executable}")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=True,
    )


def validate_preconditions() -> tuple[dict, dict]:
    if not COMMON.exists():
        raise FileNotFoundError(COMMON)

    if STRICT_MARKER not in COMMON.read_text(encoding="utf-8"):
        raise RuntimeError(
            "El pipeline oficial no contiene STRICT_HOURLY_CONTINUITY_V1."
        )

    if not STRICT_AUDIT.exists():
        raise FileNotFoundError(
            f"Falta auditoría estricta:\n{STRICT_AUDIT}"
        )

    strict = json.loads(
        STRICT_AUDIT.read_text(encoding="utf-8")
    )

    if strict.get("status") != EXPECTED_STRICT_STATUS:
        raise RuntimeError(
            f"Auditoría estricta no aprobada: {strict.get('status')}"
        )

    if int(strict.get("accepted_violations", -1)) != 0:
        raise RuntimeError(
            "La auditoría estricta contiene violaciones aceptadas."
        )

    if int(strict.get("total_windows", -1)) != 179_304:
        raise RuntimeError(
            f"total_windows inesperado: {strict.get('total_windows')}"
        )

    if int(strict.get("total_test_windows", -1)) != 50_356:
        raise RuntimeError(
            f"total_test_windows inesperado: {strict.get('total_test_windows')}"
        )

    if not SCENARIO_AUDIT.exists():
        raise FileNotFoundError(
            f"Falta cierre de los 13 escenarios:\n{SCENARIO_AUDIT}"
        )

    scenario = json.loads(
        SCENARIO_AUDIT.read_text(encoding="utf-8")
    )

    if scenario.get("status") != "COMPLETE":
        raise RuntimeError(
            f"13 escenarios no están COMPLETE: {scenario.get('status')}"
        )

    if int(scenario.get("scenario_count", -1)) != 13:
        raise RuntimeError("scenario_count != 13")

    if int(scenario.get("model_count", -1)) != 5:
        raise RuntimeError("model_count != 5")

    if int(scenario.get("expected_combinations", -1)) != 65:
        raise RuntimeError("expected_combinations != 65")

    if int(scenario.get("canonical_feature_count", -1)) != 22:
        raise RuntimeError("canonical_feature_count != 22")

    if int(scenario.get("seed", -1)) != 42:
        raise RuntimeError("seed final != 42")

    for role, path in SCRIPTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Falta script para {role}:\n{path}"
            )

    print("=" * 108)
    print("PRECONDICIONES PARA REANUDAR: OK")
    print("=" * 108)
    print(f"Filtro estricto             : {strict['status']}")
    print(f"Ventanas estrictas totales : {strict['total_windows']:,}")
    print(f"Ventanas estrictas TEST    : {strict['total_test_windows']:,}")
    print(f"Violaciones aceptadas      : {strict['accepted_violations']}")
    print(f"13 escenarios              : {scenario['status']}")
    print(f"Combinaciones              : {scenario['expected_combinations']}/65")
    print(f"Seed escenario             : {scenario['seed']}")
    print()
    print("Se reanuda DESDE multi-seed; NO se repiten los 13 escenarios.")

    return strict, scenario


def backup_partial_final_if_needed() -> Path | None:
    if not MULTISEED_FINAL.exists():
        return None

    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = OUT / f"multiseed_final_deterministic__PARTIAL_BACKUP_{token}"
    MULTISEED_FINAL.rename(backup)
    print()
    print("Se encontró un multiseed_final_deterministic previo/parcial.")
    print(f"Se preservó en: {backup}")
    return backup


def ensure_outputs_for_figures() -> None:
    required = [
        OUT / "reproducibility" / "multiseed_final_global_resumen.csv",
        OUT / "statistical_analysis" / "global_final_dm_hierarchical_bootstrap.csv",
        OUT / "performance" / "model_computational_efficiency_final.csv",
        OUT
        / "scenarios_final_seed42"
        / "manifest"
        / "final_13_scenarios_seed42_metrics.csv",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos previos a figuras:\n- "
            + "\n- ".join(missing)
        )


def write_resume_manifest(
    strict: dict,
    scenario: dict,
    partial_backup: Path | None,
) -> Path:
    path = (
        OUT
        / "strict_window_migration"
        / "strict_regeneration_complete.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STRICT_REGENERATION_COMPLETE",
        "resume_mode": "FROM_GLOBAL_MULTISEED_AFTER_13_SCENARIOS_COMPLETE",
        "strict_audit_status": strict["status"],
        "strict_total_windows": int(strict["total_windows"]),
        "strict_total_test_windows": int(strict["total_test_windows"]),
        "accepted_violations": int(strict["accepted_violations"]),
        "scenario_status": scenario["status"],
        "scenario_seed": int(scenario["seed"]),
        "scenario_combinations": int(scenario["expected_combinations"]),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "partial_multiseed_backup": (
            str(partial_backup) if partial_backup else None
        ),
        "scripts_used": {
            key: str(value)
            for key, value in SCRIPTS.items()
        },
        "steps_completed": [
            "13_scenarios_seed42_already_completed_before_resume",
            "global_multiseed_final_deterministic_5_seeds",
            "multiseed_final_summary",
            "dm_hac_bootstrap_holm",
            "model_efficiency",
            "publication_figures",
            "alignment_and_final_reproducibility_package",
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    strict, scenario = validate_preconditions()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS

    # IMPORTANT:
    # outputs/multiseed (the older staging archive used by 21_run_multiseed.py)
    # is intentionally ignored. The definitive downstream workflow uses
    # outputs/multiseed_final_deterministic.
    partial_backup = backup_partial_final_if_needed()

    run_step(
        "2/7 REGENERAR GLOBAL MULTI-SEED FINAL DETERMINISTA",
        SCRIPTS["multiseed"],
    )

    run_step(
        "3/7 RESUMIR MULTI-SEED FINAL",
        SCRIPTS["summary"],
    )

    run_step(
        "4/7 RECALCULAR DM + HAC + BOOTSTRAP + HOLM",
        SCRIPTS["statistics"],
    )

    run_step(
        "5/7 RECALCULAR EFICIENCIA DE MODELOS",
        SCRIPTS["efficiency"],
    )

    ensure_outputs_for_figures()

    run_step(
        "6/7 REGENERAR FIGURAS CIENTÍFICAS",
        SCRIPTS["figures"],
    )

    run_step(
        "7/7 VERIFICAR ALINEACIÓN Y REFRESCAR PAQUETE FINAL",
        SCRIPTS["package"],
    )

    manifest = write_resume_manifest(
        strict,
        scenario,
        partial_backup,
    )

    print()
    print("=" * 108)
    print("REGENERACIÓN ESTRICTA REANUDADA Y COMPLETADA")
    print("=" * 108)
    print("13 escenarios seed42       : ya estaban COMPLETE")
    print("Global multi-seed          : regenerado con 06_run_multiseed_final_deterministic.py")
    print("Resumen multi-seed         : regenerado")
    print("DM/HAC/bootstrap/Holm      : regenerado")
    print("Eficiencia                 : regenerada")
    print("Figuras                    : regeneradas")
    print("Paquete reproducibilidad   : refrescado")
    print(f"Manifest final             : {manifest}")
    print()
    print("Ahora sí deben usarse únicamente estos outputs para actualizar el manuscrito.")


if __name__ == "__main__":
    main()
