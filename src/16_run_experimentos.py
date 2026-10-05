from __future__ import (
    annotations,
)  # Permite referencias de tipos pospuestas para una anotación más limpia.

import argparse  # Procesa argumentos de consola para elegir modelos, escenarios y parámetros del orquestador.
import importlib.util  # Carga dinámicamente el archivo 10_common_training_pipeline.py.
import json  # Exporta el resumen consolidado del orquestador en formato JSON.
import subprocess  # Ejecuta los scripts hijos de forma robusta y secuencial.
import sys  # Usa el mismo intérprete Python activo para lanzar los subprocess.
import time  # Mide duración real de cada ejecución de modelo.
from dataclasses import (
    asdict,
    dataclass,
    field,
)  # Define estructuras limpias y serializables para configuración y resultados.
from datetime import (
    datetime,
    timezone,
)  # Genera marcas temporales UTC para trazabilidad de la corrida completa.
from pathlib import (
    Path,
)  # Maneja rutas de archivos y carpetas de forma robusta en Windows.
from typing import (
    Any,
    Dict,
    List,
)  # Declara tipos para mejorar claridad y mantenimiento del código.

import pandas as pd  # Construye el resumen consolidado en CSV para Tableau.


CURRENT_DIR = (
    Path(__file__).resolve().parent
)  # Resuelve la carpeta src donde vive este script.
PROJECT_ROOT = (
    CURRENT_DIR.parent
)  # Resuelve la raíz del proyecto para usar rutas coherentes en Windows.
COMMON_PIPELINE_PATH = (
    CURRENT_DIR / "10_common_training_pipeline.py"
)  # Apunta al pipeline común con la numeración nueva.
COMMON_MODULE_NAME = "common_training_pipeline_v10_for_orchestrator"  # Define un nombre interno estable para registrar el módulo cargado.
ORCHESTRATOR_NAME = "run_experimentos"  # Define el nombre lógico del orquestador para consola y exportaciones.

DEFAULT_MODEL_ORDER = [
    "baseline_persistencia",
    "lstm",
    "gru",
    "cnn1d",
    "cnn_lstm",
]  # Define el orden oficial de ejecución cuando el usuario pide --models all.

MODEL_TO_SCRIPT_FILENAME = {
    "baseline_persistencia": "11_baseline_persistencia.py",
    "lstm": "12_entrenar_lstm.py",
    "gru": "13_entrenar_gru.py",
    "cnn1d": "14_entrenar_cnn1d.py",
    "cnn_lstm": "15_entrenar_cnn_lstm.py",
}  # Mapea cada modelo válido a su script oficial del proyecto.

MODEL_TO_DISPLAY_NAME = {
    "baseline_persistencia": "BASELINE_PERSISTENCIA",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn1d": "CNN1D",
    "cnn_lstm": "CNN_LSTM",
}  # Define nombres visibles y consistentes para consola.


@dataclass
class OrchestratorConfig:
    project_root: Path = PROJECT_ROOT  # Conserva la raíz del proyecto.
    lookback: int = 48  # Mantiene el lookback por defecto solicitado.
    horizon: int = 1  # Mantiene el horizon por defecto solicitado.
    batch_size: int = 64  # Mantiene el batch_size por defecto solicitado.
    epochs: int = 50  # Mantiene el número máximo de épocas por defecto.
    patience: int = 5  # Mantiene la paciencia por defecto.
    learning_rate: float = 0.001  # Mantiene la learning rate por defecto.
    num_workers: int = 0  # Mantiene compatibilidad segura con Windows.
    seed: int = 42  # Mantiene la semilla por defecto.
    target_mode: str = "delta"  # Mantiene el modo de target por defecto.
    models_requested: List[str] = field(
        default_factory=list
    )  # Conserva las claves de modelos solicitadas por el usuario.
    scenarios_requested: List[str] = field(
        default_factory=list
    )  # Conserva las claves de escenarios solicitadas por el usuario.


@dataclass
class ModelExecutionResult:
    modelo: str  # Conserva la clave oficial del modelo ejecutado.
    script_filename: str  # Conserva el nombre del script llamado.
    script_path: str  # Conserva la ruta absoluta del script ejecutado.
    estado: str  # Conserva el estado final del modelo: ok o fallido.
    codigo_salida: int  # Conserva el return code final del subprocess.
    comando_ejecutado: str  # Conserva el comando exacto lanzado.
    inicio_utc: str  # Conserva el timestamp de inicio en UTC.
    fin_utc: str  # Conserva el timestamp de fin en UTC.
    duracion_segundos: float  # Conserva la duración total de la ejecución.
    escenarios_solicitados: List[
        str
    ]  # Conserva los escenarios pedidos para ese modelo.
    parametros_usados: Dict[
        str, Any
    ]  # Conserva los parámetros numéricos y de modo usados.
    error_message: str = ""  # Conserva un mensaje resumido si la ejecución falló.


def load_common_pipeline_module() -> Any:
    if not COMMON_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "No existe el archivo requerido '10_common_training_pipeline.py' en la carpeta src.\n"
            f"Ruta esperada: {COMMON_PIPELINE_PATH}"
        )  # Informa claramente cuándo falta el pipeline común requerido.

    spec = importlib.util.spec_from_file_location(
        COMMON_MODULE_NAME, COMMON_PIPELINE_PATH
    )  # Construye la especificación del módulo a partir de su ruta física.
    if spec is None or spec.loader is None:
        raise ImportError(
            "No fue posible crear la especificación de importación para "
            f"'{COMMON_PIPELINE_PATH.name}'."
        )  # Informa un fallo explícito si Python no logra preparar la carga dinámica.

    module = importlib.util.module_from_spec(
        spec
    )  # Crea el objeto módulo todavía sin ejecutar su contenido.
    sys.modules[COMMON_MODULE_NAME] = (
        module  # Registra el módulo para que quede disponible dentro del intérprete actual.
    )
    spec.loader.exec_module(
        module
    )  # Ejecuta el archivo Python y deja expuestas sus funciones, clases y constantes.
    return module  # Devuelve el pipeline común ya cargado dinámicamente.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orquesta la ejecución secuencial de baseline_persistencia, lstm, gru, "
            "cnn1d y cnn_lstm reutilizando los scripts oficiales del proyecto."
        )
    )  # Define una ayuda clara para consola y VS Code.

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Modelos a ejecutar. Usa baseline_persistencia, lstm, gru, cnn1d, "
            "cnn_lstm o 'all'. También puedes separarlos por comas."
        ),
    )  # Permite ejecutar todos los modelos o solo un subconjunto.

    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["global"],
        help=(
            "Escenarios a ejecutar. Usa claves como global, pre_covid, covid, post_covid, "
            "trimestre_mas_bajo, trimestre_mas_alto, intra_2024_Q1, intra_2024_Q3, "
            "intra_2025_Q1, cross_2024_Q1_to_2024_Q3, cross_2024_Q1_to_2025_Q1, "
            "cross_pre_to_post, cross_bajo_to_alto, o 'all'. "
            "También puedes separarlos por comas."
        ),
    )  # Mantiene la misma interfaz de escenarios del resto del proyecto.

    parser.add_argument(
        "--lookback",
        type=int,
        default=48,
        help="Cantidad de pasos históricos usados por cada ventana temporal.",
    )  # Conserva el lookback por defecto solicitado.
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Cantidad de pasos futuros a predecir.",
    )  # Conserva el horizon por defecto solicitado.
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Tamaño de lote usado durante entrenamiento y evaluación.",
    )  # Conserva el batch_size por defecto solicitado.
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número máximo de épocas de entrenamiento.",
    )  # Conserva epochs por defecto solicitado.
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciencia de early stopping.",
    )  # Conserva patience por defecto solicitado.
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje del optimizador Adam.",
    )  # Conserva learning_rate por defecto solicitado.
    parser.add_argument(
        "--target-mode",
        dest="target_mode",
        type=str,
        default="delta",
        help="Modo de target a usar en los modelos entrenables: 'price' o 'delta'.",
    )  # Mantiene compatibilidad total con el resto del proyecto.
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=0,
        help="Número de workers del DataLoader para compatibilidad con Windows.",
    )  # Respeta la configuración segura para Windows y VS Code.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )  # Mantiene la reproducibilidad del experimento.

    return (
        parser.parse_args()
    )  # Devuelve todos los argumentos ya parseados desde consola.


def validate_positive_integer(value: int, label: str) -> None:
    if value <= 0:
        raise ValueError(
            f"El parámetro '{label}' debe ser mayor que cero. Valor recibido: {value}"
        )  # Evita configuraciones inválidas antes de lanzar procesos.


def validate_non_negative_integer(value: int, label: str) -> None:
    if value < 0:
        raise ValueError(
            f"El parámetro '{label}' no puede ser negativo. Valor recibido: {value}"
        )  # Evita configuraciones inválidas para parámetros como num_workers.


def validate_positive_float(value: float, label: str) -> None:
    if value <= 0:
        raise ValueError(
            f"El parámetro '{label}' debe ser mayor que cero. Valor recibido: {value}"
        )  # Evita configuraciones numéricas inválidas antes de lanzar procesos.


def normalize_model_keys(raw_values: List[str]) -> List[str]:
    normalized: List[str] = []  # Acumula modelos ya limpiados y normalizados.
    seen = (
        set()
    )  # Evita duplicados cuando el usuario repite modelos o mezcla comas y espacios.

    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            clean_token = token.strip()
            if not clean_token:
                continue  # Ignora fragmentos vacíos producidos por comas consecutivas o espacios.
            lowered = (
                clean_token.lower()
            )  # Normaliza a minúsculas para una resolución robusta.
            if lowered == "all":
                return [
                    "all"
                ]  # Si aparece all, prioriza la ejecución del catálogo completo.
            if lowered not in seen:
                normalized.append(lowered)  # Conserva el orden de llegada del usuario.
                seen.add(lowered)  # Marca el modelo como ya agregado.

    if not normalized:
        raise ValueError(
            "No se detectaron modelos válidos en '--models'."
        )  # Informa de forma explícita cuándo la entrada quedó vacía tras limpiar.

    return normalized  # Devuelve la lista final de modelos normalizados.


def normalize_scenario_keys(raw_values: List[str]) -> List[str]:
    normalized: List[str] = []  # Acumula escenarios ya limpiados y normalizados.
    seen = (
        set()
    )  # Evita duplicados cuando el usuario repite escenarios o mezcla comas y espacios.

    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            clean_token = token.strip()
            if not clean_token:
                continue  # Ignora fragmentos vacíos producidos por comas consecutivas o espacios.
            lowered = (
                clean_token.lower()
            )  # Normaliza a minúsculas para una resolución robusta.
            if lowered == "all":
                return [
                    "all"
                ]  # Si aparece all, prioriza la ejecución del catálogo completo.
            if lowered not in seen:
                normalized.append(lowered)  # Conserva el orden de llegada del usuario.
                seen.add(lowered)  # Marca el escenario como ya agregado.

    if not normalized:
        raise ValueError(
            "No se detectaron escenarios válidos en '--scenarios'."
        )  # Informa de forma explícita cuándo la entrada quedó vacía tras limpiar.

    return normalized  # Devuelve la lista final de escenarios normalizados.


def resolve_models(requested_keys: List[str]) -> List[str]:
    if requested_keys == ["all"]:
        return (
            DEFAULT_MODEL_ORDER.copy()
        )  # Devuelve todos los modelos en el orden oficial.

    invalid_models = [
        key for key in requested_keys if key not in MODEL_TO_SCRIPT_FILENAME
    ]  # Detecta modelos inexistentes o mal escritos.
    if invalid_models:
        available = ", ".join(
            DEFAULT_MODEL_ORDER
        )  # Construye el listado visible de modelos válidos.
        invalid_text = ", ".join(
            invalid_models
        )  # Construye el listado visible de modelos inválidos.
        raise KeyError(
            f"Modelos no reconocidos: {invalid_text}. Modelos válidos: {available}"
        )  # Informa de forma clara los modelos incorrectos.

    return requested_keys  # Devuelve los modelos solicitados y ya validados.


def resolve_scenarios(common: Any, requested_keys: List[str]) -> List[Any]:
    catalog = common.build_default_scenarios(
        project_root=common.PROJECT_ROOT
    )  # Construye el catálogo oficial de escenarios desde el pipeline común.

    if requested_keys == ["all"]:
        return list(
            catalog.values()
        )  # Devuelve todos los escenarios conservando el orden oficial del catálogo.

    return common.get_scenarios_from_keys(
        requested_keys, project_root=common.PROJECT_ROOT
    )  # Reutiliza la resolución oficial de escenarios del pipeline común.


def validate_orchestrator_args(common: Any, args: argparse.Namespace) -> None:
    validate_positive_integer(
        args.lookback, "lookback"
    )  # Valida lookback antes de lanzar procesos.
    validate_positive_integer(
        args.horizon, "horizon"
    )  # Valida horizon antes de lanzar procesos.
    validate_positive_integer(
        args.batch_size, "batch_size"
    )  # Valida batch_size antes de lanzar procesos.
    validate_positive_integer(args.epochs, "epochs")  # Valida epochs antes de ejecutar.
    validate_positive_integer(
        args.patience, "patience"
    )  # Valida patience antes de ejecutar.
    validate_positive_float(
        args.learning_rate, "learning_rate"
    )  # Valida learning_rate antes de ejecutar.
    validate_non_negative_integer(
        args.num_workers, "num_workers"
    )  # Valida num_workers para Windows.
    common.validate_target_mode(
        args.target_mode
    )  # Valida el modo de target con la lógica oficial del proyecto.


def build_orchestrator_config(
    common: Any,
    args: argparse.Namespace,
    models_requested: List[str],
    scenarios_requested: List[str],
) -> OrchestratorConfig:
    target_mode = common.validate_target_mode(
        args.target_mode
    )  # Valida y normaliza una sola vez el modo de target.

    return OrchestratorConfig(
        project_root=common.PROJECT_ROOT,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        target_mode=target_mode,
        models_requested=models_requested,
        scenarios_requested=scenarios_requested,
    )  # Construye una configuración limpia y serializable para todo el orquestador.


def resolve_script_path(model_key: str) -> Path:
    filename = MODEL_TO_SCRIPT_FILENAME[
        model_key
    ]  # Recupera el nombre oficial del script asociado al modelo.
    return (
        CURRENT_DIR / filename
    )  # Devuelve la ruta absoluta del script correspondiente.


def ensure_required_scripts_exist(models_to_run: List[str]) -> None:
    missing_paths: List[str] = (
        []
    )  # Acumula scripts faltantes antes de iniciar la corrida.

    for model_key in models_to_run:
        script_path = resolve_script_path(
            model_key
        )  # Resuelve la ruta oficial del script del modelo actual.
        if not script_path.exists():
            missing_paths.append(
                str(script_path)
            )  # Registra scripts faltantes para informar en bloque.

    if missing_paths:
        missing_text = "\n".join(
            missing_paths
        )  # Construye una salida legible para Windows y VS Code.
        raise FileNotFoundError(
            "Faltan uno o más scripts requeridos para ejecutar el orquestador:\n"
            f"{missing_text}"
        )  # Detiene la ejecución antes de lanzar subprocess si falta algún archivo clave.


def build_command(script_path: Path, config: OrchestratorConfig) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--scenarios",
        *config.scenarios_requested,
        "--lookback",
        str(config.lookback),
        "--horizon",
        str(config.horizon),
        "--batch-size",
        str(config.batch_size),
        "--epochs",
        str(config.epochs),
        "--patience",
        str(config.patience),
        "--learning-rate",
        str(config.learning_rate),
        "--target-mode",
        config.target_mode,
        "--num-workers",
        str(config.num_workers),
        "--seed",
        str(config.seed),
    ]  # Construye el comando completo del modelo actual usando el mismo intérprete Python activo.

    return command  # Devuelve el comando listo para pasarlo a subprocess.


def format_command_for_display(command: List[str]) -> str:
    return subprocess.list2cmdline(
        command
    )  # Convierte la lista del comando a un string legible y amigable para Windows.


def print_header(
    config: OrchestratorConfig,
    models_to_run: List[str],
    resolved_scenarios: List[Any],
) -> None:
    model_names = ", ".join(
        models_to_run
    )  # Construye la lista visible de modelos en el orden real de ejecución.
    scenario_names = ", ".join(
        scenario.display_name for scenario in resolved_scenarios
    )  # Construye la lista visible de escenarios ya resueltos.
    python_path = (
        sys.executable
    )  # Recupera el intérprete actual para trazabilidad de entorno.

    print("ORQUESTADOR MAESTRO DE EXPERIMENTOS")  # Imprime el encabezado principal.
    print(
        f"experimento        = {ORCHESTRATOR_NAME}"
    )  # Imprime el nombre lógico del orquestador.
    print(
        f"python             = {python_path}"
    )  # Imprime el Python activo del entorno.
    print(
        f"project_root       = {config.project_root}"
    )  # Imprime la raíz del proyecto para confirmar contexto.
    print(
        f"lookback           = {config.lookback}"
    )  # Imprime el lookback solicitado por consola.
    print(
        f"horizon            = {config.horizon}"
    )  # Imprime el horizon solicitado por consola.
    print(
        f"batch_size         = {config.batch_size}"
    )  # Imprime el batch_size solicitado por consola.
    print(f"epochs             = {config.epochs}")  # Imprime epochs del experimento.
    print(
        f"patience           = {config.patience}"
    )  # Imprime patience del experimento.
    print(
        f"learning_rate      = {config.learning_rate}"
    )  # Imprime la learning rate del experimento.
    print(
        f"target_mode        = {config.target_mode}"
    )  # Imprime el target_mode validado.
    print(
        f"num_workers        = {config.num_workers}"
    )  # Imprime num_workers para trazabilidad en Windows.
    print(f"seed               = {config.seed}")  # Imprime la semilla aleatoria.
    print(
        f"modelos            = {model_names}"
    )  # Imprime el orden real de modelos a ejecutar.
    print(
        f"escenarios         = {scenario_names}"
    )  # Imprime la lista visible de escenarios resueltos.
    print()  # Inserta una línea en blanco para mejorar legibilidad.


def run_model_process(
    model_key: str,
    script_path: Path,
    command: List[str],
    config: OrchestratorConfig,
    index: int,
    total_models: int,
) -> ModelExecutionResult:
    display_name = MODEL_TO_DISPLAY_NAME.get(
        model_key, model_key
    )  # Recupera el nombre visible del modelo actual.
    command_display = format_command_for_display(
        command
    )  # Convierte el comando a texto legible para consola y resumen.
    start_dt = datetime.now(timezone.utc)  # Registra la hora exacta de inicio en UTC.
    start_counter = (
        time.perf_counter()
    )  # Inicia el contador de alta precisión para medir duración.
    status = "fallido"  # Inicializa el estado como fallido hasta demostrar éxito.
    return_code = (
        1  # Inicializa el código de salida con un valor no exitoso por seguridad.
    )
    error_message = ""  # Inicializa el detalle de error vacío.

    print("=" * 100)  # Separa visualmente cada modelo del siguiente.
    print(
        f"[{index}/{total_models}] EJECUTANDO MODELO: {display_name}"
    )  # Informa el progreso general del orquestador.
    print(
        f"script             = {script_path.name}"
    )  # Imprime el archivo que se va a lanzar.
    print(f"ruta_script        = {script_path}")  # Imprime la ruta absoluta del script.
    print(
        f"inicio_utc         = {start_dt.isoformat()}"
    )  # Imprime el timestamp exacto de inicio.
    print(
        f"comando            = {command_display}"
    )  # Imprime el comando exacto que se está lanzando.
    print("=" * 100)  # Cierra el bloque de cabecera del modelo actual.

    try:
        completed = subprocess.run(
            command,
            cwd=str(config.project_root),
            check=False,
        )  # Ejecuta el script hijo usando el mismo entorno Python y la raíz del proyecto.

        return_code = int(
            completed.returncode
        )  # Recupera el código de salida final del subprocess.
        if return_code == 0:
            status = (
                "ok"  # Marca la ejecución como exitosa si el código de salida es cero.
            )
        else:
            status = "fallido"  # Marca la ejecución como fallida si el código de salida no es cero.
            error_message = (
                "El subprocess terminó con un código de salida distinto de cero. "
                "Revisa la salida mostrada arriba para ubicar el error real del modelo."
            )  # Resume de forma pedagógica cómo interpretar un fallo del proceso hijo.

    except FileNotFoundError as exc:
        status = (
            "fallido"  # Marca la ejecución como fallida si no pudo lanzarse el proceso.
        )
        return_code = 9001  # Usa un código interno claro para fallo de lanzamiento.
        error_message = f"No fue posible lanzar el subprocess del modelo '{model_key}'. Detalle: {exc}"  # Informa claramente un problema de archivo o ejecutable faltante.
    except Exception as exc:
        status = "fallido"  # Marca la ejecución como fallida si hubo una excepción inesperada.
        return_code = (
            9002  # Usa un código interno claro para error inesperado de lanzamiento.
        )
        error_message = f"Ocurrió un error inesperado al ejecutar el modelo '{model_key}'. Detalle: {exc}"  # Informa de forma clara una excepción fuera del return code del subprocess.

    end_dt = datetime.now(timezone.utc)  # Registra la hora exacta de fin en UTC.
    duration_seconds = float(
        time.perf_counter() - start_counter
    )  # Calcula la duración real de la ejecución actual.

    print()  # Inserta una línea en blanco tras la salida del modelo.
    print(
        f"resultado_modelo   = {model_key}"
    )  # Imprime la clave oficial del modelo ejecutado.
    print(f"estado             = {status}")  # Imprime el estado final del modelo.
    print(
        f"codigo_salida      = {return_code}"
    )  # Imprime el return code final del subprocess.
    print(f"fin_utc            = {end_dt.isoformat()}")  # Imprime el timestamp final.
    print(
        f"duracion_segundos  = {duration_seconds:.2f}"
    )  # Imprime la duración total de la ejecución.
    if error_message:
        print(
            f"detalle            = {error_message}"
        )  # Imprime un resumen entendible si hubo algún fallo.
    print()  # Inserta una línea en blanco para separar este modelo del siguiente.

    return ModelExecutionResult(
        modelo=model_key,
        script_filename=script_path.name,
        script_path=str(script_path),
        estado=status,
        codigo_salida=return_code,
        comando_ejecutado=command_display,
        inicio_utc=start_dt.isoformat(),
        fin_utc=end_dt.isoformat(),
        duracion_segundos=duration_seconds,
        escenarios_solicitados=config.scenarios_requested.copy(),
        parametros_usados={
            "lookback": config.lookback,
            "horizon": config.horizon,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "patience": config.patience,
            "learning_rate": config.learning_rate,
            "target_mode": config.target_mode,
            "num_workers": config.num_workers,
            "seed": config.seed,
        },
        error_message=error_message,
    )  # Devuelve el registro completo del modelo ejecutado para el resumen consolidado.


def build_consolidated_payload(
    config: OrchestratorConfig,
    resolved_scenarios: List[Any],
    results: List[ModelExecutionResult],
) -> Dict[str, Any]:
    total_models = len(results)  # Cuenta cuántos modelos fueron ejecutados realmente.
    ok_models = sum(
        1 for result in results if result.estado == "ok"
    )  # Cuenta cuántos modelos terminaron correctamente.
    failed_models = (
        total_models - ok_models
    )  # Calcula cuántos modelos terminaron con fallo.

    config_dict = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }  # Convierte la dataclass de configuración a un diccionario serializable.

    return {
        "orquestador": ORCHESTRATOR_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "project_root": str(config.project_root),
        "escenarios_resueltos": [
            scenario.display_name for scenario in resolved_scenarios
        ],
        "config": config_dict,
        "resumen": {
            "total_modelos": total_models,
            "modelos_ok": ok_models,
            "modelos_fallidos": failed_models,
            "estado_global": "ok" if failed_models == 0 else "parcial_con_errores",
        },
        "resultados": [asdict(result) for result in results],
    }  # Construye el JSON consolidado final con configuración, resumen global y resultados por modelo.


def build_consolidated_csv_rows(
    config: OrchestratorConfig,
    resolved_scenarios: List[Any],
    results: List[ModelExecutionResult],
) -> List[Dict[str, Any]]:
    resolved_scenarios_display = ", ".join(
        scenario.display_name for scenario in resolved_scenarios
    )  # Convierte los escenarios resueltos a un texto legible y compatible con CSV.

    rows: List[Dict[str, Any]] = []  # Acumula una fila plana por cada modelo ejecutado.

    for result in results:
        rows.append(
            {
                "modelo": result.modelo,
                "estado": result.estado,
                "codigo_salida": result.codigo_salida,
                "script_filename": result.script_filename,
                "script_path": result.script_path,
                "comando_ejecutado": result.comando_ejecutado,
                "inicio_utc": result.inicio_utc,
                "fin_utc": result.fin_utc,
                "duracion_segundos": result.duracion_segundos,
                "escenarios_solicitados": ", ".join(result.escenarios_solicitados),
                "escenarios_resueltos": resolved_scenarios_display,
                "lookback": result.parametros_usados.get("lookback"),
                "horizon": result.parametros_usados.get("horizon"),
                "batch_size": result.parametros_usados.get("batch_size"),
                "epochs": result.parametros_usados.get("epochs"),
                "patience": result.parametros_usados.get("patience"),
                "learning_rate": result.parametros_usados.get("learning_rate"),
                "target_mode": result.parametros_usados.get("target_mode"),
                "num_workers": result.parametros_usados.get("num_workers"),
                "seed": result.parametros_usados.get("seed"),
                "error_message": result.error_message,
            }
        )  # Construye una fila plana, clara y consistente para Tableau y auditoría.

    return rows  # Devuelve la tabla lista para convertirse en DataFrame.


def export_consolidated_summary(
    common: Any,
    config: OrchestratorConfig,
    resolved_scenarios: List[Any],
    results: List[ModelExecutionResult],
) -> Dict[str, str]:
    metrics_dir = common.ensure_directory(
        config.project_root / "outputs" / "metrics"
    )  # Asegura la carpeta de métricas usando la misma lógica del pipeline común.

    timestamp_token = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )  # Genera un sufijo temporal para no sobreescribir corridas previas.

    json_path = (
        metrics_dir
        / f"{ORCHESTRATOR_NAME}__resumen_consolidado__{timestamp_token}.json"
    )  # Define la ruta del JSON consolidado final.
    csv_path = (
        metrics_dir / f"{ORCHESTRATOR_NAME}__resumen_consolidado__{timestamp_token}.csv"
    )  # Define la ruta del CSV consolidado final.

    payload = build_consolidated_payload(
        config=config,
        resolved_scenarios=resolved_scenarios,
        results=results,
    )  # Construye el documento JSON completo del resumen consolidado.

    csv_rows = build_consolidated_csv_rows(
        config=config,
        resolved_scenarios=resolved_scenarios,
        results=results,
    )  # Aplana el resumen para una exportación CSV compatible con Tableau.

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )  # Exporta el resumen consolidado legible en formato JSON.

    pd.DataFrame(csv_rows).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )  # Exporta el resumen consolidado plano en CSV con BOM para Windows, Excel y Tableau.

    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }  # Devuelve las rutas exportadas del resumen consolidado final.


def print_consolidated_summary(
    results: List[ModelExecutionResult],
    summary_paths: Dict[str, str],
) -> None:
    print(
        "RESUMEN CONSOLIDADO FINAL"
    )  # Imprime el encabezado del cierre del orquestador.

    if not results:
        print(
            "No se generaron resultados para resumir."
        )  # Informa si no hubo modelos ejecutados.
        return

    total_models = len(results)  # Cuenta cuántos modelos se intentaron ejecutar.
    ok_models = sum(
        1 for result in results if result.estado == "ok"
    )  # Cuenta cuántos modelos terminaron bien.
    failed_models = (
        total_models - ok_models
    )  # Calcula cuántos modelos terminaron con fallo.

    for result in results:
        print(
            f"- {result.modelo} | "
            f"estado={result.estado} | "
            f"codigo_salida={result.codigo_salida} | "
            f"duracion_segundos={result.duracion_segundos:.2f}"
        )  # Imprime una línea compacta por modelo para lectura rápida en consola.

    print()  # Inserta una línea en blanco antes del bloque de cierre.
    print(
        f"total_modelos      = {total_models}"
    )  # Informa el total de modelos ejecutados.
    print(
        f"modelos_ok         = {ok_models}"
    )  # Informa cuántos modelos terminaron bien.
    print(
        f"modelos_fallidos   = {failed_models}"
    )  # Informa cuántos modelos terminaron con error.
    print(
        f"resumen_json       = {summary_paths['json']}"
    )  # Informa la ruta final del JSON consolidado.
    print(
        f"resumen_csv        = {summary_paths['csv']}"
    )  # Informa la ruta final del CSV consolidado.


def main() -> None:
    common = (
        load_common_pipeline_module()
    )  # Carga dinámicamente el pipeline común numerado como 10.
    args = parse_args()  # Lee la configuración de ejecución enviada por consola.
    validate_orchestrator_args(
        common, args
    )  # Valida primero todos los parámetros numéricos y de target.
    requested_models = normalize_model_keys(
        args.models
    )  # Limpia y normaliza las claves de modelos pedidas por el usuario.
    requested_scenarios = normalize_scenario_keys(
        args.scenarios
    )  # Limpia y normaliza las claves de escenarios pedidas por el usuario.
    models_to_run = resolve_models(
        requested_models
    )  # Resuelve la lista final de modelos usando el orden oficial cuando corresponde.
    resolved_scenarios = resolve_scenarios(
        common, requested_scenarios
    )  # Resuelve la lista final de escenarios usando el catálogo oficial.
    config = build_orchestrator_config(
        common=common,
        args=args,
        models_requested=models_to_run,
        scenarios_requested=requested_scenarios,
    )  # Construye la configuración limpia del orquestador maestro.
    ensure_required_scripts_exist(
        models_to_run
    )  # Valida que todos los scripts requeridos existan antes de lanzar procesos.

    print_header(
        config=config,
        models_to_run=models_to_run,
        resolved_scenarios=resolved_scenarios,
    )  # Imprime la configuración inicial con el formato esperado del proyecto.

    execution_results: List[ModelExecutionResult] = (
        []
    )  # Acumula el resumen final de todos los modelos ejecutados.

    for index, model_key in enumerate(models_to_run, start=1):
        script_path = resolve_script_path(
            model_key
        )  # Resuelve la ruta oficial del script del modelo actual.
        command = build_command(
            script_path=script_path,
            config=config,
        )  # Construye el comando completo a ejecutar con subprocess.

        result = run_model_process(
            model_key=model_key,
            script_path=script_path,
            command=command,
            config=config,
            index=index,
            total_models=len(models_to_run),
        )  # Ejecuta el modelo actual, deja visible su consola y devuelve un resumen estructurado.

        execution_results.append(
            result
        )  # Agrega el resultado del modelo actual al consolidado final.

    summary_paths = export_consolidated_summary(
        common=common,
        config=config,
        resolved_scenarios=resolved_scenarios,
        results=execution_results,
    )  # Exporta el resumen consolidado completo en JSON y CSV.

    print_consolidated_summary(
        results=execution_results,
        summary_paths=summary_paths,
    )  # Imprime el cierre del experimento y las rutas finales generadas.

    if any(result.estado != "ok" for result in execution_results):
        raise SystemExit(
            1
        )  # Devuelve código 1 si al menos un modelo falló para que VS Code lo detecte.

    raise SystemExit(
        0
    )  # Devuelve código 0 cuando todos los modelos terminan correctamente.


if __name__ == "__main__":
    try:
        main()  # Lanza la ejecución completa del orquestador maestro.
    except Exception as exc:
        print(
            "ERROR EN 16_run_experimentos.py"
        )  # Imprime un encabezado simple de error para consola.
        print(str(exc))  # Muestra el detalle principal del fallo de forma entendible.
        raise SystemExit(
            1
        )  # Finaliza con código de error para que VS Code detecte la falla.
