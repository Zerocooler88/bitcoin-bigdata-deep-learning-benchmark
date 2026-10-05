from __future__ import (
    annotations,
)  # Permite referencias de tipos pospuestas para una anotación más limpia.

import argparse  # Procesa argumentos de consola para elegir escenarios y parámetros del experimento GRU.
import importlib.util  # Carga dinámicamente el archivo 10_common_training_pipeline.py.
import json  # Exporta el resumen consolidado del experimento en formato JSON.
import sys  # Registra el módulo cargado dinámicamente dentro del intérprete actual.
from dataclasses import (
    asdict,
)  # Convierte la configuración dataclass a diccionario serializable.
from datetime import (
    datetime,
    timezone,
)  # Genera marcas temporales UTC para el resumen consolidado.
from pathlib import (
    Path,
)  # Maneja rutas de archivos y carpetas de forma robusta en Windows.
from typing import (
    Any,
    Dict,
    List,
)  # Declara tipos para mejorar claridad y mantenimiento del código.

import pandas as pd  # Construye el resumen consolidado en CSV para Tableau.
import torch  # Proporciona tensores y utilidades centrales de PyTorch.
import torch.nn as nn  # Define módulos, capas y operaciones de la red GRU.


CURRENT_DIR = (
    Path(__file__).resolve().parent
)  # Resuelve la carpeta src donde vive este script.
COMMON_PIPELINE_PATH = (
    CURRENT_DIR / "10_common_training_pipeline.py"
)  # Apunta al pipeline común con la numeración nueva.
COMMON_MODULE_NAME = "common_training_pipeline_v10_for_gru"  # Define un nombre interno estable para registrar el módulo cargado.
MODEL_NAME = (
    "gru"  # Define el nombre oficial del modelo para métricas, rutas y consola.
)

DEFAULT_RNN_HIDDEN_SIZE = 64  # Define el tamaño del estado oculto recurrente.
DEFAULT_RNN_NUM_LAYERS = 2  # Define la cantidad de capas recurrentes apiladas.
DEFAULT_RNN_DROPOUT = (
    0.20  # Define un dropout moderado para estabilizar el entrenamiento.
)
DEFAULT_MLP_HIDDEN_SIZE = (
    32  # Define el tamaño de la capa oculta de la cabeza MLP final.
)


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
            "Entrena el modelo GRU reutilizando 10_common_training_pipeline.py "
            "con la misma filosofía de escenarios, métricas y exportaciones."
        )
    )  # Define una ayuda clara para consola y VS Code.

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
    )  # Mantiene una interfaz compatible con el resto del proyecto.

    parser.add_argument(
        "--lookback",
        type=int,
        default=48,
        help="Cantidad de pasos históricos usados por cada ventana temporal.",
    )  # Conserva el lookback por defecto solicitado para el experimento.
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Cantidad de pasos futuros a predecir.",
    )  # Conserva el horizon por defecto solicitado para el experimento.
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Tamaño de lote usado durante entrenamiento y evaluación.",
    )  # Conserva el batch_size por defecto solicitado para el experimento.
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número máximo de épocas de entrenamiento.",
    )  # Conserva el número máximo de épocas solicitado.
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciencia de early stopping.",
    )  # Conserva la paciencia solicitada para el experimento.
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje usada por Adam.",
    )  # Conserva la tasa de aprendizaje solicitada.
    parser.add_argument(
        "--target-mode",
        dest="target_mode",
        type=str,
        default="delta",
        help="Modo de target para el entrenamiento: 'price' o 'delta'.",
    )  # Mantiene la interfaz oficial del proyecto.
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
        )  # Evita configuraciones inválidas antes de empezar a preparar escenarios.


def validate_non_negative_integer(value: int, label: str) -> None:
    if value < 0:
        raise ValueError(
            f"El parámetro '{label}' no puede ser negativo. Valor recibido: {value}"
        )  # Evita configuraciones inválidas en parámetros como num_workers.


def validate_positive_float(value: float, label: str) -> None:
    if value <= 0:
        raise ValueError(
            f"El parámetro '{label}' debe ser mayor que cero. Valor recibido: {value}"
        )  # Evita configuraciones numéricas inválidas antes del experimento.


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
        return [
            "global"
        ]  # Usa global como respaldo si por alguna razón la lista quedó vacía.

    return normalized  # Devuelve la lista final de claves de escenario normalizadas.


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


def build_experiment_config(common: Any, args: argparse.Namespace) -> Any:
    validate_positive_integer(
        args.lookback, "lookback"
    )  # Valida lookback antes de construir la configuración.
    validate_positive_integer(
        args.horizon, "horizon"
    )  # Valida horizon antes de construir la configuración.
    validate_positive_integer(
        args.batch_size, "batch_size"
    )  # Valida batch_size antes del entrenamiento.
    validate_positive_integer(
        args.epochs, "epochs"
    )  # Valida epochs antes del entrenamiento.
    validate_positive_integer(
        args.patience, "patience"
    )  # Valida patience antes del entrenamiento.
    validate_positive_float(
        args.learning_rate, "learning_rate"
    )  # Valida learning_rate antes del entrenamiento.
    validate_non_negative_integer(
        args.num_workers, "num_workers"
    )  # Valida num_workers para Windows.
    common.validate_target_mode(
        args.target_mode
    )  # Valida el target_mode solicitado por interfaz.

    return common.ExperimentConfig(
        project_root=common.PROJECT_ROOT,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        random_seed=args.seed,
        model_subdir=MODEL_NAME,
        save_best_weights=True,
        verbose=True,
    )  # Construye la configuración base reutilizando exactamente la dataclass del pipeline común.


def print_header(common: Any, args: argparse.Namespace, scenarios: List[Any]) -> None:
    display_names = ", ".join(
        scenario.display_name for scenario in scenarios
    )  # Construye la lista visible de escenarios ya resueltos.
    target_mode = common.validate_target_mode(
        args.target_mode
    )  # Valida el modo de target una sola vez para imprimirlo de forma consistente.

    print("GRU")  # Imprime el encabezado principal del experimento.
    print(
        f"experimento        = {MODEL_NAME}"
    )  # Imprime el nombre estándar del modelo.
    print(
        f"lookback           = {args.lookback}"
    )  # Imprime el lookback solicitado por consola.
    print(
        f"horizon            = {args.horizon}"
    )  # Imprime el horizon solicitado por consola.
    print(
        f"batch_size         = {args.batch_size}"
    )  # Imprime batch_size del experimento.
    print(f"epochs             = {args.epochs}")  # Imprime epochs del experimento.
    print(f"patience           = {args.patience}")  # Imprime patience del experimento.
    print(
        f"num_workers        = {args.num_workers}"
    )  # Imprime num_workers para trazabilidad en Windows.
    print(
        f"seed               = {args.seed}"
    )  # Imprime la semilla aleatoria del experimento.
    print(
        f"learning_rate      = {args.learning_rate}"
    )  # Imprime learning_rate del experimento.
    print(
        f"target_mode        = {target_mode}"
    )  # Imprime el modo de target solicitado.
    print(
        f"escenarios         = {display_names}"
    )  # Imprime la lista oficial de escenarios a ejecutar.
    print()  # Inserta una línea en blanco para mejorar la legibilidad inicial.


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        lookback: int,
        hidden_size: int = DEFAULT_RNN_HIDDEN_SIZE,
        num_layers: int = DEFAULT_RNN_NUM_LAYERS,
        dropout: float = DEFAULT_RNN_DROPOUT,
        mlp_hidden_size: int = DEFAULT_MLP_HIDDEN_SIZE,
    ) -> None:
        super().__init__()  # Inicializa correctamente el módulo base de PyTorch.

        if input_size <= 0:
            raise ValueError(
                f"input_size debe ser mayor que cero. Valor recibido: {input_size}"
            )  # Evita construir una red inválida si no hay variables de entrada.
        if lookback <= 0:
            raise ValueError(
                f"lookback debe ser mayor que cero. Valor recibido: {lookback}"
            )  # Valida la dimensión temporal esperada.
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size debe ser mayor que cero. Valor recibido: {hidden_size}"
            )  # Evita una configuración GRU inválida.
        if num_layers <= 0:
            raise ValueError(
                f"num_layers debe ser mayor que cero. Valor recibido: {num_layers}"
            )  # Evita una configuración recurrente inválida.
        if mlp_hidden_size <= 0:
            raise ValueError(
                f"mlp_hidden_size debe ser mayor que cero. Valor recibido: {mlp_hidden_size}"
            )  # Evita una cabeza densa inválida.
        if dropout < 0:
            raise ValueError(
                f"dropout no puede ser negativo. Valor recibido: {dropout}"
            )  # Evita una configuración inválida de dropout.

        effective_recurrent_dropout = (
            dropout if num_layers > 1 else 0.0
        )  # PyTorch solo aplica dropout interno si num_layers es mayor que uno.

        self.input_size = (
            input_size  # Conserva la dimensión de entrada para trazabilidad del modelo.
        )
        self.lookback = (
            lookback  # Conserva la longitud temporal de entrada para trazabilidad.
        )
        self.hidden_size = (
            hidden_size  # Conserva el tamaño del estado oculto para trazabilidad.
        )
        self.num_layers = (
            num_layers  # Conserva la cantidad de capas recurrentes para trazabilidad.
        )
        self.dropout_rate = dropout  # Conserva la tasa de dropout declarada.
        self.mlp_hidden_size = (
            mlp_hidden_size  # Conserva el tamaño de la capa oculta final.
        )

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_recurrent_dropout,
        )  # Define la GRU principal que procesa la secuencia temporal completa.

        self.output_dropout = nn.Dropout(
            p=dropout
        )  # Aplica regularización adicional antes de la cabeza MLP final.

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden_size, 1),
        )  # Define una cabeza MLP homogénea para comparar de forma justa con LSTM.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_output, _ = self.gru(
            x
        )  # Procesa la secuencia completa y devuelve estados ocultos por paso temporal.
        last_time_step = gru_output[
            :, -1, :
        ]  # Toma el último paso temporal como resumen secuencial de la ventana.
        last_time_step = self.output_dropout(
            last_time_step
        )  # Aplica dropout antes de la regresión final.
        prediction = self.regression_head(
            last_time_step
        )  # Proyecta el resumen temporal a una salida escalar.
        return (
            prediction  # Devuelve un tensor [batch_size, 1] compatible con el pipeline.
        )


def model_builder(input_size: int, lookback: int) -> nn.Module:
    return GRURegressor(
        input_size=input_size,
        lookback=lookback,
        hidden_size=DEFAULT_RNN_HIDDEN_SIZE,
        num_layers=DEFAULT_RNN_NUM_LAYERS,
        dropout=DEFAULT_RNN_DROPOUT,
        mlp_hidden_size=DEFAULT_MLP_HIDDEN_SIZE,
    )  # Construye una instancia GRU consistente con el experimento.


def run_gru_for_scenario(
    common: Any,
    scenario: Any,
    config: Any,
    train_target_mode: str,
) -> Dict[str, Any]:
    print(
        f"Ejecutando {MODEL_NAME} en escenario: {scenario.display_name}"
    )  # Informa claramente qué escenario está corriendo en este momento.

    summary = common.run_training_for_scenario(
        scenario=scenario,
        config=config,
        model_name=MODEL_NAME,
        model_builder=model_builder,
        train_target_mode=train_target_mode,
    )  # Reutiliza el entrenamiento, evaluación y exportación oficiales del pipeline común.

    print()  # Inserta una línea en blanco entre escenarios para una consola más limpia.
    return summary  # Devuelve el resumen oficial del escenario ya exportado.


def build_consolidated_record(
    scenario_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "modelo": scenario_summary.get("model_name", MODEL_NAME),
        "escenario": scenario_summary.get("scenario_name", ""),
        "best_epoch": int(scenario_summary.get("best_epoch", 0)),
        "training_target_mode": scenario_summary.get("training_target_mode", "delta"),
        "validation_metrics": scenario_summary.get("validation_metrics", {}),
        "test_metrics": scenario_summary.get("test_metrics", {}),
        "baseline_validation_metrics": scenario_summary.get(
            "baseline_validation_metrics", {}
        ),
        "baseline_test_metrics": scenario_summary.get("baseline_test_metrics", {}),
        "paths": scenario_summary.get("paths", {}),
    }  # Normaliza el resumen a la estructura consolidada pedida para JSON y CSV.


def build_consolidated_payload(
    common: Any,
    config: Any,
    args: argparse.Namespace,
    requested_scenarios: List[Any],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    config_dict = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }  # Convierte la configuración dataclass a un diccionario serializable y amigable para JSON.

    return {
        "modelo": MODEL_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios_requested": [
            scenario.display_name for scenario in requested_scenarios
        ],
        "config": {
            **config_dict,
            "target_mode": common.validate_target_mode(args.target_mode),
        },
        "arquitectura_modelo": {
            "tipo": "GRURegressor",
            "hidden_size": DEFAULT_RNN_HIDDEN_SIZE,
            "num_layers": DEFAULT_RNN_NUM_LAYERS,
            "dropout": DEFAULT_RNN_DROPOUT,
            "mlp_hidden_size": DEFAULT_MLP_HIDDEN_SIZE,
            "cabeza_salida": "Linear(hidden_size, 32) -> ReLU -> Dropout -> Linear(32, 1)",
        },
        "resultados": records,
    }  # Construye el JSON consolidado final con configuración, arquitectura y resultados.


def build_consolidated_csv_rows(
    common: Any,
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = (
        []
    )  # Acumula filas planas compatibles con Tableau.

    for record in records:
        validation = record.get(
            "validation_metrics", {}
        )  # Recupera métricas de validation del escenario actual.
        test = record.get(
            "test_metrics", {}
        )  # Recupera métricas de test del escenario actual.
        baseline_val = record.get(
            "baseline_validation_metrics", {}
        )  # Recupera el baseline de validation del escenario actual.
        baseline_test = record.get(
            "baseline_test_metrics", {}
        )  # Recupera el baseline de test del escenario actual.
        paths = record.get(
            "paths", {}
        )  # Recupera rutas exportadas del escenario actual.

        flat_rows.append(
            {
                "modelo": record.get("modelo", MODEL_NAME),
                "escenario": record.get("escenario", ""),
                "mejor_epoca": record.get("best_epoch", 0),
                "modo_target_entrenamiento": common.translate_target_mode_label(
                    args.target_mode
                ),
                "val_mae": validation.get("MAE"),
                "val_rmse": validation.get("RMSE"),
                "val_mse": validation.get("MSE"),
                "val_mape": validation.get("MAPE"),
                "val_r_cuadrado": validation.get("R2"),
                "test_mae": test.get("MAE"),
                "test_rmse": test.get("RMSE"),
                "test_mse": test.get("MSE"),
                "test_mape": test.get("MAPE"),
                "test_r_cuadrado": test.get("R2"),
                "baseline_val_mae": baseline_val.get("MAE"),
                "baseline_val_rmse": baseline_val.get("RMSE"),
                "baseline_val_mse": baseline_val.get("MSE"),
                "baseline_val_mape": baseline_val.get("MAPE"),
                "baseline_val_r_cuadrado": baseline_val.get("R2"),
                "baseline_test_mae": baseline_test.get("MAE"),
                "baseline_test_rmse": baseline_test.get("RMSE"),
                "baseline_test_mse": baseline_test.get("MSE"),
                "baseline_test_mape": baseline_test.get("MAPE"),
                "baseline_test_r_cuadrado": baseline_test.get("R2"),
                "path_pred_validation_csv": paths.get("pred_val"),
                "path_pred_test_csv": paths.get("pred_test"),
                "path_metrics_csv": paths.get("metrics_csv"),
                "path_metrics_json": paths.get("metrics_json"),
                "path_history_csv": paths.get("history_csv"),
                "path_weights_pt": paths.get("weights_pt"),
            }
        )  # Construye una fila ancha y consistente para dashboards y uniones en Tableau.

    return flat_rows  # Devuelve la tabla plana lista para convertirse en DataFrame.


def export_consolidated_summary(
    common: Any,
    config: Any,
    args: argparse.Namespace,
    requested_scenarios: List[Any],
    records: List[Dict[str, Any]],
) -> Dict[str, str]:
    metrics_dir = common.ensure_directory(
        config.project_root / "outputs" / "metrics"
    )  # Asegura la carpeta de métricas usando la misma lógica del pipeline común.

    json_path = (
        metrics_dir / f"{MODEL_NAME}__resumen_consolidado.json"
    )  # Define la ruta del JSON consolidado final.
    csv_path = (
        metrics_dir / f"{MODEL_NAME}__resumen_consolidado.csv"
    )  # Define la ruta del CSV consolidado final.

    payload = build_consolidated_payload(
        common=common,
        config=config,
        args=args,
        requested_scenarios=requested_scenarios,
        records=records,
    )  # Construye el documento JSON completo del resumen consolidado.

    csv_rows = build_consolidated_csv_rows(
        common=common,
        args=args,
        records=records,
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
    records: List[Dict[str, Any]],
    summary_paths: Dict[str, str],
) -> None:
    print(
        "RESUMEN CONSOLIDADO FINAL"
    )  # Imprime el encabezado del cierre del experimento.

    if not records:
        print(
            "No se generaron resultados para resumir."
        )  # Informa si no hubo escenarios exitosos.
        return

    for record in records:
        validation = record.get(
            "validation_metrics", {}
        )  # Recupera métricas de validation del escenario actual.
        test = record.get(
            "test_metrics", {}
        )  # Recupera métricas de test del escenario actual.

        val_rmse = validation.get(
            "RMSE", float("nan")
        )  # Recupera RMSE de validation para la línea resumen.
        test_rmse = test.get(
            "RMSE", float("nan")
        )  # Recupera RMSE de test para la línea resumen.

        print(
            f"- {record.get('escenario', '')} | "
            f"best_epoch={record.get('best_epoch', 0)} | "
            f"val_rmse={val_rmse:.6f} | "
            f"test_rmse={test_rmse:.6f}"
        )  # Imprime una línea compacta por escenario para lectura rápida en consola.

    print()  # Inserta una línea en blanco antes de mostrar las rutas finales.
    print(
        f"resumen_json      = {summary_paths['json']}"
    )  # Informa la ruta final del JSON consolidado.
    print(
        f"resumen_csv       = {summary_paths['csv']}"
    )  # Informa la ruta final del CSV consolidado.


def main() -> None:
    common = (
        load_common_pipeline_module()
    )  # Carga dinámicamente el pipeline común numerado como 10.
    args = parse_args()  # Lee la configuración de ejecución enviada por consola.
    requested_keys = normalize_scenario_keys(
        args.scenarios
    )  # Limpia y normaliza las claves de escenarios pedidas por el usuario.
    scenarios = resolve_scenarios(
        common, requested_keys
    )  # Resuelve la lista final de escenarios usando el catálogo oficial.
    config = build_experiment_config(
        common, args
    )  # Construye la configuración del experimento reutilizando la dataclass oficial.
    target_mode = common.validate_target_mode(
        args.target_mode
    )  # Valida una sola vez el modo de target para usarlo en toda la ejecución.

    print_header(
        common, args, scenarios
    )  # Imprime la configuración inicial con el formato esperado del proyecto.

    consolidated_records: List[Dict[str, Any]] = (
        []
    )  # Acumula el resumen final de todos los escenarios ejecutados.

    for scenario in scenarios:
        try:
            scenario_summary = run_gru_for_scenario(
                common=common,
                scenario=scenario,
                config=config,
                train_target_mode=target_mode,
            )  # Ejecuta el entrenamiento GRU del escenario actual usando exclusivamente lógica común.
        except Exception as exc:
            raise RuntimeError(
                f"Falló la ejecución del escenario '{scenario.display_name}'. "
                f"Detalle: {exc}"
            ) from exc  # Reempaqueta el error con un mensaje más claro para Windows y VS Code.

        consolidated_records.append(
            build_consolidated_record(scenario_summary)
        )  # Agrega el resumen normalizado del escenario al acumulado final.

    summary_paths = export_consolidated_summary(
        common=common,
        config=config,
        args=args,
        requested_scenarios=scenarios,
        records=consolidated_records,
    )  # Exporta el resumen consolidado completo en JSON y CSV.

    print_consolidated_summary(
        records=consolidated_records,
        summary_paths=summary_paths,
    )  # Imprime el cierre del experimento y las rutas finales generadas.


if __name__ == "__main__":
    try:
        main()  # Lanza la ejecución completa del experimento GRU.
    except Exception as exc:
        print("ERROR EN 13_entrenar_gru.py")  # Imprime un encabezado simple de error.
        print(str(exc))  # Muestra el detalle principal del fallo de forma entendible.
        raise SystemExit(
            1
        )  # Finaliza con código de error para que VS Code detecte la falla.
