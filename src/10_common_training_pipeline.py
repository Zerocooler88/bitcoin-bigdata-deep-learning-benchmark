from __future__ import (
    annotations,
)  # Permite referencias de tipos pospuestas para una anotación más limpia.

import json  # Serializa y deserializa configuraciones y reportes en formato JSON.
import math  # Proporciona funciones matemáticas auxiliares para métricas y validaciones numéricas.
import random  # Controla la aleatoriedad básica para reproducibilidad experimental.
import time  # Mide tiempos de entrenamiento, evaluación y velocidad de procesamiento.
from copy import deepcopy  # Clona objetos complejos sin compartir referencias internas.
from dataclasses import (
    asdict,
    dataclass,
    field,
)  # Define estructuras de configuración limpias y exportables.
from pathlib import (
    Path,
)  # Maneja rutas de archivos y carpetas de forma robusta en Windows.
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)  # Declara tipos para mejorar claridad y mantenimiento del código.

import numpy as np  # Manipula arreglos numéricos y ventanas temporales de forma eficiente.
import pandas as pd  # Lee parquets y organiza datasets tabulares para el pipeline experimental.
import torch  # Proporciona tensores, GPU y utilidades centrales de PyTorch.
import torch.backends.cudnn as cudnn  # Controla optimizaciones cuDNN para entrenamiento en GPU.
import torch.nn as nn  # Define capas, pérdidas y módulos de redes neuronales.
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)  # Calcula métricas de regresión para validación y prueba.
from sklearn.preprocessing import (
    StandardScaler,
)  # Estandariza variables de entrada y salida sin fuga de información.
from torch.nn.utils.clip_grad import (
    clip_grad_norm_,
)  # Importa clipping de gradiente desde la API pública.
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # Construye datasets y cargadores por lotes para entrenamiento en PyTorch.
from tqdm.auto import (
    tqdm,
)  # Muestra barras de progreso automáticas en consola durante entrenamiento y evaluación.

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)  # Resuelve la raíz del proyecto desde la carpeta src.

DATA_1H_DIR = (
    PROJECT_ROOT / "data" / "1h"
)  # Centraliza la carpeta principal de datasets 1h.
DEFAULT_GLOBAL_PARQUET = (
    DATA_1H_DIR / "global" / "btcusdt_spot_1h_model_input_2019_2026.parquet"
)  # Define la ruta esperada del parquet global.
DEFAULT_TRIMESTRES_DIR = (
    DATA_1H_DIR / "trimestres"
)  # Define la carpeta esperada de parquets trimestrales.
DEFAULT_OUTPUTS_DIR = (
    PROJECT_ROOT / "outputs"
)  # Centraliza la carpeta de salidas del proyecto.
DEFAULT_MODELS_DIR = (
    PROJECT_ROOT / "models"
)  # Centraliza la carpeta de pesos entrenados.
EPSILON = 1e-8  # Evita divisiones por cero en métricas porcentuales.
CANONICAL_FEATURES_22 = [
    "anio",
    "mes",
    "dia",
    "hora",
    "trimestre",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "rango_hl",
    "cuerpo_oc",
    "return_1h",
    "log_return_1h",
    "sma_24h",
    "sma_168h",
    "volatilidad_24h",
    "volumen_sma_24h",
]

assert len(CANONICAL_FEATURES_22) == 22


@dataclass
class ExperimentConfig:
    project_root: Path = (
        PROJECT_ROOT  # Conserva una única raíz de trabajo para el experimento.
    )
    lookback: int = 48  # Fija la ventana histórica por defecto en 48 horas.
    horizon: int = 1  # Fija la predicción por defecto a una hora hacia adelante.
    batch_size: int = 64  # Mantiene el tamaño de lote comparable entre modelos.
    epochs: int = 50  # Define el máximo de épocas de entrenamiento.
    patience: int = 5  # Activa early stopping con tolerancia de cinco épocas.
    learning_rate: float = 1e-3  # Mantiene Adam con tasa de aprendizaje estable.
    num_workers: int = 0  # Prioriza estabilidad inicial en Windows.
    random_seed: int = 42  # Hace reproducible la partición y el entrenamiento.
    train_ratio: float = (
        0.70  # Usa 70% del tramo para entrenamiento en escenarios cerrados.
    )
    val_ratio: float = 0.15  # Usa 15% del tramo para validación en escenarios cerrados.
    test_ratio: float = 0.15  # Usa 15% del tramo para prueba en escenarios cerrados.
    target_column: Optional[str] = (
        None  # Permite forzar la columna objetivo si ya está definida.
    )
    timestamp_column: Optional[str] = (
        None  # Permite forzar la columna temporal si ya está definida.
    )
    feature_columns: Optional[List[str]] = field(
        default_factory=lambda: CANONICAL_FEATURES_22.copy()
    )  # Permite forzar las columnas predictoras si ya están definidas.
    model_subdir: str = "generic"  # Organiza pesos por familia de modelo.
    save_best_weights: bool = (
        True  # Guarda el mejor checkpoint si el entrenamiento mejora.
    )
    verbose: bool = True  # Habilita mensajes de trazabilidad en consola.
    max_grad_norm: float = (
        1.0  # Aplica clipping de gradiente para estabilizar redes recurrentes.
    )


@dataclass
class ScenarioDefinition:
    key: str  # Define la clave corta usada por el orquestador.
    display_name: str  # Define el nombre visible en archivos y consola.
    mode: str  # Distingue escenarios globales, filtrados y cross.
    source_path: Optional[Path] = None  # Apunta al parquet principal del escenario.
    train_path: Optional[Path] = (
        None  # Apunta al parquet de entrenamiento para cross si se usa ruta directa.
    )
    test_path: Optional[Path] = (
        None  # Apunta al parquet de prueba para cross si se usa ruta directa.
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Conserva metadatos útiles del escenario.


@dataclass
class WindowedSplit:
    x_raw: np.ndarray  # Conserva ventanas sin escalar para trazabilidad.
    y_raw: np.ndarray  # Conserva el precio objetivo real en unidades monetarias.
    timestamps: np.ndarray  # Conserva la marca temporal asociada a cada objetivo.
    baseline_raw: np.ndarray  # Conserva la persistencia ingenua en escala real.
    train_target_raw: (
        np.ndarray
    )  # Conserva el objetivo real de entrenamiento según el modo elegido.
    target_mode: str = (
        "price"  # Indica si el entrenamiento usa precio directo o delta residual.
    )
    x_scaled: Optional[np.ndarray] = None  # Conserva ventanas escaladas para PyTorch.
    train_target_scaled: Optional[np.ndarray] = (
        None  # Conserva el objetivo escalado de entrenamiento para PyTorch.
    )


@dataclass
class PreparedScenario:
    scenario: ScenarioDefinition  # Conserva la definición original del escenario.
    timestamp_column: str  # Conserva la columna temporal detectada o forzada.
    target_column: str  # Conserva la variable objetivo detectada o forzada.
    feature_columns: List[
        str
    ]  # Conserva las variables predictoras efectivamente usadas.
    training_target_mode: (
        str  # Conserva si el entrenamiento usa precio o delta residual.
    )
    x_scaler: StandardScaler  # Conserva el escalador ajustado solo con entrenamiento.
    y_scaler: StandardScaler  # Conserva el escalador del objetivo de entrenamiento.
    splits: Dict[str, WindowedSplit]  # Conserva los tres subconjuntos ya vectorizados.
    loaders: Dict[
        str, DataLoader
    ]  # Conserva los DataLoader listos para entrenamiento y evaluación.
    raw_frames: Dict[str, pd.DataFrame]  # Conserva los DataFrame base por trazabilidad.


class WindowDataset(Dataset):
    def __init__(self, x_scaled: np.ndarray, target_scaled: np.ndarray) -> None:
        self.x = torch.from_numpy(
            x_scaled.astype(np.float32)
        )  # Convierte las ventanas escaladas a tensores float32.
        self.y = torch.from_numpy(
            target_scaled.astype(np.float32)
        )  # Convierte el objetivo escalado a tensores float32.

    def __len__(self) -> int:
        return len(self.x)  # Devuelve la cantidad de muestras disponibles.

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.y[index],
        )  # Entrega una ventana y su objetivo en el índice solicitado.


class EarlyStoppingState:
    def __init__(self, patience: int) -> None:
        self.patience = (
            patience  # Conserva la paciencia configurada para early stopping.
        )
        self.best_val_mse = float(
            "inf"
        )  # Inicializa el mejor error de validación con infinito.
        self.best_epoch = 0  # Inicializa la mejor época en cero.
        self.wait = 0  # Inicializa el contador de épocas sin mejora.
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = (
            None  # Conserva el mejor estado del modelo.
        )

    def update(self, model: nn.Module, epoch: int, val_mse: float) -> bool:
        if val_mse < self.best_val_mse:
            self.best_val_mse = float(
                val_mse
            )  # Actualiza el mejor error observado en validación.
            self.best_epoch = epoch  # Registra la época asociada al mejor resultado.
            self.wait = 0  # Reinicia el contador al detectar mejora.
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }  # Clona los pesos del mejor modelo.
            return True  # Indica que hubo mejora.
        self.wait += 1  # Incrementa el contador cuando no hubo mejora.
        return False  # Indica que la época actual no mejoró al mejor modelo.

    @property
    def should_stop(self) -> bool:
        return (
            self.wait >= self.patience
        )  # Activa el corte cuando se consume la paciencia.


def validate_target_mode(target_mode: str) -> str:
    normalized_mode = (
        target_mode.strip().lower()
    )  # Normaliza el modo solicitado para validarlo de forma robusta.
    if normalized_mode not in {"price", "delta"}:
        raise ValueError(
            f"Modo de target no soportado: {target_mode}. Usa 'price' o 'delta'."
        )  # Informa modos inválidos antes de preparar ventanas o entrenamiento.
    return normalized_mode  # Devuelve el modo validado y normalizado.


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")  # Usa CPU cuando CUDA no está disponible.


def ensure_directory(path: Path) -> Path:
    path.mkdir(
        parents=True, exist_ok=True
    )  # Crea la carpeta solicitada si todavía no existe.
    return path  # Devuelve la misma ruta para facilitar composición.


def require_path(path_value: Optional[Path], label: str) -> Path:
    if path_value is None:
        raise ValueError(
            f"La ruta requerida '{label}' no fue definida en el escenario."
        )  # Valida que la ruta obligatoria exista antes de usarla.
    return path_value  # Devuelve una ruta tipada como Path para satisfacer a Pylance.


def require_array(array_value: Optional[np.ndarray], label: str) -> np.ndarray:
    if array_value is None:
        raise ValueError(
            f"El arreglo requerido '{label}' no fue generado correctamente."
        )  # Valida que el arreglo escalado exista antes de construir el DataLoader.
    return array_value  # Devuelve un arreglo tipado como ndarray para satisfacer a Pylance.


def resolve_quarter_path(project_root: Path, quarter_label: str) -> Path:
    filename = f"btcusdt_spot_1h_{quarter_label}_model_input.parquet"  # Construye el nombre esperado del archivo trimestral.
    return (
        project_root / "data" / "1h" / "trimestres" / filename
    )  # Devuelve la ruta absoluta del parquet trimestral.


def build_default_scenarios(
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, ScenarioDefinition]:
    global_path = (
        project_root
        / "data"
        / "1h"
        / "global"
        / "btcusdt_spot_1h_model_input_2019_2026.parquet"
    )  # Define la ruta del escenario global.

    return {
        "global": ScenarioDefinition(
            key="global",
            display_name="global_2019_2026",
            mode="global",
            source_path=global_path,
        ),
        "pre_covid": ScenarioDefinition(
            key="pre_covid",
            display_name="pre_covid",
            mode="filtered",
            source_path=global_path,
            metadata={"filter": {"type": "regime", "value": "pre_covid"}},
        ),
        "covid": ScenarioDefinition(
            key="covid",
            display_name="covid",
            mode="filtered",
            source_path=global_path,
            metadata={"filter": {"type": "regime", "value": "covid"}},
        ),
        "post_covid": ScenarioDefinition(
            key="post_covid",
            display_name="post_covid",
            mode="filtered",
            source_path=global_path,
            metadata={"filter": {"type": "regime", "value": "post_covid"}},
        ),
        "trimestre_mas_bajo": ScenarioDefinition(
            key="trimestre_mas_bajo",
            display_name="trimestre_mas_bajo",
            mode="filtered",
            source_path=global_path,
            metadata={
                "filter": {"type": "extreme_quarter", "value": "lowest_mean_close"}
            },
        ),
        "trimestre_mas_alto": ScenarioDefinition(
            key="trimestre_mas_alto",
            display_name="trimestre_mas_alto",
            mode="filtered",
            source_path=global_path,
            metadata={
                "filter": {"type": "extreme_quarter", "value": "highest_mean_close"}
            },
        ),
        "intra_2024_Q1": ScenarioDefinition(
            key="intra_2024_Q1",
            display_name="intra_2024_Q1",
            mode="filtered",
            source_path=global_path,
            metadata={"filter": {"type": "quarter", "value": "2024-Q1"}},
        ),
        "intra_2024_Q3": ScenarioDefinition(
            key="intra_2024_Q3",
            display_name="intra_2024_Q3",
            mode="filtered",
            source_path=global_path,
            metadata={"filter": {"type": "quarter", "value": "2024-Q3"}},
        ),
        "intra_2025_Q1": ScenarioDefinition(
            key="intra_2025_Q1",
            display_name="intra_2025_Q1",
            mode="filtered",
            source_path=global_path,
            metadata={"filter": {"type": "quarter", "value": "2025-Q1"}},
        ),
        "cross_2024_Q1_to_2024_Q3": ScenarioDefinition(
            key="cross_2024_Q1_to_2024_Q3",
            display_name="cross_2024_Q1_to_2024_Q3",
            mode="cross",
            source_path=global_path,
            metadata={
                "train_filter": {"type": "quarter", "value": "2024-Q1"},
                "test_filter": {"type": "quarter", "value": "2024-Q3"},
            },
        ),
        "cross_2024_Q1_to_2025_Q1": ScenarioDefinition(
            key="cross_2024_Q1_to_2025_Q1",
            display_name="cross_2024_Q1_to_2025_Q1",
            mode="cross",
            source_path=global_path,
            metadata={
                "train_filter": {"type": "quarter", "value": "2024-Q1"},
                "test_filter": {"type": "quarter", "value": "2025-Q1"},
            },
        ),
        "cross_pre_to_post": ScenarioDefinition(
            key="cross_pre_to_post",
            display_name="cross_pre_to_post",
            mode="cross",
            source_path=global_path,
            metadata={
                "train_filter": {"type": "regime", "value": "pre_covid"},
                "test_filter": {"type": "regime", "value": "post_covid"},
            },
        ),
        "cross_bajo_to_alto": ScenarioDefinition(
            key="cross_bajo_to_alto",
            display_name="cross_bajo_to_alto",
            mode="cross",
            source_path=global_path,
            metadata={
                "train_filter": {
                    "type": "extreme_quarter",
                    "value": "lowest_mean_close",
                },
                "test_filter": {
                    "type": "extreme_quarter",
                    "value": "highest_mean_close",
                },
            },
        ),
    }  # Regresa el catálogo fijo de escenarios obligatorios.


def get_scenarios_from_keys(
    keys: Sequence[str], project_root: Path = PROJECT_ROOT
) -> List[ScenarioDefinition]:
    catalog = build_default_scenarios(
        project_root=project_root
    )  # Construye el catálogo estándar desde la raíz del proyecto.
    resolved: List[ScenarioDefinition] = (
        []
    )  # Inicializa la lista de escenarios solicitados.
    for key in keys:
        if key not in catalog:
            available = ", ".join(
                sorted(catalog.keys())
            )  # Construye el listado visible de escenarios válidos.
            raise KeyError(
                f"Escenario no reconocido: {key}. Opciones válidas: {available}"
            )  # Informa escenarios inválidos de forma explícita.
        resolved.append(
            catalog[key]
        )  # Agrega el escenario solicitado a la lista final.
    return resolved  # Devuelve escenarios listos para preparación.


def infer_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [
        "open_datetime_utc",
        "datetime_utc",
        "open_time_utc",
        "timestamp_utc",
        "datetime",
        "timestamp",
        "open_time",
        "date",
    ]  # Define nombres de columnas temporales comunes en datasets OHLCV.
    for candidate in candidates:
        if candidate in df.columns:
            return candidate  # Devuelve la primera coincidencia encontrada en el dataframe.
    for column in df.columns:
        lower_name = (
            column.lower()
        )  # Convierte el nombre actual a minúsculas para la heurística de respaldo.
        if "time" in lower_name or "date" in lower_name:
            return (
                column  # Usa una heurística segura si la convención exacta no coincide.
            )
    raise KeyError(
        "No fue posible detectar la columna temporal del parquet."
    )  # Informa explícitamente la ausencia de columna temporal.


def load_parquet_frame(
    parquet_path: Path, timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo parquet: {parquet_path}"
        )  # Detiene el flujo si falta el insumo esperado.
    df = pd.read_parquet(
        parquet_path
    )  # Lee el parquet completo en memoria usando pandas.
    ts_col = timestamp_column or infer_timestamp_column(
        df
    )  # Detecta la columna temporal si no fue forzada.
    df[ts_col] = pd.to_datetime(
        df[ts_col], utc=True, errors="coerce"
    )  # Convierte la columna temporal a datetime UTC.
    df = (
        df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    )  # Elimina fechas inválidas y ordena cronológicamente.
    return df  # Devuelve el dataframe limpio y ordenado.


def infer_target_column(df: pd.DataFrame, forced_target: Optional[str] = None) -> str:
    if forced_target is not None:
        if forced_target not in df.columns:
            raise KeyError(
                f"La columna objetivo forzada no existe: {forced_target}"
            )  # Valida que la columna objetivo forzada exista.
        return forced_target  # Respeta la columna objetivo enviada por el usuario.
    candidates = [
        "target_close",
        "close",
        "close_price",
        "price_close",
        "target",
        "y",
        "label",
    ]  # Prioriza target_close cuando el dataset ya fue preparado para modelado.
    for candidate in candidates:
        if candidate in df.columns:
            return candidate  # Devuelve la primera coincidencia compatible con el objetivo.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()  # Extrae columnas numéricas disponibles en el parquet.
    if not numeric_columns:
        raise KeyError(
            "No fue posible detectar una columna objetivo numérica."
        )  # Detiene el flujo si no hay variables numéricas.
    return numeric_columns[
        -1
    ]  # Usa la última columna numérica como respaldo conservador.


def infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    forced_features: Optional[Sequence[str]] = None,
) -> List[str]:
    if forced_features is not None:
        missing = [
            column for column in forced_features if column not in df.columns
        ]  # Detecta variables forzadas que no existen.
        if missing:
            raise KeyError(
                f"Columnas predictoras inexistentes: {missing}"
            )  # Informa predictoras forzadas inválidas.
        return list(
            forced_features
        )  # Respeta las columnas predictoras enviadas manualmente.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()  # Recupera todas las columnas numéricas del parquet.
    feature_columns = [
        column
        for column in numeric_columns
        if column != target_column and column != timestamp_column
    ]  # Excluye objetivo y tiempo de las predictoras.
    if not feature_columns:
        raise ValueError(
            "No existen columnas numéricas suficientes para construir las ventanas de entrada."
        )  # Detiene el flujo si no hay predictoras válidas.
    return feature_columns  # Devuelve el conjunto final de variables de entrada.


def clean_frame_for_modeling(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    required_columns = list(feature_columns) + [
        target_column,
        timestamp_column,
    ]  # Reúne las columnas mínimas necesarias para entrenar y evaluar.
    cleaned_df = df.dropna(
        subset=required_columns
    ).copy()  # Elimina filas con NaN en columnas críticas.
    cleaned_df = cleaned_df.sort_values(timestamp_column).reset_index(
        drop=True
    )  # Reordena cronológicamente después de la limpieza.
    if len(cleaned_df) == 0:
        raise ValueError(
            "Después de eliminar NaN en variables predictoras, objetivo y tiempo, el DataFrame quedó vacío."
        )  # Detiene el flujo si la limpieza deja el dataframe vacío.
    return cleaned_df  # Devuelve el dataframe limpio y listo para el modelado.


def split_frame_chronologically(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float
) -> Dict[str, pd.DataFrame]:
    if not math.isclose(
        train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError(
            "Las proporciones train/val/test deben sumar 1.0."
        )  # Asegura particiones coherentes antes de dividir.
    n_rows = len(df)  # Cuenta la cantidad total de observaciones disponibles.
    if n_rows < 10:
        raise ValueError(
            "El dataset es demasiado pequeño para una partición temporal estable."
        )  # Evita experimentos inválidos con pocos registros.
    train_end = max(
        1, int(n_rows * train_ratio)
    )  # Calcula el límite superior del bloque de entrenamiento.
    val_end = max(
        train_end + 1, int(n_rows * (train_ratio + val_ratio))
    )  # Calcula el límite superior del bloque de validación.
    train_df = df.iloc[
        :train_end
    ].copy()  # Extrae el tramo cronológico de entrenamiento.
    val_df = df.iloc[
        train_end:val_end
    ].copy()  # Extrae el tramo cronológico de validación.
    test_df = df.iloc[val_end:].copy()  # Extrae el tramo cronológico de prueba.
    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "La partición temporal generó un conjunto vacío en validación o prueba."
        )  # Evita escenarios sin evaluación válida.
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }  # Devuelve los tres subconjuntos temporales.


def split_cross_quarter_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, pd.DataFrame]:
    source_ratio_sum = (
        train_ratio + val_ratio
    )  # Suma las proporciones internas del bloque fuente.
    if source_ratio_sum <= 0:
        raise ValueError(
            "La suma de train_ratio y val_ratio debe ser mayor que cero para escenarios cross."
        )  # Valida proporciones antes de dividir.
    train_portion = (
        train_ratio / source_ratio_sum
    )  # Reescala train/val sobre el bloque fuente.
    n_rows = len(source_df)  # Cuenta observaciones disponibles en el bloque fuente.
    if n_rows < 10:
        raise ValueError(
            "El bloque fuente es demasiado pequeño para generar train y validation."
        )  # Evita cross con base insuficiente.
    train_end = max(
        1, int(n_rows * train_portion)
    )  # Calcula el final cronológico del bloque de entrenamiento.
    train_df = source_df.iloc[
        :train_end
    ].copy()  # Usa la parte inicial del bloque fuente para entrenar.
    val_df = source_df.iloc[
        train_end:
    ].copy()  # Usa la parte final del bloque fuente para validar.
    test_df = target_df.copy()  # Usa el bloque destino completo como prueba externa.
    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "El escenario cross generó un conjunto vacío en validation o test."
        )  # Asegura que ambos subconjuntos existan.
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }  # Devuelve la estructura cronológica del escenario cross.


def build_quarter_summary(
    df: pd.DataFrame, quarter_column: str, target_column: str
) -> pd.DataFrame:
    if quarter_column not in df.columns:
        raise KeyError(
            f"No existe la columna de trimestre requerida: {quarter_column}"
        )  # Valida que el dataset tenga la columna year_quarter.
    if target_column not in df.columns:
        raise KeyError(
            f"No existe la columna objetivo requerida para resumir trimestres: {target_column}"
        )  # Valida que el dataset tenga la columna objetivo.
    summary = (
        df.groupby(quarter_column, dropna=False)[target_column]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "filas",
                "mean": "close_promedio",
                "min": "close_minimo",
                "max": "close_maximo",
            }
        )
        .sort_values(quarter_column)
        .reset_index(drop=True)
    )  # Resume filas y estadísticos del close por trimestre.
    return summary  # Devuelve el resumen trimestral ordenado.


def resolve_extreme_quarter_label(
    df: pd.DataFrame, target_column: str, extreme_value: str
) -> str:
    summary = build_quarter_summary(
        df, quarter_column="year_quarter", target_column=target_column
    )  # Calcula el resumen por trimestre para hallar extremos.
    if len(summary) == 0:
        raise ValueError(
            "No fue posible calcular el trimestre extremo porque el dataset está vacío."
        )  # Detiene el flujo si no hay trimestres disponibles.
    if extreme_value == "lowest_mean_close":
        return str(
            summary.sort_values("close_promedio", ascending=True).iloc[0][
                "year_quarter"
            ]
        )  # Devuelve el trimestre con menor close promedio.
    if extreme_value == "highest_mean_close":
        return str(
            summary.sort_values("close_promedio", ascending=False).iloc[0][
                "year_quarter"
            ]
        )  # Devuelve el trimestre con mayor close promedio.
    raise ValueError(
        f"Tipo de extremo trimestral no soportado: {extreme_value}"
    )  # Informa filtros extremos no implementados.


def apply_filter_spec(
    df: pd.DataFrame,
    filter_spec: Dict[str, Any],
    target_column: str,
    timestamp_column: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    filter_type = (
        str(filter_spec.get("type", "")).strip().lower()
    )  # Normaliza el tipo de filtro.
    value = filter_spec.get("value")  # Recupera el valor principal del filtro.
    resolved_metadata: Dict[str, Any] = {}  # Acumula metadatos resueltos del filtro.

    if filter_type == "quarter":
        if "year_quarter" not in df.columns:
            raise KeyError(
                "No existe la columna 'year_quarter' requerida para el filtro por trimestre."
            )  # Valida la existencia de la columna trimestral.
        filtered_df = df.loc[
            df["year_quarter"] == value
        ].copy()  # Filtra el trimestre solicitado.
        resolved_metadata["resolved_quarter"] = str(value)  # Guarda el trimestre usado.
    elif filter_type == "regime":
        if "regimen_mercado" not in df.columns:
            raise KeyError(
                "No existe la columna 'regimen_mercado' requerida para el filtro por régimen."
            )  # Valida la existencia de la columna de régimen.
        filtered_df = df.loc[
            df["regimen_mercado"] == value
        ].copy()  # Filtra el régimen solicitado.
        resolved_metadata["resolved_regime"] = str(value)  # Guarda el régimen usado.
    elif filter_type == "extreme_quarter":
        resolved_quarter = resolve_extreme_quarter_label(
            df=df, target_column=target_column, extreme_value=str(value)
        )  # Resuelve el trimestre extremo según close promedio.
        filtered_df = df.loc[
            df["year_quarter"] == resolved_quarter
        ].copy()  # Filtra el trimestre extremo hallado.
        resolved_metadata["resolved_quarter"] = (
            resolved_quarter  # Guarda el trimestre extremo resultante.
        )
        resolved_metadata["resolved_extreme_type"] = str(
            value
        )  # Guarda el tipo de extremo aplicado.
    else:
        raise ValueError(
            f"Tipo de filtro no soportado: {filter_type}"
        )  # Informa filtros no implementados de forma explícita.

    filtered_df = filtered_df.sort_values(timestamp_column).reset_index(
        drop=True
    )  # Ordena cronológicamente el bloque filtrado.
    if len(filtered_df) == 0:
        raise ValueError(
            f"El filtro aplicado no devolvió filas. Filtro: {filter_spec}"
        )  # Evita escenarios vacíos por filtros incompatibles.
    return (
        filtered_df,
        resolved_metadata,
    )  # Devuelve el bloque filtrado y metadatos resueltos.


def build_train_target(
    future_price: float, baseline_price: float, target_mode: str
) -> float:
    validated_mode = validate_target_mode(
        target_mode
    )  # Valida el modo de target antes de construir el valor supervisado.
    if validated_mode == "price":
        return float(future_price)  # Usa el precio futuro directo como target clásico.
    return float(
        future_price - baseline_price
    )  # Usa el delta residual respecto al último cierre visible.


def reconstruct_price_predictions(
    pred_target_raw: np.ndarray, baseline_raw: np.ndarray, target_mode: str
) -> np.ndarray:
    pred_target_flat = np.asarray(pred_target_raw, dtype=np.float64).reshape(
        -1
    )  # Convierte las salidas del modelo a un vector plano en escala real.
    baseline_flat = np.asarray(baseline_raw, dtype=np.float64).reshape(
        -1
    )  # Convierte la persistencia base a un vector plano en escala real.
    validated_mode = validate_target_mode(
        target_mode
    )  # Valida el modo de reconstrucción antes de combinar resultados.
    if validated_mode == "price":
        return pred_target_flat  # Devuelve el precio directo cuando el target fue el precio.
    return (
        baseline_flat + pred_target_flat
    )  # Reconstruye el precio sumando el delta predicho al baseline.



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


def fit_scalers(train_split: WindowedSplit) -> tuple[StandardScaler, StandardScaler]:
    n_features = train_split.x_raw.shape[
        -1
    ]  # Recupera la cantidad de variables predictoras.
    x_scaler = StandardScaler()  # Inicializa el escalador de entradas.
    x_scaler.fit(
        train_split.x_raw.reshape(-1, n_features)
    )  # Ajusta el escalador solo con entrenamiento para evitar fuga.
    y_scaler = StandardScaler()  # Inicializa el escalador del target de entrenamiento.
    y_scaler.fit(
        train_split.train_target_raw.reshape(-1, 1)
    )  # Ajusta el target solo con entrenamiento para evitar fuga.
    return x_scaler, y_scaler  # Devuelve ambos escaladores listos para transformar.


def apply_scalers(
    split: WindowedSplit, x_scaler: StandardScaler, y_scaler: StandardScaler
) -> WindowedSplit:
    n_samples, lookback, n_features = (
        split.x_raw.shape
    )  # Recupera la geometría original del tensor de entrada.
    x_scaled = (
        x_scaler.transform(split.x_raw.reshape(-1, n_features))
        .reshape(n_samples, lookback, n_features)
        .astype(np.float32)
    )  # Escala y recompone las ventanas de entrada.
    train_target_scaled = y_scaler.transform(
        split.train_target_raw.reshape(-1, 1)
    ).astype(
        np.float32
    )  # Escala el objetivo de entrenamiento respetando forma 2D.
    return WindowedSplit(
        x_raw=split.x_raw,
        y_raw=split.y_raw,
        timestamps=split.timestamps,
        baseline_raw=split.baseline_raw,
        train_target_raw=split.train_target_raw,
        target_mode=split.target_mode,
        x_scaled=x_scaled,
        train_target_scaled=train_target_scaled,
    )  # Devuelve el subconjunto original enriquecido con variables escaladas.


def build_dataloader(
    x_scaled: np.ndarray,
    target_scaled: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int = 0,
) -> DataLoader:
    dataset = WindowDataset(
        x_scaled=x_scaled, target_scaled=target_scaled
    )  # Envuelve arreglos numpy en un Dataset compatible con PyTorch.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )  # Construye el DataLoader con pin_memory condicionado a CUDA.


def inverse_transform_targets(
    values: np.ndarray, y_scaler: StandardScaler
) -> np.ndarray:
    values_2d = np.asarray(values, dtype=np.float32).reshape(
        -1, 1
    )  # Recompone cualquier vector a columna 2D para el scaler.
    restored = y_scaler.inverse_transform(values_2d).reshape(
        -1
    )  # Revierte la escala y devuelve un vector plano en unidades reales.
    return restored  # Entrega la salida en la escala original del target entrenado.


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    true_flat = np.asarray(y_true, dtype=np.float64).reshape(
        -1
    )  # Convierte y_true a vector float64 para métricas estables.
    pred_flat = np.asarray(y_pred, dtype=np.float64).reshape(
        -1
    )  # Convierte y_pred a vector float64 para métricas estables.
    if np.isnan(true_flat).any():
        raise ValueError(
            "Las métricas no se pueden calcular porque y_true contiene NaN."
        )  # Informa si el objetivo real llega contaminado.
    if np.isnan(pred_flat).any():
        raise ValueError(
            "Las métricas no se pueden calcular porque y_pred contiene NaN."
        )  # Informa si el modelo produjo predicciones inválidas.
    if np.isinf(true_flat).any() or np.isinf(pred_flat).any():
        raise ValueError(
            "Las métricas no se pueden calcular porque y_true o y_pred contienen valores infinitos."
        )  # Evita métricas corruptas por overflow numérico.
    mse = mean_squared_error(
        true_flat, pred_flat
    )  # Calcula error cuadrático medio en escala real.
    rmse = math.sqrt(mse)  # Calcula raíz del error cuadrático medio en escala real.
    mae = mean_absolute_error(
        true_flat, pred_flat
    )  # Calcula error absoluto medio en escala real.
    denominator = np.clip(
        np.abs(true_flat), EPSILON, None
    )  # Protege el denominador para el cálculo porcentual.
    mape = float(
        np.mean(np.abs((true_flat - pred_flat) / denominator)) * 100.0
    )  # Calcula error porcentual absoluto medio en porcentaje.
    r2 = (
        float(r2_score(true_flat, pred_flat)) if len(true_flat) > 1 else float("nan")
    )  # Calcula R² cuando hay más de una observación.
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAPE": float(mape),
        "R2": float(r2),
    }  # Devuelve el bloque estándar de métricas obligatorias.


def translate_split_label(split_name: str) -> str:
    mapping = {
        "train": "entrenamiento",
        "val": "validacion",
        "validation": "validacion",
        "test": "prueba",
        "baseline_validation": "baseline_validacion",
        "baseline_test": "baseline_prueba",
    }  # Traduce nombres internos de partición a etiquetas limpias para CSV y Tableau.
    return mapping.get(
        split_name, split_name
    )  # Devuelve la traducción si existe o conserva el valor original como respaldo.


def translate_target_mode_label(target_mode: str) -> str:
    validated_mode = validate_target_mode(
        target_mode
    )  # Valida el modo de target antes de traducirlo a una etiqueta de salida.
    if validated_mode == "price":
        return "precio"  # Traduce el modo clásico de precio directo.
    return "delta"  # Traduce el modo residual usado para entrenamiento sobre cambios.


def prepare_scenario(
    scenario: ScenarioDefinition,
    config: ExperimentConfig,
    device: torch.device,
    target_mode: str = "price",
) -> PreparedScenario:
    validated_target_mode = validate_target_mode(
        target_mode
    )  # Valida el modo de target antes de preparar particiones y ventanas.

    active_scenario = deepcopy(
        scenario
    )  # Clona el escenario para poder enriquecer metadatos resueltos sin mutar el catálogo.

    if scenario.mode in {"global", "filtered"}:
        source_path = require_path(
            scenario.source_path, "scenario.source_path"
        )  # Valida que el escenario tenga parquet principal definido.
        source_df = load_parquet_frame(
            source_path, timestamp_column=config.timestamp_column
        )  # Lee el parquet principal del escenario.
        timestamp_column = config.timestamp_column or infer_timestamp_column(
            source_df
        )  # Resuelve la columna temporal efectiva.
        target_column = infer_target_column(
            source_df, forced_target=config.target_column
        )  # Resuelve la columna objetivo efectiva.
        feature_columns = infer_feature_columns(
            source_df,
            target_column=target_column,
            timestamp_column=timestamp_column,
            forced_features=config.feature_columns,
        )  # Resuelve las columnas predictoras efectivas.
        source_df = clean_frame_for_modeling(
            source_df,
            feature_columns=feature_columns,
            target_column=target_column,
            timestamp_column=timestamp_column,
        )  # Elimina filas con NaN antes de cualquier partición temporal.

        if scenario.mode == "global":
            scenario_df = source_df.copy()  # Conserva el dataset global completo.
        else:
            filter_spec = scenario.metadata.get(
                "filter", {}
            )  # Recupera la especificación de filtro del escenario.
            scenario_df, resolved_meta = apply_filter_spec(
                df=source_df,
                filter_spec=filter_spec,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )  # Filtra el bloque solicitado y resuelve metadatos.
            active_scenario.metadata.update(
                resolved_meta
            )  # Conserva el filtro realmente resuelto.

        raw_frames = split_frame_chronologically(
            scenario_df,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )  # Genera train, validation y test cronológicos.

    elif scenario.mode == "cross":
        if scenario.train_path is not None and scenario.test_path is not None:
            train_df = load_parquet_frame(
                scenario.train_path, timestamp_column=config.timestamp_column
            )  # Lee el bloque fuente si el cross usa archivos directos.
            test_df = load_parquet_frame(
                scenario.test_path, timestamp_column=config.timestamp_column
            )  # Lee el bloque destino si el cross usa archivos directos.
            reference_df = pd.concat(
                [train_df.head(5), test_df.head(5)], axis=0, ignore_index=True
            )  # Usa una muestra combinada para detectar columnas compatibles.
            timestamp_column = config.timestamp_column or infer_timestamp_column(
                reference_df
            )  # Resuelve la columna temporal efectiva.
            target_column = infer_target_column(
                reference_df, forced_target=config.target_column
            )  # Resuelve la columna objetivo efectiva.
            feature_columns = infer_feature_columns(
                reference_df,
                target_column=target_column,
                timestamp_column=timestamp_column,
                forced_features=config.feature_columns,
            )  # Resuelve las columnas predictoras efectivas.
            train_df = clean_frame_for_modeling(
                train_df,
                feature_columns=feature_columns,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )  # Limpia el bloque fuente.
            test_df = clean_frame_for_modeling(
                test_df,
                feature_columns=feature_columns,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )  # Limpia el bloque destino.
            raw_frames = split_cross_quarter_frames(
                train_df,
                test_df,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
            )  # Genera train, validation interna y test externo.
        else:
            source_path = require_path(
                scenario.source_path, "scenario.source_path"
            )  # Valida que el escenario cross tenga parquet principal definido.
            source_df = load_parquet_frame(
                source_path, timestamp_column=config.timestamp_column
            )  # Lee el parquet principal del escenario cross.
            timestamp_column = config.timestamp_column or infer_timestamp_column(
                source_df
            )  # Resuelve la columna temporal efectiva.
            target_column = infer_target_column(
                source_df, forced_target=config.target_column
            )  # Resuelve la columna objetivo efectiva.
            feature_columns = infer_feature_columns(
                source_df,
                target_column=target_column,
                timestamp_column=timestamp_column,
                forced_features=config.feature_columns,
            )  # Resuelve las columnas predictoras efectivas.
            source_df = clean_frame_for_modeling(
                source_df,
                feature_columns=feature_columns,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )  # Limpia el dataset global antes de construir bloques cross.

            train_filter = scenario.metadata.get(
                "train_filter", {}
            )  # Recupera el filtro del bloque de entrenamiento.
            test_filter = scenario.metadata.get(
                "test_filter", {}
            )  # Recupera el filtro del bloque de prueba.

            train_df, resolved_train_meta = apply_filter_spec(
                df=source_df,
                filter_spec=train_filter,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )  # Resuelve y aplica el filtro del bloque train.
            test_df, resolved_test_meta = apply_filter_spec(
                df=source_df,
                filter_spec=test_filter,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )  # Resuelve y aplica el filtro del bloque test.

            active_scenario.metadata.update(
                {
                    "resolved_train_filter": resolved_train_meta,
                    "resolved_test_filter": resolved_test_meta,
                }
            )  # Conserva los filtros realmente resueltos del cross.

            raw_frames = split_cross_quarter_frames(
                train_df,
                test_df,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
            )  # Genera train, validation interna y test externo.
    else:
        raise ValueError(
            f"Modo de escenario no soportado: {scenario.mode}"
        )  # Informa explícitamente modos de escenario no implementados.

    windowed_splits = {
        split_name: build_windows(
            df=frame,
            feature_columns=feature_columns,
            target_column=target_column,
            timestamp_column=timestamp_column,
            lookback=config.lookback,
            horizon=config.horizon,
            target_mode=validated_target_mode,
        )
        for split_name, frame in raw_frames.items()
    }  # Convierte cada subconjunto cronológico en ventanas supervisadas.

    x_scaler, y_scaler = fit_scalers(
        windowed_splits["train"]
    )  # Ajusta escaladores solo con el subconjunto de entrenamiento.
    scaled_splits = {
        split_name: apply_scalers(split, x_scaler=x_scaler, y_scaler=y_scaler)
        for split_name, split in windowed_splits.items()
    }  # Escala train, validation y test sin fuga.

    train_x_scaled = require_array(
        scaled_splits["train"].x_scaled, "train.x_scaled"
    )  # Garantiza que las entradas escaladas de entrenamiento no sean None.
    train_target_scaled = require_array(
        scaled_splits["train"].train_target_scaled, "train.train_target_scaled"
    )  # Garantiza que el target escalado de entrenamiento no sea None.
    val_x_scaled = require_array(
        scaled_splits["val"].x_scaled, "val.x_scaled"
    )  # Garantiza que las entradas escaladas de validación no sean None.
    val_target_scaled = require_array(
        scaled_splits["val"].train_target_scaled, "val.train_target_scaled"
    )  # Garantiza que el target escalado de validación no sea None.
    test_x_scaled = require_array(
        scaled_splits["test"].x_scaled, "test.x_scaled"
    )  # Garantiza que las entradas escaladas de prueba no sean None.
    test_target_scaled = require_array(
        scaled_splits["test"].train_target_scaled, "test.train_target_scaled"
    )  # Garantiza que el target escalado de prueba no sea None.

    loaders = {
        "train": build_dataloader(
            train_x_scaled,
            train_target_scaled,
            batch_size=config.batch_size,
            shuffle=True,
            device=device,
            num_workers=config.num_workers,
        ),
        "val": build_dataloader(
            val_x_scaled,
            val_target_scaled,
            batch_size=config.batch_size,
            shuffle=False,
            device=device,
            num_workers=config.num_workers,
        ),
        "test": build_dataloader(
            test_x_scaled,
            test_target_scaled,
            batch_size=config.batch_size,
            shuffle=False,
            device=device,
            num_workers=config.num_workers,
        ),
    }  # Construye DataLoader consistentes para las tres particiones.

    return PreparedScenario(
        scenario=active_scenario,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=feature_columns,
        training_target_mode=validated_target_mode,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        splits=scaled_splits,
        loaders=loaders,
        raw_frames=raw_frames,
    )  # Devuelve todo el paquete del escenario listo para entrenar o evaluar.


@torch.no_grad()
def predict_scaled(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()  # Coloca el modelo en modo evaluación para desactivar dropout.
    predictions: List[np.ndarray] = []  # Acumula predicciones escaladas por batch.
    for batch_x, _ in loader:
        batch_x = batch_x.to(
            device, non_blocking=(device.type == "cuda")
        )  # Mueve el lote de entrada al dispositivo activo.
        batch_pred = model(batch_x)  # Ejecuta inferencia del lote actual.
        if torch.isnan(batch_pred).any() or torch.isinf(batch_pred).any():
            raise ValueError(
                "El modelo produjo NaN o valores infinitos durante predict_scaled."
            )  # Detiene el flujo si la inferencia produce valores inválidos.
        batch_pred_np = (
            batch_pred.detach().cpu().numpy()
        )  # Mueve las predicciones del batch actual al CPU.
        predictions.append(batch_pred_np)  # Almacena las predicciones del batch actual.
    if not predictions:
        return np.empty(
            (0, 1), dtype=np.float32
        )  # Devuelve arreglo vacío si el loader no tiene muestras.
    return np.concatenate(predictions, axis=0).astype(
        np.float32
    )  # Concatena todos los batches en una sola matriz.


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    split: WindowedSplit,
    y_scaler: StandardScaler,
    device: torch.device,
) -> Dict[str, Any]:
    criterion = nn.MSELoss()  # Reutiliza MSE como pérdida base para evaluación interna.
    model.eval()  # Coloca el modelo en modo evaluación para obtener métricas estables.
    total_loss = 0.0  # Acumula la pérdida escalada ponderada por tamaño de batch.
    total_samples = 0  # Acumula la cantidad total de observaciones evaluadas.
    predictions_scaled: List[np.ndarray] = (
        []
    )  # Acumula predicciones escaladas por batch.
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(
            device, non_blocking=(device.type == "cuda")
        )  # Mueve entradas al dispositivo activo.
        batch_y = batch_y.to(
            device, non_blocking=(device.type == "cuda")
        )  # Mueve objetivos al dispositivo activo.
        batch_pred = model(batch_x)  # Ejecuta inferencia sobre el lote actual.
        if torch.isnan(batch_pred).any() or torch.isinf(batch_pred).any():
            raise ValueError(
                "El modelo produjo NaN o valores infinitos durante evaluate_model."
            )  # Informa inmediatamente si la evaluación se volvió inestable.
        batch_loss = criterion(
            batch_pred, batch_y
        )  # Calcula la pérdida MSE en espacio escalado del target entrenado.
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            raise ValueError(
                "La pérdida de evaluación produjo NaN o valores infinitos."
            )  # Informa inmediatamente si la pérdida de evaluación es inválida.
        total_loss += batch_loss.item() * batch_x.size(
            0
        )  # Acumula la pérdida ponderada por muestras del batch.
        total_samples += batch_x.size(
            0
        )  # Acumula la cantidad de muestras ya evaluadas.
        predictions_scaled.append(
            batch_pred.detach().cpu().numpy()
        )  # Envía predicciones al CPU para su consolidación.
    if total_samples == 0:
        raise ValueError(
            "El DataLoader de evaluación no contiene muestras."
        )  # Evita divisiones por cero y evaluaciones vacías.
    pred_scaled = np.concatenate(
        predictions_scaled, axis=0
    )  # Concatena todas las predicciones escaladas.
    pred_target_raw = inverse_transform_targets(
        pred_scaled, y_scaler=y_scaler
    )  # Revierte la escala de las predicciones al espacio del target entrenado.
    pred_price_raw = reconstruct_price_predictions(
        pred_target_raw=pred_target_raw,
        baseline_raw=split.baseline_raw,
        target_mode=split.target_mode,
    )  # Reconstruye el precio final si el modelo fue entrenado sobre delta.
    y_true_price_raw = split.y_raw.reshape(
        -1
    )  # Recupera el precio objetivo real en vector plano.
    metrics = compute_regression_metrics(
        y_true_price_raw, pred_price_raw
    )  # Calcula métricas obligatorias en precio real reconstruido.
    return {
        "scaled_MSE": float(total_loss / total_samples),
        "pred_scaled": pred_scaled.reshape(-1),
        "pred_target_raw": pred_target_raw.reshape(-1),
        "pred_raw": pred_price_raw.reshape(-1),
        "y_true_raw": y_true_price_raw.reshape(-1),
        "metrics": metrics,
    }  # Devuelve pérdida interna y métricas finales en unidades reales de precio.


def save_history_csv(
    history_rows: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    scenario_name: str,
    training_target_mode: str,
) -> Path:
    history_df = pd.DataFrame(
        history_rows
    )  # Convierte la historia por época en un DataFrame tabular.
    history_df.insert(
        0,
        "modo_target_entrenamiento",
        translate_target_mode_label(training_target_mode),
    )  # Inserta el modo de target entrenado para análisis posterior en Tableau.
    history_df.insert(
        0, "escenario", scenario_name
    )  # Inserta el escenario del experimento para facilitar uniones y filtros.
    history_df.insert(
        0, "modelo", model_name
    )  # Inserta el nombre del modelo para facilitar dashboards comparativos.
    history_df.to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )  # Exporta la historia con BOM para abrir bien en Excel y Tableau.
    return output_path  # Devuelve la ruta del archivo generado.


def save_metrics_files(
    metrics_payload: Dict[str, Any], csv_path: Path, json_path: Path
) -> Dict[str, Path]:
    flat_rows: List[Dict[str, Any]] = (
        []
    )  # Inicializa la estructura plana para el CSV de métricas.
    training_target_mode = metrics_payload.get(
        "training_target_mode", "price"
    )  # Recupera el modo de target entrenado para enriquecer la salida tabular.
    for split_name in ["validation", "test", "baseline_validation", "baseline_test"]:
        split_metrics = metrics_payload.get(
            split_name, {}
        )  # Recupera el bloque de métricas correspondiente.
        row = {
            "modelo": metrics_payload.get("model", ""),
            "escenario": metrics_payload.get("scenario", ""),
            "particion": translate_split_label(split_name),
            "mejor_epoca": metrics_payload.get("best_epoch", ""),
            "modo_target_entrenamiento": translate_target_mode_label(
                training_target_mode
            ),
            "MAE": split_metrics.get("MAE"),
            "RMSE": split_metrics.get("RMSE"),
            "MSE": split_metrics.get("MSE"),
            "MAPE": split_metrics.get("MAPE"),
            "r_cuadrado": split_metrics.get("R2"),
        }  # Construye una fila consistente y lista para Tableau.
        flat_rows.append(row)  # Agrega la fila al conjunto exportable.
    pd.DataFrame(flat_rows).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )  # Exporta métricas resumidas a CSV con encabezados finales para Tableau.
    json_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )  # Exporta un reporte completo en JSON legible.
    return {"csv": csv_path, "json": json_path}  # Devuelve ambas rutas exportadas.


def save_predictions_csv(
    scenario_name: str,
    model_name: str,
    split_name: str,
    split: WindowedSplit,
    pred_raw: np.ndarray,
    output_path: Path,
) -> Path:
    y_true = split.y_raw.reshape(
        -1
    )  # Recupera el valor real del objetivo en el subconjunto.
    baseline = split.baseline_raw.reshape(
        -1
    )  # Recupera la persistencia ingenua del subconjunto.
    pred = np.asarray(pred_raw, dtype=np.float64).reshape(
        -1
    )  # Convierte la predicción final del modelo a vector plano.
    output_df = pd.DataFrame(
        {
            "fecha_hora_utc": pd.to_datetime(split.timestamps, utc=True),
            "escenario": scenario_name,
            "modelo": model_name,
            "particion": translate_split_label(split_name),
            "modo_target_entrenamiento": translate_target_mode_label(split.target_mode),
            "valor_real": y_true,
            "valor_predicho": pred,
            "valor_baseline": baseline,
            "residual_modelo": y_true - pred,
            "residual_baseline": y_true - baseline,
            "error_absoluto_modelo": np.abs(y_true - pred),
            "error_absoluto_baseline": np.abs(y_true - baseline),
        }
    )  # Construye el archivo de predicciones listo para Tableau y storytelling.
    output_df.to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )  # Exporta el detalle de predicciones a CSV.
    return output_path  # Devuelve la ruta del archivo generado.


def print_metric_block(title: str, metrics: Dict[str, float]) -> None:
    print(title)  # Imprime el encabezado de la sección métrica.
    print(f"MAE  : {metrics['MAE']:.6f}")  # Imprime el error absoluto medio.
    print(
        f"RMSE : {metrics['RMSE']:.6f}"
    )  # Imprime la raíz del error cuadrático medio.
    print(f"MSE  : {metrics['MSE']:.6f}")  # Imprime el error cuadrático medio.
    print(
        f"MAPE : {metrics['MAPE']:.6f}%"
    )  # Imprime el error porcentual absoluto medio.
    print(f"R2   : {metrics['R2']:.6f}")  # Imprime el coeficiente de determinación.


def build_output_paths(
    project_root: Path, model_subdir: str, model_name: str, scenario_name: str
) -> Dict[str, Path]:
    predictions_dir = ensure_directory(
        project_root / "outputs" / "predictions"
    )  # Asegura la carpeta de predicciones del proyecto.
    metrics_dir = ensure_directory(
        project_root / "outputs" / "metrics"
    )  # Asegura la carpeta de métricas del proyecto.
    history_dir = ensure_directory(
        project_root / "outputs" / "history"
    )  # Asegura la carpeta de historia del proyecto.
    model_dir = ensure_directory(
        project_root / "models" / model_subdir
    )  # Asegura la carpeta de pesos del modelo actual.
    safe_prefix = f"{model_name}__{scenario_name}"  # Estandariza el prefijo de archivos del experimento.
    return {
        "pred_val": predictions_dir / f"{safe_prefix}__validation.csv",
        "pred_test": predictions_dir / f"{safe_prefix}__test.csv",
        "metrics_csv": metrics_dir / f"{safe_prefix}__metrics.csv",
        "metrics_json": metrics_dir / f"{safe_prefix}__metrics.json",
        "history_csv": history_dir / f"{safe_prefix}__history.csv",
        "weights_pt": model_dir / f"{safe_prefix}__best.pt",
    }  # Devuelve el mapa completo de archivos de salida del experimento.


def fit_model(
    model: nn.Module,
    model_name: str,
    prepared: PreparedScenario,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model = model.to(
        device
    )  # Envía el modelo al dispositivo activo antes del entrenamiento.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )  # Usa Adam como optimizador fijo del proyecto.
    criterion = nn.MSELoss()  # Usa MSE como función de pérdida del entrenamiento.
    early_stopping = EarlyStoppingState(
        patience=config.patience
    )  # Inicializa la lógica de early stopping.
    history_rows: List[Dict[str, Any]] = []  # Acumula el historial completo por época.
    output_paths = build_output_paths(
        config.project_root,
        config.model_subdir,
        model_name,
        prepared.scenario.display_name,
    )  # Resuelve todas las rutas de salida del experimento.
    train_loader = prepared.loaders["train"]  # Recupera el DataLoader de entrenamiento.
    val_loader = prepared.loaders["val"]  # Recupera el DataLoader de validación.
    test_loader = prepared.loaders["test"]  # Recupera el DataLoader de prueba.
    val_baseline_metrics = compute_regression_metrics(
        prepared.splits["val"].y_raw.reshape(-1),
        prepared.splits["val"].baseline_raw.reshape(-1),
    )  # Calcula baseline de validación en escala real.
    test_baseline_metrics = compute_regression_metrics(
        prepared.splits["test"].y_raw.reshape(-1),
        prepared.splits["test"].baseline_raw.reshape(-1),
    )  # Calcula baseline de prueba en escala real.

    for epoch in range(1, config.epochs + 1):
        epoch_start = (
            time.perf_counter()
        )  # Marca el inicio de la época para medir tiempo.
        model.train()  # Activa modo entrenamiento para habilitar dropout y gradientes.
        running_loss = 0.0  # Acumula la pérdida ponderada en espacio escalado.
        seen_samples = 0  # Acumula cuántas muestras han sido procesadas en la época.
        seen_batches = 0  # Acumula cuántos batches han sido procesados en la época.
        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch:02d}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )  # Muestra progreso por batch con porcentaje automático y ancho dinámico.

        for batch_x, batch_y in progress:
            batch_x = batch_x.to(
                device, non_blocking=(device.type == "cuda")
            )  # Mueve entradas al dispositivo activo.
            batch_y = batch_y.to(
                device, non_blocking=(device.type == "cuda")
            )  # Mueve objetivos al dispositivo activo.
            optimizer.zero_grad(
                set_to_none=True
            )  # Limpia gradientes acumulados de forma eficiente.
            batch_pred = model(batch_x)  # Ejecuta el forward pass del batch actual.
            if torch.isnan(batch_pred).any() or torch.isinf(batch_pred).any():
                raise ValueError(
                    f"El modelo produjo NaN o valores infinitos durante entrenamiento en la época {epoch:02d}."
                )  # Detiene el flujo si la salida del modelo es inválida.
            batch_loss = criterion(
                batch_pred, batch_y
            )  # Calcula la pérdida MSE del batch actual en el target entrenado.
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                raise ValueError(
                    f"La pérdida produjo NaN o valores infinitos durante entrenamiento en la época {epoch:02d}."
                )  # Detiene el flujo si la pérdida es inválida.
            batch_loss.backward()  # Propaga gradientes hacia atrás.
            if config.max_grad_norm > 0:
                clip_grad_norm_(
                    model.parameters(), max_norm=config.max_grad_norm
                )  # Aplica clipping de gradiente para estabilizar el entrenamiento.
            optimizer.step()  # Actualiza los pesos del modelo con Adam.
            running_loss += batch_loss.item() * batch_x.size(
                0
            )  # Acumula la pérdida ponderada por muestras.
            seen_samples += batch_x.size(
                0
            )  # Acumula la cantidad de muestras procesadas.
            seen_batches += 1  # Acumula la cantidad de batches procesados.
            avg_train = running_loss / max(
                seen_samples, 1
            )  # Calcula el promedio de entrenamiento acumulado.
            elapsed = max(
                time.perf_counter() - epoch_start, EPSILON
            )  # Mide el tiempo transcurrido desde el inicio de la época.
            batch_speed = (
                seen_batches / elapsed
            )  # Calcula la velocidad promedio en batches por segundo.
            progress.set_postfix(
                batch_loss=f"{batch_loss.item():.6f}",
                avg_train=f"{avg_train:.6f}",
                speed=f"{batch_speed:.2f} batch/s",
            )  # Actualiza batch_loss, avg_train y velocidad en consola.

        train_mse = running_loss / max(
            seen_samples, 1
        )  # Calcula la pérdida promedio final de entrenamiento en la época.
        val_eval = evaluate_model(
            model, val_loader, prepared.splits["val"], prepared.y_scaler, device=device
        )  # Evalúa el modelo sobre validation al cierre de la época.
        epoch_time = (
            time.perf_counter() - epoch_start
        )  # Calcula el tiempo total de la época recién terminada.
        samples_per_sec = seen_samples / max(
            epoch_time, EPSILON
        )  # Calcula la velocidad media en muestras por segundo.
        val_metrics = val_eval["metrics"]  # Recupera métricas reales de validation.

        history_row = {
            "epoca": epoch,
            "train_MSE": float(train_mse),
            "val_MSE": float(val_metrics["MSE"]),
            "val_MAE": float(val_metrics["MAE"]),
            "val_RMSE": float(val_metrics["RMSE"]),
            "val_MAPE": float(val_metrics["MAPE"]),
            "val_r_cuadrado": float(val_metrics["R2"]),
            "baseline_val_MAE": float(val_baseline_metrics["MAE"]),
            "baseline_val_RMSE": float(val_baseline_metrics["RMSE"]),
            "muestras_por_segundo": float(samples_per_sec),
            "tiempo_epoca_segundos": float(epoch_time),
        }  # Registra la fila resumen de la época con encabezados finales para Tableau.
        history_rows.append(history_row)  # Agrega la fila actual al historial completo.

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_MSE: {train_mse:.6f} | "
            f"val_MSE: {val_metrics['MSE']:.6f} | "
            f"VAL -> MAE: {val_metrics['MAE']:.6f}, RMSE: {val_metrics['RMSE']:.6f}, MAPE: {val_metrics['MAPE']:.4f}%, R2: {val_metrics['R2']:.6f} | "
            f"BASELINE_VAL -> MAE: {val_baseline_metrics['MAE']:.6f}, RMSE: {val_baseline_metrics['RMSE']:.6f} | "
            f"velocidad: {samples_per_sec:.2f} muestras/s | "
            f"tiempo: {epoch_time:.2f}s"
        )  # Imprime el resumen obligatorio por época con métricas y velocidad.

        improved = early_stopping.update(
            model=model, epoch=epoch, val_mse=val_metrics["MSE"]
        )  # Actualiza el estado del mejor modelo según MSE real de validation.
        if (
            improved
            and config.save_best_weights
            and early_stopping.best_state_dict is not None
        ):
            torch.save(
                early_stopping.best_state_dict, output_paths["weights_pt"]
            )  # Guarda los mejores pesos observados hasta la época actual.
        if early_stopping.should_stop:
            print(
                f"Early stopping activado en la época {epoch:02d}. Mejor época: {early_stopping.best_epoch:02d}"
            )  # Informa el punto exacto de corte anticipado.
            break

    if early_stopping.best_state_dict is None:
        early_stopping.best_state_dict = deepcopy(
            model.state_dict()
        )  # Respalda el estado actual si nunca se registró una mejora explícita.
        early_stopping.best_epoch = len(
            history_rows
        )  # Usa la última época ejecutada como mejor época de respaldo.

    model.load_state_dict(
        early_stopping.best_state_dict
    )  # Restaura en memoria los mejores pesos antes de la evaluación final.
    final_val_eval = evaluate_model(
        model, val_loader, prepared.splits["val"], prepared.y_scaler, device=device
    )  # Evalúa validation con el mejor modelo restaurado.
    final_test_eval = evaluate_model(
        model, test_loader, prepared.splits["test"], prepared.y_scaler, device=device
    )  # Evalúa test con el mejor modelo restaurado.

    save_predictions_csv(
        prepared.scenario.display_name,
        model_name,
        "validation",
        prepared.splits["val"],
        final_val_eval["pred_raw"],
        output_paths["pred_val"],
    )  # Exporta predicciones detalladas de validation.
    save_predictions_csv(
        prepared.scenario.display_name,
        model_name,
        "test",
        prepared.splits["test"],
        final_test_eval["pred_raw"],
        output_paths["pred_test"],
    )  # Exporta predicciones detalladas de test.
    save_history_csv(
        history_rows=history_rows,
        output_path=output_paths["history_csv"],
        model_name=model_name,
        scenario_name=prepared.scenario.display_name,
        training_target_mode=prepared.training_target_mode,
    )  # Exporta el historial completo por época con encabezados finales para Tableau.

    metrics_payload = {
        "model": model_name,
        "scenario": prepared.scenario.display_name,
        "scenario_metadata": prepared.scenario.metadata,
        "target_column": prepared.target_column,
        "feature_columns": prepared.feature_columns,
        "training_target_mode": prepared.training_target_mode,
        "best_epoch": early_stopping.best_epoch,
        "validation": final_val_eval["metrics"],
        "test": final_test_eval["metrics"],
        "baseline_validation": val_baseline_metrics,
        "baseline_test": test_baseline_metrics,
        "config": {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(config).items()
            },
            "device": device.type,
        },
    }  # Consolida el reporte completo de métricas, configuración y trazabilidad.

    save_metrics_files(
        metrics_payload=metrics_payload,
        csv_path=output_paths["metrics_csv"],
        json_path=output_paths["metrics_json"],
    )  # Exporta las métricas finales a CSV y JSON.

    print_metric_block(
        "METRICAS VALIDATION", final_val_eval["metrics"]
    )  # Imprime el bloque obligatorio de métricas de validación.
    print()  # Inserta una línea en blanco para mejorar legibilidad en consola.
    print_metric_block(
        "METRICAS TEST", final_test_eval["metrics"]
    )  # Imprime el bloque obligatorio de métricas de prueba.
    print()  # Inserta una línea en blanco para mejorar legibilidad en consola.
    print_metric_block(
        "BASELINE TEST", test_baseline_metrics
    )  # Imprime el bloque obligatorio del baseline en prueba.

    return {
        "model_name": model_name,
        "scenario_name": prepared.scenario.display_name,
        "training_target_mode": prepared.training_target_mode,
        "best_epoch": early_stopping.best_epoch,
        "validation_metrics": final_val_eval["metrics"],
        "test_metrics": final_test_eval["metrics"],
        "baseline_validation_metrics": val_baseline_metrics,
        "baseline_test_metrics": test_baseline_metrics,
        "paths": {key: str(value) for key, value in output_paths.items()},
    }  # Devuelve un resumen del experimento listo para el orquestador.


def run_training_for_scenario(
    scenario: ScenarioDefinition,
    config: ExperimentConfig,
    model_name: str,
    model_builder: Callable[[int, int], nn.Module],
    train_target_mode: str = "delta",
) -> Dict[str, Any]:
    set_global_seed(
        config.random_seed
    )  # Fija semillas antes de preparar datos y crear el modelo.
    device = (
        get_device()
    )  # Selecciona GPU o CPU y activa cudnn.benchmark cuando corresponda.
    prepared = prepare_scenario(
        scenario=scenario,
        config=config,
        device=device,
        target_mode=train_target_mode,
    )  # Prepara particiones, ventanas, escaladores y loaders del escenario.
    model = model_builder(
        len(prepared.feature_columns), config.lookback
    )  # Construye el modelo con el número real de variables de entrada.
    return fit_model(
        model=model,
        model_name=model_name,
        prepared=prepared,
        config=config,
        device=device,
    )  # Ejecuta el entrenamiento completo y devuelve el resumen final.


def run_baseline_for_split(split: WindowedSplit) -> Dict[str, float]:
    return compute_regression_metrics(
        split.y_raw.reshape(-1), split.baseline_raw.reshape(-1)
    )  # Evalúa persistencia usando el último cierre visible en cada ventana.


def export_baseline_predictions(
    prepared: PreparedScenario,
    model_name: str,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    output_paths = build_output_paths(
        config.project_root,
        config.model_subdir,
        model_name,
        prepared.scenario.display_name,
    )  # Resuelve rutas del baseline con la misma convención de nombres.
    val_metrics = run_baseline_for_split(
        prepared.splits["val"]
    )  # Calcula baseline sobre validation en escala real.
    test_metrics = run_baseline_for_split(
        prepared.splits["test"]
    )  # Calcula baseline sobre test en escala real.

    save_predictions_csv(
        prepared.scenario.display_name,
        model_name,
        "validation",
        prepared.splits["val"],
        prepared.splits["val"].baseline_raw.reshape(-1),
        output_paths["pred_val"],
    )  # Exporta las predicciones baseline de validation.

    save_predictions_csv(
        prepared.scenario.display_name,
        model_name,
        "test",
        prepared.splits["test"],
        prepared.splits["test"].baseline_raw.reshape(-1),
        output_paths["pred_test"],
    )  # Exporta las predicciones baseline de test.

    metrics_payload = {
        "model": model_name,
        "scenario": prepared.scenario.display_name,
        "scenario_metadata": prepared.scenario.metadata,
        "target_column": prepared.target_column,
        "feature_columns": prepared.feature_columns,
        "training_target_mode": prepared.training_target_mode,
        "best_epoch": 0,
        "validation": val_metrics,
        "test": test_metrics,
        "baseline_validation": val_metrics,
        "baseline_test": test_metrics,
        "config": {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(config).items()
            },
            "device": get_device().type,
        },
    }  # Consolida el reporte del baseline con el mismo formato que los modelos entrenables.

    save_metrics_files(
        metrics_payload=metrics_payload,
        csv_path=output_paths["metrics_csv"],
        json_path=output_paths["metrics_json"],
    )  # Exporta el reporte del baseline a CSV y JSON.
    print_metric_block(
        "METRICAS VALIDATION", val_metrics
    )  # Imprime métricas de validation del baseline.
    print()  # Inserta una línea en blanco para legibilidad.
    print_metric_block(
        "METRICAS TEST", test_metrics
    )  # Imprime métricas de test del baseline.
    print()  # Inserta una línea en blanco para legibilidad.
    print_metric_block(
        "BASELINE TEST", test_metrics
    )  # Repite el baseline test para respetar el estándar de salida común.

    return {
        "model_name": model_name,
        "scenario_name": prepared.scenario.display_name,
        "training_target_mode": prepared.training_target_mode,
        "best_epoch": 0,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_validation_metrics": val_metrics,
        "baseline_test_metrics": test_metrics,
        "paths": {key: str(value) for key, value in output_paths.items()},
    }  # Devuelve el resumen final del baseline listo para el orquestador.


def prepare_baseline_scenario(
    scenario: ScenarioDefinition, config: ExperimentConfig
) -> PreparedScenario:
    device = (
        get_device()
    )  # Recupera el dispositivo activo para construir loaders consistentes.
    return prepare_scenario(
        scenario=scenario,
        config=config,
        device=device,
        target_mode="price",
    )  # Reutiliza la misma preparación del pipeline principal en modo baseline.


if __name__ == "__main__":
    catalog = build_default_scenarios(
        PROJECT_ROOT
    )  # Construye el catálogo estándar de escenarios obligatorios.
    print(
        "Escenarios disponibles:"
    )  # Presenta un resumen rápido cuando el archivo se ejecuta directamente.
    for key, scenario in catalog.items():
        print(
            f"- {key}: {scenario.display_name}"
        )  # Enumera la clave y nombre visible de cada escenario.
