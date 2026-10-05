from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


# ============================================================
# RUTAS Y CONFIGURACION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "multiseed_final_deterministic"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "performance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMMON_PIPELINE_PATH = (
    SRC_DIR
    / "10_common_training_pipeline.py"
)

MODEL_SCRIPTS = {
    "lstm": SRC_DIR / "12_entrenar_lstm.py",
    "gru": SRC_DIR / "13_entrenar_gru.py",
    "cnn1d": SRC_DIR / "14_entrenar_cnn1d.py",
    "cnn_lstm": SRC_DIR / "15_entrenar_cnn_lstm.py",
}

SEEDS = [11, 23, 42, 84, 168]

LOOKBACK = 48
HORIZON = 1
BATCH_SIZE = 64

# La medicion de inferencia se repite varias veces para estabilizar el valor.
INFERENCE_REPEATS = 10

# La primera pasada se usa como warm-up y NO se reporta.
WARMUP_PASSES = 1


# ============================================================
# CARGA DINAMICA DE MODULOS
# ============================================================

def load_module_from_path(
    module_name: str,
    path: Path,
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"No fue posible cargar:\n{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# ENTRENAMIENTO: RESUMEN A PARTIR DE HISTORY.CSV
# ============================================================

def read_training_history(
    model_name: str,
    seed: int,
) -> dict:
    history_path = (
        FINAL_ROOT
        / f"seed_{seed}"
        / "history"
        / f"{model_name}__global_2019_2026__history.csv"
    )

    metrics_path = (
        FINAL_ROOT
        / f"seed_{seed}"
        / "metrics"
        / f"{model_name}__global_2019_2026__metrics.json"
    )

    if not history_path.exists():
        raise FileNotFoundError(
            f"No existe:\n{history_path}"
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No existe:\n{metrics_path}"
        )

    history = pd.read_csv(
        history_path
    )

    required_columns = {
        "tiempo_epoca_segundos",
        "muestras_por_segundo",
    }

    missing = (
        required_columns
        - set(history.columns)
    )

    if missing:
        raise ValueError(
            f"{history_path.name}: faltan columnas {sorted(missing)}"
        )

    metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    # IMPORTANTE:
    # tiempo_epoca_segundos se mide desde el inicio de la epoca hasta despues
    # de evaluar validation. Por eso representa tiempo registrado por epoca
    # de entrenamiento + validacion, no un wall-clock completo del proceso.
    recorded_run_seconds = float(
        history[
            "tiempo_epoca_segundos"
        ].sum()
    )

    return {
        "model": model_name,
        "seed": seed,
        "epochs_executed": int(
            len(history)
        ),
        "best_epoch": int(
            metrics.get(
                "best_epoch",
                0,
            )
        ),
        "recorded_run_seconds": (
            recorded_run_seconds
        ),
        "mean_epoch_seconds": float(
            history[
                "tiempo_epoca_segundos"
            ].mean()
        ),
        "mean_training_samples_per_second": float(
            history[
                "muestras_por_segundo"
            ].mean()
        ),
        "test_RMSE": float(
            metrics[
                "test"
            ][
                "RMSE"
            ]
        ),
        "test_MAE": float(
            metrics[
                "test"
            ][
                "MAE"
            ]
        ),
    }


# ============================================================
# INFERENCIA
# ============================================================

@torch.inference_mode()
def one_inference_pass(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> int:
    total_samples = 0

    for batch_x, _ in loader:
        batch_x = batch_x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        _ = model(
            batch_x
        )

        total_samples += int(
            batch_x.size(0)
        )

    return total_samples


def benchmark_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    # Warm-up
    for _ in range(
        WARMUP_PASSES
    ):
        _ = one_inference_pass(
            model,
            loader,
            device,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )
        torch.cuda.reset_peak_memory_stats(
            device
        )

    elapsed_seconds = []
    n_samples = None

    for _ in range(
        INFERENCE_REPEATS
    ):
        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        start = time.perf_counter()

        current_samples = (
            one_inference_pass(
                model,
                loader,
                device,
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        elapsed_seconds.append(
            elapsed
        )

        if n_samples is None:
            n_samples = (
                current_samples
            )
        elif n_samples != current_samples:
            raise RuntimeError(
                "La cantidad de muestras cambio entre repeticiones."
            )

    times = np.asarray(
        elapsed_seconds,
        dtype=float,
    )

    mean_seconds = float(
        times.mean()
    )

    sd_seconds = float(
        times.std(
            ddof=1
        )
    )

    median_seconds = float(
        np.median(
            times
        )
    )

    samples_per_second = float(
        n_samples
        / mean_seconds
    )

    ms_per_sample = float(
        1000.0
        * mean_seconds
        / n_samples
    )

    if device.type == "cuda":
        peak_allocated_mb = float(
            torch.cuda.max_memory_allocated(
                device
            )
            / (
                1024 ** 2
            )
        )

        peak_reserved_mb = float(
            torch.cuda.max_memory_reserved(
                device
            )
            / (
                1024 ** 2
            )
        )
    else:
        peak_allocated_mb = float(
            "nan"
        )
        peak_reserved_mb = float(
            "nan"
        )

    return {
        "inference_repeats": (
            INFERENCE_REPEATS
        ),
        "test_samples": int(
            n_samples
        ),
        "mean_inference_seconds_full_test": (
            mean_seconds
        ),
        "sd_inference_seconds_full_test": (
            sd_seconds
        ),
        "median_inference_seconds_full_test": (
            median_seconds
        ),
        "inference_samples_per_second": (
            samples_per_second
        ),
        "inference_ms_per_sample": (
            ms_per_sample
        ),
        "peak_gpu_memory_allocated_mb_inference": (
            peak_allocated_mb
        ),
        "peak_gpu_memory_reserved_mb_inference": (
            peak_reserved_mb
        ),
    }


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    print(
        "=" * 100
    )
    print(
        "PERFIL COMPUTACIONAL FINAL Q1"
    )
    print(
        "Training histories + parameter count + inference latency"
    )
    print(
        "=" * 100
    )

    if not FINAL_ROOT.exists():
        raise FileNotFoundError(
            f"No existe la generacion final:\n{FINAL_ROOT}"
        )

    # --------------------------------------------------------
    # 1. Cargar pipeline comun
    # --------------------------------------------------------

    common = load_module_from_path(
        "common_pipeline_efficiency_q1",
        COMMON_PIPELINE_PATH,
    )

    common.set_global_seed(
        42
    )

    device = (
        common.get_device()
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU: "
            + torch.cuda.get_device_name(
                device
            )
        )

    print(
        f"PyTorch: {torch.__version__}"
    )
    print(
        f"CUDA PyTorch: {torch.version.cuda}"
    )
    print()

    # --------------------------------------------------------
    # 2. Preparar SOLO una vez el test global
    # --------------------------------------------------------

    config = common.ExperimentConfig(
        project_root=PROJECT_ROOT,
        lookback=LOOKBACK,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        epochs=50,
        patience=5,
        learning_rate=0.001,
        num_workers=0,
        random_seed=42,
        model_subdir="efficiency_profile",
    )

    scenario = (
        common.get_scenarios_from_keys(
            ["global"],
            project_root=PROJECT_ROOT,
        )[0]
    )

    prepared = (
        common.prepare_scenario(
            scenario=scenario,
            config=config,
            device=device,
            target_mode="delta",
        )
    )

    if len(
        prepared.feature_columns
    ) != 22:
        raise ValueError(
            "La prueba de eficiencia no esta usando 22 features."
        )

    test_loader = (
        prepared.loaders[
            "test"
        ]
    )

    print(
        f"Features: {len(prepared.feature_columns)}"
    )
    print(
        f"Test samples: {len(prepared.splits['test'].y_raw):,}"
    )
    print()

    # --------------------------------------------------------
    # 3. Resumen de training en las cinco seeds
    # --------------------------------------------------------

    training_rows = []

    for model_name in MODEL_SCRIPTS:
        for seed in SEEDS:
            training_rows.append(
                read_training_history(
                    model_name=model_name,
                    seed=seed,
                )
            )

    training_detail = pd.DataFrame(
        training_rows
    )

    training_summary = (
        training_detail.groupby(
            "model",
            as_index=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "count",
            ),
            mean_epochs_executed=(
                "epochs_executed",
                "mean",
            ),
            sd_epochs_executed=(
                "epochs_executed",
                "std",
            ),
            mean_best_epoch=(
                "best_epoch",
                "mean",
            ),
            sd_best_epoch=(
                "best_epoch",
                "std",
            ),
            mean_recorded_run_seconds=(
                "recorded_run_seconds",
                "mean",
            ),
            sd_recorded_run_seconds=(
                "recorded_run_seconds",
                "std",
            ),
            min_recorded_run_seconds=(
                "recorded_run_seconds",
                "min",
            ),
            max_recorded_run_seconds=(
                "recorded_run_seconds",
                "max",
            ),
            mean_epoch_seconds=(
                "mean_epoch_seconds",
                "mean",
            ),
            sd_epoch_seconds=(
                "mean_epoch_seconds",
                "std",
            ),
            mean_training_samples_per_second=(
                "mean_training_samples_per_second",
                "mean",
            ),
            sd_training_samples_per_second=(
                "mean_training_samples_per_second",
                "std",
            ),
            mean_test_RMSE=(
                "test_RMSE",
                "mean",
            ),
            sd_test_RMSE=(
                "test_RMSE",
                "std",
            ),
        )
    )

    # --------------------------------------------------------
    # 4. Parametros + inference usando seed 42
    # --------------------------------------------------------

    inference_rows = []

    for model_name, script_path in (
        MODEL_SCRIPTS.items()
    ):
        print(
            "-" * 100
        )
        print(
            f"Profiling: {model_name}"
        )

        module = load_module_from_path(
            f"efficiency_{model_name}",
            script_path,
        )

        model = module.model_builder(
            input_size=22,
            lookback=LOOKBACK,
        )

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        parameter_bytes = sum(
            parameter.numel()
            * parameter.element_size()
            for parameter in model.parameters()
        )

        parameter_memory_mb = float(
            parameter_bytes
            / (
                1024 ** 2
            )
        )

        weights_path = (
            FINAL_ROOT
            / "seed_42"
            / "models"
            / model_name
            / f"{model_name}__global_2019_2026__best.pt"
        )

        if not weights_path.exists():
            raise FileNotFoundError(
                f"No existen los pesos seed42:\n{weights_path}"
            )

        state_dict = torch.load(
            weights_path,
            map_location=device,
        )

        model.load_state_dict(
            state_dict
        )

        model = model.to(
            device
        )

        profile = (
            benchmark_inference(
                model=model,
                loader=test_loader,
                device=device,
            )
        )

        inference_rows.append(
            {
                "model": model_name,
                "profile_seed": 42,
                "total_parameters": int(
                    total_parameters
                ),
                "trainable_parameters": int(
                    trainable_parameters
                ),
                "parameter_memory_mb_float32": (
                    parameter_memory_mb
                ),
                **profile,
            }
        )

        print(
            f"Parametros: {total_parameters:,}"
        )
        print(
            f"Inferencia: {profile['inference_ms_per_sample']:.6f} ms/sample"
        )
        print(
            f"Throughput: {profile['inference_samples_per_second']:.2f} samples/s"
        )

        if device.type == "cuda":
            print(
                "Peak GPU allocated: "
                f"{profile['peak_gpu_memory_allocated_mb_inference']:.2f} MB"
            )

        # Liberar memoria antes del siguiente modelo.
        del model
        del state_dict

        if device.type == "cuda":
            torch.cuda.empty_cache()

    inference_summary = pd.DataFrame(
        inference_rows
    )

    # --------------------------------------------------------
    # 5. Unir para tabla final
    # --------------------------------------------------------

    final_table = (
        training_summary.merge(
            inference_summary,
            on="model",
            how="inner",
        )
        .sort_values(
            "mean_test_RMSE"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 6. Guardar
    # --------------------------------------------------------

    training_detail_path = (
        OUTPUT_DIR
        / "training_efficiency_multiseed_detail.csv"
    )

    training_summary_path = (
        OUTPUT_DIR
        / "training_efficiency_multiseed_summary.csv"
    )

    inference_path = (
        OUTPUT_DIR
        / "inference_efficiency_seed42.csv"
    )

    final_path = (
        OUTPUT_DIR
        / "model_computational_efficiency_final.csv"
    )

    training_detail.to_csv(
        training_detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    training_summary.to_csv(
        training_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    inference_summary.to_csv(
        inference_path,
        index=False,
        encoding="utf-8-sig",
    )

    final_table.to_csv(
        final_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 100
    )
    print(
        "TABLA COMPUTACIONAL FINAL"
    )
    print(
        "=" * 100
    )

    display_columns = [
        "model",
        "total_parameters",
        "mean_epochs_executed",
        "mean_recorded_run_seconds",
        "sd_recorded_run_seconds",
        "mean_epoch_seconds",
        "mean_training_samples_per_second",
        "inference_ms_per_sample",
        "inference_samples_per_second",
        "peak_gpu_memory_allocated_mb_inference",
        "mean_test_RMSE",
        "sd_test_RMSE",
    ]

    print(
        final_table[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "NOTAS:"
    )
    print(
        "1. mean_recorded_run_seconds = suma de tiempo_epoca_segundos."
    )
    print(
        "2. Cada tiempo de epoca incluye entrenamiento + evaluacion de validation de esa epoca."
    )
    print(
        "3. No incluye carga inicial del parquet, preparacion de ventanas ni evaluacion final posterior al early stopping."
    )
    print(
        "4. La inferencia se mide sobre TODO el test global, con 1 warm-up y "
        f"{INFERENCE_REPEATS} repeticiones."
    )
    print(
        "5. peak_gpu_memory_allocated_mb_inference corresponde a inferencia, NO a memoria maxima de entrenamiento."
    )
    print()

    print(
        "Archivos generados:"
    )
    print(
        training_detail_path
    )
    print(
        training_summary_path
    )
    print(
        inference_path
    )
    print(
        final_path
    )


if __name__ == "__main__":
    main()
