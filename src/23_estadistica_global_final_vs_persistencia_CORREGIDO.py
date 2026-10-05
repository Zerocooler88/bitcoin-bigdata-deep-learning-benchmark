from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "multiseed_final_deterministic"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "statistical_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [11, 23, 42, 84, 168]
MODELS = ["lstm", "gru", "cnn1d", "cnn_lstm"]

BASELINE_PATH = (
    FINAL_ROOT
    / "baseline_persistencia"
    / "predictions"
    / "baseline_persistencia__global_2019_2026__test.csv"
)

# Serie horaria 24/7:
# usamos una semana como bloque conservador para preservar dependencia temporal.
BLOCK_LENGTH = 168

# HAC/Newey-West con una semana de rezagos.
HAC_LAG = 168

# Numero de replicas bootstrap.
BOOTSTRAP_REPS = 5000

# Semilla fija del propio procedimiento estadistico.
BOOTSTRAP_SEED = 20261004


# ============================================================
# FUNCIONES BASICAS
# ============================================================

def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def read_prediction_file(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "fecha_hora_utc",
        "valor_real",
        "valor_predicho",
        "valor_baseline",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name}: faltan columnas: {sorted(missing)}"
        )

    df["fecha_hora_utc"] = pd.to_datetime(
        df["fecha_hora_utc"],
        utc=True,
        errors="raise",
    )

    return (
        df.sort_values(
            "fecha_hora_utc"
        )
        .reset_index(drop=True)
    )


# ============================================================
# AUDITORIA DE ALINEACION
# ============================================================

def assert_aligned(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    label: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(
            f"{label}: distinto numero de filas "
            f"({len(candidate)} vs {len(reference)})."
        )

    if not reference[
        "fecha_hora_utc"
    ].equals(
        candidate[
            "fecha_hora_utc"
        ]
    ):
        raise ValueError(
            f"{label}: los timestamps no coinciden."
        )

    if not np.allclose(
        reference[
            "valor_real"
        ].to_numpy(float),
        candidate[
            "valor_real"
        ].to_numpy(float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"{label}: valor_real no coincide."
        )

    if not np.allclose(
        reference[
            "valor_baseline"
        ].to_numpy(float),
        candidate[
            "valor_baseline"
        ].to_numpy(float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"{label}: valor_baseline no coincide."
        )


# ============================================================
# DIEBOLD-MARIANO CON HAC / NEWEY-WEST
# ============================================================

def newey_west_long_run_variance(
    d: np.ndarray,
    max_lag: int,
) -> float:
    """
    Estima la varianza de largo plazo del diferencial de perdida
    usando pesos Bartlett (HAC/Newey-West).
    """

    d = np.asarray(
        d,
        dtype=float,
    )

    n = len(d)

    centered = (
        d - d.mean()
    )

    gamma0 = float(
        np.dot(
            centered,
            centered,
        )
        / n
    )

    lrv = gamma0

    max_lag = min(
        max_lag,
        n - 1,
    )

    for lag in range(
        1,
        max_lag + 1,
    ):
        weight = (
            1.0
            - lag
            / (max_lag + 1.0)
        )

        gamma = float(
            np.dot(
                centered[lag:],
                centered[:-lag],
            )
            / n
        )

        lrv += (
            2.0
            * weight
            * gamma
        )

    return max(
        lrv,
        1e-15,
    )


def dm_test_average_over_seeds(
    y_true: np.ndarray,
    prediction_matrix: np.ndarray,
    baseline_pred: np.ndarray,
    hac_lag: int,
) -> tuple[float, float, float]:
    """
    DM sobre el diferencial de perdida cuadratica por timestamp.

    Primero promedia la perdida de las cinco seeds del modelo
    en cada timestamp y luego la compara con persistence.

    d_t = loss_model_promedio(t) - loss_persistence(t)

    d_t < 0 favorece al modelo.
    d_t > 0 favorece a persistence.
    """

    model_loss = np.mean(
        (
            prediction_matrix
            - y_true[None, :]
        ) ** 2,
        axis=0,
    )

    baseline_loss = (
        baseline_pred
        - y_true
    ) ** 2

    d = (
        model_loss
        - baseline_loss
    )

    mean_d = float(
        np.mean(d)
    )

    long_run_variance = (
        newey_west_long_run_variance(
            d,
            max_lag=hac_lag,
        )
    )

    standard_error = math.sqrt(
        long_run_variance
        / len(d)
    )

    statistic = (
        mean_d
        / standard_error
    )

    # p bilateral con aproximacion normal.
    p_value = math.erfc(
        abs(statistic)
        / math.sqrt(2.0)
    )

    return (
        float(statistic),
        float(p_value),
        mean_d,
    )


# ============================================================
# BLOCK BOOTSTRAP JERARQUICO
# ============================================================

def circular_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Bootstrap circular por bloques.

    Selecciona inicios aleatorios y conserva bloques temporales
    consecutivos de block_length observaciones.
    """

    n_blocks = math.ceil(
        n / block_length
    )

    starts = rng.integers(
        low=0,
        high=n,
        size=n_blocks,
    )

    relative = np.arange(
        block_length,
        dtype=int,
    )

    blocks = []

    for start in starts:
        block = (
            start
            + relative
        ) % n

        blocks.append(block)

    return (
        np.concatenate(
            blocks
        )[:n]
    )


def hierarchical_block_bootstrap(
    y_true: np.ndarray,
    prediction_matrix: np.ndarray,
    baseline_pred: np.ndarray,
    reps: int,
    block_length: int,
    random_seed: int,
) -> dict:
    """
    Bootstrap jerarquico:

    1. remuestrea las cinco seeds con reemplazo;
    2. remuestrea bloques temporales contiguos;
    3. calcula el promedio del RMSE/MAE de las seeds;
    4. resta el RMSE/MAE de persistence.

    Esto incorpora:
    - variabilidad por inicializacion;
    - dependencia temporal.
    """

    rng = np.random.default_rng(
        random_seed
    )

    n_seeds, n_time = (
        prediction_matrix.shape
    )

    delta_rmse = np.empty(
        reps,
        dtype=float,
    )

    delta_mae = np.empty(
        reps,
        dtype=float,
    )

    for b in range(reps):

        sampled_seed_idx = (
            rng.integers(
                low=0,
                high=n_seeds,
                size=n_seeds,
            )
        )

        sampled_time_idx = (
            circular_block_indices(
                n=n_time,
                block_length=block_length,
                rng=rng,
            )
        )

        y_b = y_true[
            sampled_time_idx
        ]

        baseline_b = baseline_pred[
            sampled_time_idx
        ]

        pred_b = (
            prediction_matrix[
                sampled_seed_idx
            ][:, sampled_time_idx]
        )

        model_rmse_per_seed = (
            np.sqrt(
                np.mean(
                    (
                        pred_b
                        - y_b[None, :]
                    ) ** 2,
                    axis=1,
                )
            )
        )

        model_mae_per_seed = (
            np.mean(
                np.abs(
                    pred_b
                    - y_b[None, :]
                ),
                axis=1,
            )
        )

        baseline_rmse = rmse(
            y_b,
            baseline_b,
        )

        baseline_mae = mae(
            y_b,
            baseline_b,
        )

        delta_rmse[b] = (
            float(
                np.mean(
                    model_rmse_per_seed
                )
            )
            - baseline_rmse
        )

        delta_mae[b] = (
            float(
                np.mean(
                    model_mae_per_seed
                )
            )
            - baseline_mae
        )

    return {
        "delta_RMSE_ci_low": float(
            np.percentile(
                delta_rmse,
                2.5,
            )
        ),
        "delta_RMSE_ci_high": float(
            np.percentile(
                delta_rmse,
                97.5,
            )
        ),
        "delta_MAE_ci_low": float(
            np.percentile(
                delta_mae,
                2.5,
            )
        ),
        "delta_MAE_ci_high": float(
            np.percentile(
                delta_mae,
                97.5,
            )
        ),
    }


# ============================================================
# CORRECCION HOLM
# ============================================================

def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(p)

    adjusted = np.empty_like(p)

    running_max = 0.0
    m = len(p)

    for rank, idx in enumerate(
        order
    ):
        candidate = (
            (m - rank)
            * p[idx]
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[idx] = min(
            running_max,
            1.0,
        )

    return adjusted


# ============================================================
# CARGA DE LAS CINCO SEEDS
# ============================================================

def load_model_prediction_matrix(
    model: str,
    reference: pd.DataFrame,
) -> tuple[
    np.ndarray,
    list[float],
    list[float],
]:

    y_true = reference[
        "valor_real"
    ].to_numpy(float)

    predictions = []
    rmses = []
    maes = []

    for seed in SEEDS:

        path = (
            FINAL_ROOT
            / f"seed_{seed}"
            / "predictions"
            / f"{model}__global_2019_2026__test.csv"
        )

        df = read_prediction_file(
            path
        )

        assert_aligned(
            reference,
            df,
            label=(
                f"{model} seed={seed}"
            ),
        )

        pred = df[
            "valor_predicho"
        ].to_numpy(float)

        predictions.append(pred)

        rmses.append(
            rmse(
                y_true,
                pred,
            )
        )

        maes.append(
            mae(
                y_true,
                pred,
            )
        )

    return (
        np.vstack(
            predictions
        ),
        rmses,
        maes,
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:

    print("=" * 100)
    print("ANALISIS ESTADISTICO FINAL Q1")
    print("DIEBOLD-MARIANO + HIERARCHICAL BLOCK BOOTSTRAP")
    print("=" * 100)

    reference = (
        read_prediction_file(
            BASELINE_PATH
        )
    )

    y_true = reference[
        "valor_real"
    ].to_numpy(float)

    # En el CSV de persistence, ambas columnas representan la misma
    # predicción conceptual. Sin embargo, save_predictions_csv convierte
    # pred_raw a float64 mientras valor_baseline conserva el float32 original.
    # Al escribir/leer CSV pueden aparecer diferencias mínimas de serialización.
    baseline_pred_exported = reference[
        "valor_predicho"
    ].to_numpy(float)

    baseline_column = reference[
        "valor_baseline"
    ].to_numpy(float)

    baseline_serialization_diff = np.abs(
        baseline_pred_exported
        - baseline_column
    )

    max_baseline_serialization_diff = float(
        np.max(
            baseline_serialization_diff
        )
    )

    print(
        "Max |valor_predicho - valor_baseline| "
        f"en persistence: {max_baseline_serialization_diff:.12f} USDT"
    )

    # 0.01 USDT es una tolerancia de auditoría muy conservadora:
    # equivale a un centavo de USDT y solo se usa para comprobar que
    # ambas columnas representan la misma persistencia.
    if max_baseline_serialization_diff > 0.01:
        raise ValueError(
            "La diferencia entre valor_predicho y valor_baseline "
            "en persistence supera 0.01 USDT. "
            "Esto ya no parece una simple diferencia de serializacion."
        )

    # Fuente autoritativa del baseline:
    # usar SIEMPRE valor_baseline, que es la columna explícitamente
    # almacenada como persistencia para todos los modelos.
    baseline_pred = baseline_column

    baseline_rmse = rmse(
        y_true,
        baseline_pred,
    )

    baseline_mae = mae(
        y_true,
        baseline_pred,
    )

    print(
        f"N test timestamps : {len(y_true):,}"
    )
    print(
        f"Persistence RMSE  : {baseline_rmse:.6f}"
    )
    print(
        f"Persistence MAE   : {baseline_mae:.6f}"
    )
    print(
        f"Seeds             : {SEEDS}"
    )
    print(
        f"Block length      : {BLOCK_LENGTH} horas"
    )
    print(
        f"HAC lag           : {HAC_LAG}"
    )
    print(
        f"Bootstrap reps    : {BOOTSTRAP_REPS:,}"
    )
    print()

    rows = []

    for model_index, model in enumerate(
        MODELS
    ):

        (
            prediction_matrix,
            seed_rmses,
            seed_maes,
        ) = (
            load_model_prediction_matrix(
                model,
                reference,
            )
        )

        mean_rmse = float(
            np.mean(
                seed_rmses
            )
        )

        sd_rmse = float(
            np.std(
                seed_rmses,
                ddof=1,
            )
        )

        mean_mae = float(
            np.mean(
                seed_maes
            )
        )

        sd_mae = float(
            np.std(
                seed_maes,
                ddof=1,
            )
        )

        delta_rmse = (
            mean_rmse
            - baseline_rmse
        )

        delta_mae = (
            mean_mae
            - baseline_mae
        )

        (
            dm_stat,
            dm_p,
            dm_mean_loss_diff,
        ) = (
            dm_test_average_over_seeds(
                y_true=y_true,
                prediction_matrix=prediction_matrix,
                baseline_pred=baseline_pred,
                hac_lag=HAC_LAG,
            )
        )

        ci = (
            hierarchical_block_bootstrap(
                y_true=y_true,
                prediction_matrix=prediction_matrix,
                baseline_pred=baseline_pred,
                reps=BOOTSTRAP_REPS,
                block_length=BLOCK_LENGTH,
                random_seed=(
                    BOOTSTRAP_SEED
                    + model_index
                ),
            )
        )

        rows.append(
            {
                "model": model,
                "n_seeds": len(
                    seed_rmses
                ),
                "n_timestamps": len(
                    y_true
                ),
                "baseline_RMSE": baseline_rmse,
                "mean_model_RMSE": mean_rmse,
                "sd_model_RMSE": sd_rmse,
                "delta_RMSE": delta_rmse,
                "delta_RMSE_ci_low": ci[
                    "delta_RMSE_ci_low"
                ],
                "delta_RMSE_ci_high": ci[
                    "delta_RMSE_ci_high"
                ],
                "baseline_MAE": baseline_mae,
                "mean_model_MAE": mean_mae,
                "sd_model_MAE": sd_mae,
                "delta_MAE": delta_mae,
                "delta_MAE_ci_low": ci[
                    "delta_MAE_ci_low"
                ],
                "delta_MAE_ci_high": ci[
                    "delta_MAE_ci_high"
                ],
                "dm_mean_squared_loss_difference": (
                    dm_mean_loss_diff
                ),
                "dm_statistic": dm_stat,
                "dm_p_value": dm_p,
            }
        )

    results = pd.DataFrame(
        rows
    )

    results[
        "dm_p_holm"
    ] = holm_adjust(
        results[
            "dm_p_value"
        ].to_numpy()
    )

    results[
        "dm_significant_0_05"
    ] = (
        results[
            "dm_p_holm"
        ]
        < 0.05
    )

    results[
        "rmse_ci_excludes_zero"
    ] = (
        (
            results[
                "delta_RMSE_ci_low"
            ]
            > 0
        )
        |
        (
            results[
                "delta_RMSE_ci_high"
            ]
            < 0
        )
    )

    results[
        "mae_ci_excludes_zero"
    ] = (
        (
            results[
                "delta_MAE_ci_low"
            ]
            > 0
        )
        |
        (
            results[
                "delta_MAE_ci_high"
            ]
            < 0
        )
    )

    results[
        "rmse_direction"
    ] = np.where(
        results[
            "delta_RMSE"
        ]
        < 0,
        "model_better",
        "persistence_better",
    )

    output_path = (
        OUTPUT_DIR
        / "global_final_dm_hierarchical_bootstrap.csv"
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 100)
    print("RESULTADOS FINALES")
    print("=" * 100)

    print(
        results[
            [
                "model",
                "mean_model_RMSE",
                "sd_model_RMSE",
                "delta_RMSE",
                "delta_RMSE_ci_low",
                "delta_RMSE_ci_high",
                "dm_statistic",
                "dm_p_value",
                "dm_p_holm",
                "dm_significant_0_05",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("INTERPRETACION")
    print("=" * 100)

    for _, row in results.iterrows():

        if row[
            "delta_RMSE"
        ] < 0:
            direction = (
                "RMSE medio favorece al modelo"
            )
        else:
            direction = (
                "RMSE medio favorece a persistence"
            )

        ci_contains_zero = (
            row[
                "delta_RMSE_ci_low"
            ]
            <= 0
            <= row[
                "delta_RMSE_ci_high"
            ]
        )

        if ci_contains_zero:
            ci_text = (
                "IC95% de delta RMSE incluye 0"
            )
        else:
            ci_text = (
                "IC95% de delta RMSE excluye 0"
            )

        if row[
            "dm_significant_0_05"
        ]:
            dm_text = (
                "DM significativo tras Holm"
            )
        else:
            dm_text = (
                "DM no significativo tras Holm"
            )

        print(
            f"{row['model']}: "
            f"delta RMSE={row['delta_RMSE']:+.6f} USDT; "
            f"IC95%=["
            f"{row['delta_RMSE_ci_low']:+.6f}, "
            f"{row['delta_RMSE_ci_high']:+.6f}]; "
            f"{direction}; "
            f"{ci_text}; "
            f"{dm_text}."
        )

    print()
    print("NOTA:")
    print(
        "Una diferencia estadisticamente detectable no implica relevancia practica."
    )
    print(
        "La magnitud del delta RMSE debe interpretarse junto con el IC95% y la variabilidad entre seeds."
    )

    print()
    print(
        f"Archivo generado: {output_path}"
    )


if __name__ == "__main__":
    main()
