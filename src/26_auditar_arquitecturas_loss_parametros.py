from __future__ import annotations

from pathlib import Path
import importlib.util
import inspect
import json
import sys

import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_PIPELINE = SRC_DIR / "10_common_training_pipeline.py"

MODEL_SCRIPTS = {
    "LSTM": SRC_DIR / "12_entrenar_lstm.py",
    "GRU": SRC_DIR / "13_entrenar_gru.py",
    "CNN-1D": SRC_DIR / "14_entrenar_cnn1d.py",
    "CNN-LSTM": SRC_DIR / "15_entrenar_cnn_lstm.py",
}

INPUT_SIZE = 22
LOOKBACK = 48


# ============================================================
# UTILIDADES
# ============================================================

def load_module(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe:\n{path}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar:\n{path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return int(total), int(trainable)


def module_inventory(model: nn.Module) -> list[dict]:
    rows = []

    for name, module in model.named_modules():
        if name == "":
            continue

        row = {
            "name": name,
            "type": module.__class__.__name__,
        }

        if isinstance(module, nn.LSTM):
            row.update(
                {
                    "input_size": module.input_size,
                    "hidden_size": module.hidden_size,
                    "num_layers": module.num_layers,
                    "dropout": module.dropout,
                    "bidirectional": module.bidirectional,
                }
            )

        elif isinstance(module, nn.GRU):
            row.update(
                {
                    "input_size": module.input_size,
                    "hidden_size": module.hidden_size,
                    "num_layers": module.num_layers,
                    "dropout": module.dropout,
                    "bidirectional": module.bidirectional,
                }
            )

        elif isinstance(module, nn.Conv1d):
            row.update(
                {
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size[0],
                    "stride": module.stride[0],
                    "padding": module.padding[0],
                }
            )

        elif isinstance(module, nn.Linear):
            row.update(
                {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
            )

        elif isinstance(module, nn.Dropout):
            row.update(
                {
                    "dropout": module.p,
                }
            )

        elif isinstance(module, nn.AdaptiveAvgPool1d):
            row.update(
                {
                    "output_size": module.output_size,
                }
            )

        rows.append(row)

    return rows


def detect_training_choices(common_module) -> dict:
    """
    Audita la función fit_model del pipeline común.
    No infiere desde comentarios: inspecciona el código fuente real cargado.
    """
    source = inspect.getsource(common_module.fit_model)

    loss_detected = (
        "nn.MSELoss()" in source
        or "torch.nn.MSELoss()" in source
    )

    adam_detected = (
        "torch.optim.Adam(" in source
        or "optim.Adam(" in source
    )

    grad_clip_detected = (
        "clip_grad_norm_" in source
    )

    return {
        "loss_function": (
            "MSELoss"
            if loss_detected
            else "NOT_AUTOMATICALLY_VERIFIED"
        ),
        "optimizer": (
            "Adam"
            if adam_detected
            else "NOT_AUTOMATICALLY_VERIFIED"
        ),
        "gradient_clipping_detected": grad_clip_detected,
        "fit_model_source": source,
    }


def summarize_architecture(
    label: str,
    model: nn.Module,
) -> dict:
    total, trainable = count_parameters(model)

    modules = module_inventory(model)

    recurrent = [
        m for m in modules
        if m["type"] in {"LSTM", "GRU"}
    ]

    convs = [
        m for m in modules
        if m["type"] == "Conv1d"
    ]

    linears = [
        m for m in modules
        if m["type"] == "Linear"
    ]

    dropouts = [
        m for m in modules
        if m["type"] == "Dropout"
    ]

    summary = {
        "model": label,
        "input_features": INPUT_SIZE,
        "lookback": LOOKBACK,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "n_conv1d_layers": len(convs),
        "n_recurrent_layers_declared": (
            recurrent[0].get("num_layers")
            if recurrent
            else 0
        ),
        "recurrent_type": (
            recurrent[0]["type"]
            if recurrent
            else ""
        ),
        "recurrent_hidden_size": (
            recurrent[0].get("hidden_size")
            if recurrent
            else ""
        ),
        "recurrent_dropout": (
            recurrent[0].get("dropout")
            if recurrent
            else ""
        ),
        "conv_filters": (
            "; ".join(
                str(m.get("out_channels"))
                for m in convs
            )
        ),
        "conv_kernels": (
            "; ".join(
                str(m.get("kernel_size"))
                for m in convs
            )
        ),
        "linear_shapes": (
            "; ".join(
                f"{m.get('in_features')}->{m.get('out_features')}"
                for m in linears
            )
        ),
        "dropout_layers": (
            "; ".join(
                str(m.get("dropout"))
                for m in dropouts
            )
        ),
    }

    return summary


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    print("=" * 100)
    print("AUDITORIA DE ARQUITECTURAS, LOSS, OPTIMIZER Y PARAMETROS")
    print("=" * 100)

    common = load_module(
        "common_pipeline_archaudit",
        COMMON_PIPELINE,
    )

    training_choices = detect_training_choices(
        common
    )

    print(
        f"Loss detectada      : {training_choices['loss_function']}"
    )
    print(
        f"Optimizer detectado : {training_choices['optimizer']}"
    )
    print(
        "Gradient clipping   : "
        f"{training_choices['gradient_clipping_detected']}"
    )
    print()

    architecture_rows = []
    inventory_rows = []
    text_blocks = []

    for label, script_path in MODEL_SCRIPTS.items():
        module = load_module(
            f"archaudit_{label.replace('-', '_')}",
            script_path,
        )

        model = module.model_builder(
            input_size=INPUT_SIZE,
            lookback=LOOKBACK,
        )

        summary = summarize_architecture(
            label,
            model,
        )

        architecture_rows.append(
            {
                **summary,
                "loss_function": training_choices[
                    "loss_function"
                ],
                "optimizer": training_choices[
                    "optimizer"
                ],
                "gradient_clipping_detected": training_choices[
                    "gradient_clipping_detected"
                ],
            }
        )

        for item in module_inventory(model):
            inventory_rows.append(
                {
                    "model": label,
                    **item,
                }
            )

        text_blocks.extend(
            [
                "=" * 100,
                label,
                "=" * 100,
                str(model),
                "",
                f"Total parameters: {summary['total_parameters']:,}",
                f"Trainable parameters: {summary['trainable_parameters']:,}",
                "",
            ]
        )

        print(
            f"{label}: "
            f"{summary['total_parameters']:,} parametros"
        )

    architecture_df = pd.DataFrame(
        architecture_rows
    )

    inventory_df = pd.DataFrame(
        inventory_rows
    )

    architecture_path = (
        OUTPUT_DIR
        / "model_architecture_audit.csv"
    )

    inventory_path = (
        OUTPUT_DIR
        / "model_module_inventory.csv"
    )

    text_path = (
        OUTPUT_DIR
        / "model_architectures_readable.txt"
    )

    training_path = (
        OUTPUT_DIR
        / "training_configuration_audit.json"
    )

    architecture_df.to_csv(
        architecture_path,
        index=False,
        encoding="utf-8-sig",
    )

    inventory_df.to_csv(
        inventory_path,
        index=False,
        encoding="utf-8-sig",
    )

    text_path.write_text(
        "\n".join(text_blocks),
        encoding="utf-8",
    )

    training_payload = {
        "loss_function": training_choices[
            "loss_function"
        ],
        "optimizer": training_choices[
            "optimizer"
        ],
        "gradient_clipping_detected": training_choices[
            "gradient_clipping_detected"
        ],
        "input_features": INPUT_SIZE,
        "lookback": LOOKBACK,
        "source_file": str(
            COMMON_PIPELINE.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    training_path.write_text(
        json.dumps(
            training_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("TABLA DE ARQUITECTURAS")
    print("=" * 100)

    display_columns = [
        "model",
        "total_parameters",
        "n_conv1d_layers",
        "recurrent_type",
        "n_recurrent_layers_declared",
        "recurrent_hidden_size",
        "recurrent_dropout",
        "conv_filters",
        "conv_kernels",
        "linear_shapes",
        "loss_function",
        "optimizer",
        "gradient_clipping_detected",
    ]

    print(
        architecture_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Archivos generados:")
    print(architecture_path)
    print(inventory_path)
    print(text_path)
    print(training_path)


if __name__ == "__main__":
    main()
