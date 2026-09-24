from __future__ import annotations

import torch
import torch.nn as nn
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from src.federated import (
    build_client_loaders,
    build_criterion,
    create_model,
    load_vocabulary,
)
from src.reproducibility import set_seed
from src.training import evaluate, train_one_epoch

app = ClientApp()


def config_value[T](context: Context, name: str, default: T) -> T:
    """Read per-client values from node config, then shared run config."""

    if name in context.node_config:
        value = context.node_config[name]
    else:
        value = context.run_config.get(name, default)

    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def create_local_state(context: Context):
    set_seed(config_value(context, "seed", 42))
    vocabulary = load_vocabulary(
        config_value(context, "vocabulary", "shared_vocabulary.json")
    )
    model = create_model(
        vocabulary_size=len(vocabulary),
        embedding_dim=config_value(context, "embedding_dim", 128),
        hidden_dim=config_value(context, "hidden_dim", 128),
        dropout=config_value(context, "dropout", 0.30),
    )
    device = torch.device(
        config_value(
            context,
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    model.to(device)

    partition_id = context.node_config["partition-id"]
    data_path = f"data/federated/non-iid/client_{int(partition_id) + 1}/train.json"
    train_loader, validation_loader = build_client_loaders(
        data_path=data_path,
        vocabulary=vocabulary,
        batch_size=config_value(context, "batch_size", 32),
        max_tokens=config_value(context, "max_tokens", 512),
        context_window=config_value(context, "context_window", 2),
        normalize_tokens=config_value(context, "normalize_tokens", False),
        structural_edges=config_value(context, "structural_edges", True),
        ast_edges=config_value(context, "ast_edges", False),
        data_flow_edges=config_value(context, "data_flow_edges", False),
        seed=config_value(context, "seed", 42),
    )
    criterion = build_criterion(
        device,
        config_value(context, "class_weights", False),
        train_loader,
    )
    return model, device, train_loader, validation_loader, criterion


def load_message_arrays(model: nn.Module, message: Message) -> None:
    arrays = message.content["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())


@app.train()
def train(message: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    model, device, train_loader, _, criterion = create_local_state(context)
    load_message_arrays(model, message)

    config = message.content.get("config")
    if config is None:
        config = ConfigRecord({})
    learning_rate = float(
        config.get(
            "learning_rate",
            config_value(context, "learning_rate", 1e-3),
        )
    )
    local_epochs = int(
        config.get("local_epochs", config_value(context, "local_epochs", 1))
    )
    weight_decay = float(
        config.get("weight_decay", config_value(context, "weight_decay", 1e-4))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for _ in range(local_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    metrics = MetricRecord(
        {
            "num-examples": len(train_loader.dataset),
            "train-loss": float(
                evaluate(model, train_loader, criterion, device)["loss"]
            ),
        }
    )
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": metrics,
            }
        ),
        reply_to=message,
    )


@app.evaluate()
def evaluate_client(message: Message, context: Context) -> Message:
    model, device, _, validation_loader, criterion = create_local_state(context)
    load_message_arrays(model, message)
    metrics = evaluate(model, validation_loader, criterion, device)
    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "num-examples": len(validation_loader.dataset),
                        "loss": float(metrics["loss"]),
                        "accuracy": float(metrics["accuracy"]),
                        "mcc": float(metrics["mcc"]),
                        "auc": float(metrics["auc"]),
                    }
                )
            }
        ),
        reply_to=message,
    )
