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

from src.federated import config_value, create_local_state
from src.training import evaluate, train_one_epoch

app = ClientApp()


def load_message_arrays(model: nn.Module, message: Message) -> None:
    arrays = message.content["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())


@app.train()
def train(message: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    data_path = f"data/federated/non-iid/client_{int(partition_id) + 1}/train.json"
    model, device, train_loader, _, criterion = create_local_state(context, data_path)
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
    partition_id = context.node_config["partition-id"]
    data_path = f"data/federated/non-iid/client_{int(partition_id) + 1}/validation.json"
    model, device, _, validation_loader, criterion = create_local_state(
        context, data_path
    )
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
