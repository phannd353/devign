from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.federated import create_model, load_vocabulary


def config_value(context: Context, name: str, default):
    value = context.run_config.get(name, default)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    vocabulary_path = config_value(context, "vocabulary", "shared_vocabulary.json")
    rounds = config_value(context, "rounds", 20)
    min_clients = config_value(context, "min_clients", 2)
    embedding_dim = config_value(context, "embedding_dim", 128)
    hidden_dim = config_value(context, "hidden_dim", 128)
    dropout = config_value(context, "dropout", 0.30)
    learning_rate = config_value(context, "learning_rate", 1e-3)
    output = config_value(context, "output", "federated_checkpoint.pt")

    vocabulary = load_vocabulary(vocabulary_path)
    model = create_model(len(vocabulary), embedding_dim, hidden_dim, dropout)
    arrays = ArrayRecord(model.state_dict())

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_available_nodes=min_clients,
        min_train_nodes=min_clients,
        min_evaluate_nodes=min_clients,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "learning_rate": learning_rate,
                "local_epochs": config_value(context, "local_epochs", 1),
                "weight_decay": config_value(context, "weight_decay", 1e-4),
            }
        ),
        num_rounds=rounds,
    )

    final_state = result.arrays.to_torch_state_dict()
    torch.save(
        {
            "model_state_dict": final_state,
            "vocabulary": vocabulary,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_relations": 8,
            "rounds": rounds,
        },
        output,
    )
    print(f"Saved final global model to: {output}")
