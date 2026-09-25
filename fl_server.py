from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch_geometric.data import DataLoader

from src.data import load_json_records, records_to_graphs
from src.federated import (
    build_criterion,
    config_value,
    create_model,
    load_vocabulary,
)
from src.model import TokenGraphRGCN
from src.training import evaluate

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
        evaluate_fn=get_global_evaluate_fn(context, model, vocabulary),
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


def get_global_evaluate_fn(
    context: Context, model: TokenGraphRGCN, vocabulary: dict[str, int]
):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and initialize it with the received weights
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)

        # Load entire eval set
        eval_records = load_json_records("data/federated/non-iid/server/test.json")
        graph_kwargs = {
            "max_tokens": config_value(context, "max_tokens", 512),
            "context_window": config_value(context, "context_window", 2),
            "normalize_tokens": config_value(context, "normalize_tokens", False),
            "structural_edges": config_value(context, "structural_edges", True),
            "ast_edges": config_value(context, "ast_edges", False),
            "data_flow_edges": config_value(context, "data_flow_edges", False),
            "seed": config_value(context, "seed", 42),
            "vocabulary": vocabulary,
        }
        eval_graphs = records_to_graphs(eval_records, **graph_kwargs)
        batch_size = config_value(context, "batch_size", 32)
        eval_loader = DataLoader(eval_graphs, batch_size=batch_size, shuffle=True)
        criterion = build_criterion(
            device, config_value(context, "class_weights", False), eval_loader
        )

        # Evaluate the global model on the test set
        test_loss, test_acc = evaluate(model, eval_loader, criterion, device)

        # Return the evaluation metrics
        return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})

    return global_evaluate
