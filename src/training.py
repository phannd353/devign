from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def compute_class_weights(
    graphs: list[Data],
    device: torch.device,
) -> torch.Tensor:
    labels = torch.tensor(
        [int(graph.y.item()) for graph in graphs],
        dtype=torch.long,
    )

    counts = torch.bincount(labels, minlength=2).float()

    if torch.any(counts == 0):
        raise ValueError(
            f"Training split must contain both classes. Counts: {counts.tolist()}"
        )

    weights = len(labels) / (2.0 * counts)

    return weights.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            node_features=batch.node_features,
        )

        labels = batch.y.view(-1).long()
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0,
        )

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    model.eval()

    total_loss = 0.0
    total_examples = 0

    labels_all = []
    predictions_all = []
    probabilities_all = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            node_features=batch.node_features,
        )

        labels = batch.y.view(-1).long()
        loss = criterion(logits, labels)

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = (probabilities >= threshold).long()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())

    average_loss = total_loss / max(total_examples, 1)

    accuracy = accuracy_score(labels_all, predictions_all)
    precision = precision_score(
        labels_all,
        predictions_all,
        zero_division=0,
    )
    recall = recall_score(
        labels_all,
        predictions_all,
        zero_division=0,
    )
    f1 = f1_score(
        labels_all,
        predictions_all,
        zero_division=0,
    )
    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0,
    )
    mcc = matthews_corrcoef(labels_all, predictions_all)

    if len(set(labels_all)) == 2:
        auc = roc_auc_score(labels_all, probabilities_all)
    else:
        auc = float("nan")

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "auc": auc,
    }


@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric: str = "mcc",
) -> tuple[float, float]:
    """Choose a decision threshold using validation predictions only."""

    if metric not in {"mcc", "macro_f1"}:
        raise ValueError("metric must be 'mcc' or 'macro_f1'.")

    model.eval()
    labels_all = []
    probabilities_all = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            node_features=batch.node_features,
        )
        labels_all.extend(batch.y.view(-1).cpu().tolist())
        probabilities_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    labels = np.asarray(labels_all)
    probabilities = np.asarray(probabilities_all)
    best_threshold = 0.5
    best_score = -float("inf")

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(labels, predictions)
        else:
            score = f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )

        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def print_metrics(
    split_name: str,
    metrics: dict[str, float],
) -> None:
    print(
        f"{split_name}: "
        f"loss={metrics['loss']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"macro-F1={metrics['macro_f1']:.4f}, "
        f"MCC={metrics['mcc']:.4f}, "
        f"AUC={metrics['auc']:.4f}"
    )
