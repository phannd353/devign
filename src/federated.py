from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.data import load_json_records, records_to_graphs, stratified_record_split
from src.model import TokenGraphRGCN
from src.training import evaluate, train_one_epoch


def load_vocabulary(path: str) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("Vocabulary JSON must contain a non-empty object.")
    try:
        vocabulary = {str(token): int(index) for token, index in vocabulary.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Vocabulary values must be integer IDs.") from exc
    if vocabulary.get("<PAD>") != 0 or vocabulary.get("<UNK>") != 1:
        raise ValueError("Vocabulary must reserve <PAD>=0 and <UNK>=1.")
    return vocabulary


def create_model(
    vocabulary_size: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
) -> TokenGraphRGCN:
    return TokenGraphRGCN(
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_relations=8,
    )


def model_to_ndarrays(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def build_client_loaders(
    data_path: str,
    vocabulary: dict[str, int],
    batch_size: int,
    max_tokens: int,
    context_window: int,
    normalize_tokens: bool,
    structural_edges: bool,
    ast_edges: bool,
    data_flow_edges: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    records = load_json_records(data_path)
    try:
        train_records, validation_records, _ = stratified_record_split(
            records,
            seed=seed,
        )
    except ValueError:
        # Small or single-class clients cannot always satisfy stratification.
        generator = np.random.default_rng(seed)
        indices = generator.permutation(len(records))
        validation_count = max(1, int(round(len(records) * 0.15)))
        validation_count = min(validation_count, len(records) - 1)
        validation_indices = indices[:validation_count]
        train_indices = indices[validation_count:]
        train_records = [records[index] for index in train_indices]
        validation_records = [records[index] for index in validation_indices]
    graph_kwargs = {
        "vocabulary": vocabulary,
        "max_tokens": max_tokens,
        "context_window": context_window,
        "normalize_tokens": normalize_tokens,
        "structural_edges": structural_edges,
        "ast_edges": ast_edges,
        "data_flow_edges": data_flow_edges,
    }
    train_graphs = records_to_graphs(train_records, **graph_kwargs)
    validation_graphs = records_to_graphs(validation_records, **graph_kwargs)
    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(validation_graphs, batch_size=batch_size, shuffle=False),
    )


def build_criterion(device: torch.device, use_class_weights: bool, loader: DataLoader):
    if not use_class_weights:
        return nn.CrossEntropyLoss()
    labels = torch.cat([batch.y.view(-1) for batch in loader])
    counts = torch.bincount(labels, minlength=2).float()
    if torch.any(counts == 0):
        raise ValueError("A client needs both classes to use class weights.")
    weights = (len(labels) / (2.0 * counts)).to(device)
    return nn.CrossEntropyLoss(weight=weights)
