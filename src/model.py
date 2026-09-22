from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    RGCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import softmax


class TokenGraphRGCN(nn.Module):
    """
    Graph-level vulnerability classifier.

    Input:
        x: [total_nodes, 1] containing integer token IDs
        edge_index: [2, total_edges]
        edge_type: [total_edges] containing relation IDs
        batch: [total_nodes] identifying each graph

    Output:
        logits: [number_of_graphs, 2]
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.30,
        node_feature_dim: int = 9,
        num_relations: int = 8,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.node_feature_projection = nn.Sequential(
            nn.Linear(node_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        if num_relations < 1:
            raise ValueError("num_relations must be at least 1.")

        self.conv1 = RGCNConv(embedding_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.input_residual = (
            nn.Identity()
            if embedding_dim == hidden_dim
            else nn.Linear(embedding_dim, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = dropout
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x has shape [total_nodes, 1].
        token_ids = x.view(-1).long()

        x = self.token_embedding(token_ids)
        if node_features is not None:
            x = x + self.node_feature_projection(node_features.float())

        x = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        residual = self.input_residual(x)
        x = self.conv2(x, edge_index, edge_type) + residual
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv3(x, edge_index, edge_type) + x
        x = self.norm3(x)
        x = F.relu(x)

        # Mean pooling captures common patterns; max pooling preserves salient
        # vulnerability-related tokens that mean pooling can dilute.
        mean_embedding = global_mean_pool(x, batch)
        max_embedding = global_max_pool(x, batch)
        attention_scores = self.attention_pool(x).squeeze(-1)
        attention_weights = softmax(
            attention_scores,
            batch,
        )
        attention_embedding = global_add_pool(
            x * attention_weights.unsqueeze(-1),
            batch,
        )
        graph_embedding = torch.cat(
            [attention_embedding, mean_embedding, max_embedding],
            dim=1,
        )

        return self.classifier(graph_embedding)
