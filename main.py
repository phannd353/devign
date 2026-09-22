from __future__ import annotations

import argparse
import copy
from collections import Counter

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch_geometric.loader import DataLoader

from src.data import (
    build_vocabulary,
    load_json_records,
    records_to_graphs,
    stratified_record_split,
)
from src.model import TokenGraphRGCN
from src.reproducibility import set_seed
from src.training import (
    compute_class_weights,
    evaluate,
    find_best_threshold,
    print_metrics,
    train_one_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True, help="Path to JSON dataset.")
    parser.add_argument("--output", type=str, default="token_gcn_checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument(
        "--normalize-tokens",
        action="store_true",
        help="Normalize non-security identifiers and literals before graph creation.",
    )
    parser.add_argument(
        "--no-structural-edges",
        dest="structural_edges",
        action="store_false",
        help="Disable matching-delimiter edges.",
    )
    parser.set_defaults(structural_edges=True)
    parser.add_argument(
        "--ast-edges",
        action="store_true",
        help="Add heuristic AST hierarchy edges.",
    )
    parser.add_argument(
        "--data-flow-edges",
        action="store_true",
        help="Add heuristic identifier definition/use edges.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Train with unweighted cross-entropy.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["mcc", "macro_f1"],
        default="mcc",
        help="Validation metric used to select the classification threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    records = load_json_records(args.data)

    label_counts = Counter(record["target"] for record in records)

    print(f"Total records: {len(records)}")
    print(f"Benign records: {label_counts[0]}")
    print(f"Vulnerable records: {label_counts[1]}")

    train_records, validation_records, test_records = stratified_record_split(
        records,
        seed=args.seed,
    )

    # Build vocabulary using training data only.
    vocabulary = build_vocabulary(
        train_records,
        max_vocab_size=args.max_vocab_size,
        normalize_tokens=args.normalize_tokens,
    )

    train_graphs = records_to_graphs(
        train_records,
        vocabulary,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        normalize_tokens=args.normalize_tokens,
        structural_edges=args.structural_edges,
        ast_edges=args.ast_edges,
        data_flow_edges=args.data_flow_edges,
    )
    validation_graphs = records_to_graphs(
        validation_records,
        vocabulary,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        normalize_tokens=args.normalize_tokens,
        structural_edges=args.structural_edges,
        ast_edges=args.ast_edges,
        data_flow_edges=args.data_flow_edges,
    )
    test_graphs = records_to_graphs(
        test_records,
        vocabulary,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        normalize_tokens=args.normalize_tokens,
        structural_edges=args.structural_edges,
        ast_edges=args.ast_edges,
        data_flow_edges=args.data_flow_edges,
    )

    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(validation_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    print(f"Vocabulary size: {len(vocabulary)}")

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_graphs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TokenGraphRGCN(
        vocabulary_size=len(vocabulary),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(
        train_graphs,
        device,
    )

    print(
        "Class weights:",
        class_weights.detach().cpu().tolist(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=None if args.no_class_weights else class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_validation_score = -float("inf")
    best_validation_f1 = -float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        train_metrics = evaluate(
            model,
            train_loader,
            criterion,
            device,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_threshold, validation_threshold_score = find_best_threshold(
            model,
            validation_loader,
            criterion,
            device,
            metric=args.threshold_metric,
        )
        thresholded_validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            threshold=validation_threshold,
        )

        scheduler.step(validation_threshold_score)

        print(
            f"\nEpoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        print_metrics("Train", train_metrics)
        print_metrics("Valid", validation_metrics)
        print(
            f"Valid threshold: {validation_threshold:.2f} | "
            f"{args.threshold_metric}={validation_threshold_score:.4f}"
        )

        current_score = validation_threshold_score

        if current_score > best_validation_score:
            best_validation_score = current_score
            best_validation_f1 = thresholded_validation_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print("Saved best model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No model checkpoint was saved.")

    model.load_state_dict(best_state)

    decision_threshold, threshold_score = find_best_threshold(
        model,
        validation_loader,
        criterion,
        device,
        metric=args.threshold_metric,
    )

    print(
        f"\nSelected decision threshold: {decision_threshold:.2f} "
        f"({args.threshold_metric}={threshold_score:.4f})"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "max_tokens": args.max_tokens,
        "context_window": args.context_window,
        "structural_edges": args.structural_edges,
        "ast_edges": args.ast_edges,
        "data_flow_edges": args.data_flow_edges,
        "normalize_tokens": args.normalize_tokens,
        "class_weights": not args.no_class_weights,
        "num_relations": 8,
        "best_validation_f1": best_validation_f1,
        "best_validation_score": best_validation_score,
        "threshold_metric": args.threshold_metric,
        "decision_threshold": decision_threshold,
    }

    torch.save(checkpoint, args.output)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=decision_threshold,
    )

    print("\nFinal test result:")
    print_metrics("Test", test_metrics)

    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_type,
                batch.batch,
                node_features=batch.node_features,
            )

            all_labels.extend(batch.y.view(-1).cpu().tolist())
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            all_predictions.extend(
                (probabilities >= decision_threshold).long().cpu().tolist()
            )

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "non-vulnerable",
                "vulnerable",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            all_labels,
            all_predictions,
            labels=[0, 1],
        )
    )

    print(f"Checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
