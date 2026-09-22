from __future__ import annotations

import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion

from src.data import load_json_records, stratified_record_split


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def find_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_mcc = -float("inf")

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = matthews_corrcoef(labels, predictions)
        if score > best_mcc:
            best_threshold = float(threshold)
            best_mcc = float(score)

    return best_threshold, best_mcc


def evaluate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(labels, predictions),
        "auc": roc_auc_score(labels, probabilities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a word/character TF-IDF vulnerability baseline."
    )
    parser.add_argument("--data", required=True, help="Path to JSON dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
    )
    args = parser.parse_args()

    records = load_json_records(args.data)
    train_records, validation_records, test_records = stratified_record_split(
        records,
        seed=args.seed,
    )

    train_text = [record["func"] for record in train_records]
    validation_text = [record["func"] for record in validation_records]
    test_text = [record["func"] for record in test_records]
    train_labels = np.asarray([record["target"] for record in train_records])
    validation_labels = np.asarray(
        [record["target"] for record in validation_records]
    )
    test_labels = np.asarray([record["target"] for record in test_records])

    vectorizer = make_vectorizer()
    train_features = vectorizer.fit_transform(train_text)
    validation_features = vectorizer.transform(validation_text)
    test_features = vectorizer.transform(test_text)

    classifier = LogisticRegression(
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="liblinear",
    )
    classifier.fit(train_features, train_labels)

    validation_probabilities = classifier.predict_proba(validation_features)[:, 1]
    threshold, validation_mcc = find_threshold(
        validation_labels,
        validation_probabilities,
    )

    test_probabilities = classifier.predict_proba(test_features)[:, 1]
    metrics = evaluate(test_labels, test_probabilities, threshold)

    print(f"Features: {train_features.shape[1]}")
    print(f"Selected validation threshold: {threshold:.2f}")
    print(f"Validation MCC: {validation_mcc:.4f}")
    print(
        "Test: "
        + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )

    test_predictions = (test_probabilities >= threshold).astype(int)
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=["non-vulnerable", "vulnerable"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, test_predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
