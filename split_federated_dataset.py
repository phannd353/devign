from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

CLIENT_COUNT = 5
CLIENT_RATIOS = (0.18, 0.19, 0.20, 0.21, 0.22)


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("The dataset must be a non-empty JSON list.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not an object.")
        if "func" not in record or "target" not in record:
            raise ValueError(f"Record {index} must contain 'func' and 'target'.")
        if int(record["target"]) not in (0, 1):
            raise ValueError(f"Record {index} has an invalid target.")
    return [
        {"func": record["func"], "target": int(record["target"])} for record in records
    ]


def allocate_counts(total: int, weights: Iterable[float]) -> list[int]:
    """Allocate an integer total according to weights using largest remainder."""

    weights = list(weights)
    if total < 0 or not weights or any(weight < 0 for weight in weights):
        raise ValueError("Invalid allocation inputs.")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Allocation weights must have a positive sum.")

    exact = [total * weight / weight_sum for weight in weights]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: exact[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def split_local_dataset(
    records: list[dict],
    rng: random.Random,
) -> dict[str, list[dict]]:
    """Split a client dataset into 80% train, 10% validation, 10% test."""

    by_class = {
        0: [record for record in records if record["target"] == 0],
        1: [record for record in records if record["target"] == 1],
    }
    for class_records in by_class.values():
        rng.shuffle(class_records)

    splits = {"train": [], "validation": [], "test": []}
    for class_records in by_class.values():
        counts = allocate_counts(len(class_records), (0.8, 0.1, 0.1))
        start = 0
        for split_name, count in zip(splits, counts):
            splits[split_name].extend(class_records[start : start + count])
            start += count

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def split_server_records(
    records: list[dict],
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Reserve 10% for the server using stratified class sampling."""

    by_class = {
        0: [record for record in records if record["target"] == 0],
        1: [record for record in records if record["target"] == 1],
    }
    server = []
    remaining = []
    for class_records in by_class.values():
        rng.shuffle(class_records)
        server_count = round(len(class_records) * 0.1)
        server.extend(class_records[:server_count])
        remaining.extend(class_records[server_count:])

    rng.shuffle(server)
    rng.shuffle(remaining)
    return server, remaining


def split_iid_clients(
    records: list[dict],
    rng: random.Random,
) -> list[list[dict]]:
    """Create five equally sized, approximately 50/50 client datasets."""

    by_class = {
        0: [record for record in records if record["target"] == 0],
        1: [record for record in records if record["target"] == 1],
    }
    client_sizes = allocate_counts(len(records), (1,) * CLIENT_COUNT)
    vulnerable_total = len(by_class[1])
    vulnerable_ratio = vulnerable_total / len(records)
    exact_vulnerable = [
        size * vulnerable_ratio for size in client_sizes
    ]
    vulnerable_counts = [int(value) for value in exact_vulnerable]
    remainder = vulnerable_total - sum(vulnerable_counts)
    order = sorted(
        range(CLIENT_COUNT),
        key=lambda index: exact_vulnerable[index] - vulnerable_counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        vulnerable_counts[index] += 1
    class_counts = {
        1: vulnerable_counts,
        0: [
            size - vulnerable
            for size, vulnerable in zip(client_sizes, vulnerable_counts)
        ],
    }

    clients = [[] for _ in range(CLIENT_COUNT)]
    for target in (0, 1):
        rng.shuffle(by_class[target])
        start = 0
        for client_index, count in enumerate(class_counts[target]):
            clients[client_index].extend(by_class[target][start : start + count])
            start += count

    if any(len(client) != client_sizes[index] for index, client in enumerate(clients)):
        raise RuntimeError("IID client allocation produced incorrect sizes.")

    for client in clients:
        rng.shuffle(client)
    return clients


def split_non_iid_clients(
    records: list[dict],
    rng: random.Random,
) -> tuple[list[list[dict]], list[float]]:
    """Create clients with 18/19/20/21/22% sizes and random class ratios."""

    client_sizes = allocate_counts(len(records), CLIENT_RATIOS)
    by_class = {
        0: [record for record in records if record["target"] == 0],
        1: [record for record in records if record["target"] == 1],
    }

    random_class_weights = [rng.uniform(0.2, 0.8) for _ in range(CLIENT_COUNT)]
    vulnerable_counts = allocate_counts(
        len(by_class[1]),
        [size * ratio for size, ratio in zip(client_sizes, random_class_weights)],
    )
    # Ensure every client has at least one example of each class when possible.
    if len(by_class[0]) >= CLIENT_COUNT and len(by_class[1]) >= CLIENT_COUNT:
        for index in range(CLIENT_COUNT):
            vulnerable_counts[index] = max(
                1,
                min(vulnerable_counts[index], client_sizes[index] - 1),
            )
        difference = len(by_class[1]) - sum(vulnerable_counts)
        while difference:
            direction = 1 if difference > 0 else -1
            for index, size in enumerate(client_sizes):
                candidate = vulnerable_counts[index] + direction
                if 1 <= candidate <= size - 1:
                    vulnerable_counts[index] = candidate
                    difference -= direction
                    if difference == 0:
                        break

    benign_counts = [
        size - vulnerable for size, vulnerable in zip(client_sizes, vulnerable_counts)
    ]
    clients = [[] for _ in range(CLIENT_COUNT)]
    for target, counts in ((0, benign_counts), (1, vulnerable_counts)):
        rng.shuffle(by_class[target])
        start = 0
        for index, count in enumerate(counts):
            clients[index].extend(by_class[target][start : start + count])
            start += count

    for client in clients:
        rng.shuffle(client)
    class_ratios = [
        vulnerable / size for vulnerable, size in zip(vulnerable_counts, client_sizes)
    ]
    return clients, class_ratios


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)


def summarize(records: list[dict]) -> dict[str, int]:
    return {
        "total": len(records),
        "non_vulnerable": sum(record["target"] == 0 for record in records),
        "vulnerable": sum(record["target"] == 1 for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create IID or Non-IID pseudo federated datasets."
    )
    parser.add_argument("--input", default="data/raw/dataset.json")
    parser.add_argument("--output-dir", default="data/federated")
    parser.add_argument("--mode", choices=("iid", "non-iid"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_records(Path(args.input))
    server_records, client_pool = split_server_records(records, rng)

    output_dir = Path(args.output_dir) / args.mode
    clients = (
        split_iid_clients(client_pool, rng)
        if args.mode == "iid"
        else split_non_iid_clients(client_pool, rng)[0]
    )

    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "source": str(Path(args.input)),
        "server": summarize(server_records),
        "clients": {},
    }
    write_json(output_dir / "server" / "test.json", server_records)

    for index, client_records in enumerate(clients, start=1):
        client_splits = split_local_dataset(client_records, rng)
        client_dir = output_dir / f"client_{index}"
        for split_name, split_records in client_splits.items():
            write_json(client_dir / f"{split_name}.json", split_records)
        metadata["clients"][f"client_{index}"] = {
            "all": summarize(client_records),
            **{split: summarize(items) for split, items in client_splits.items()},
        }

    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
