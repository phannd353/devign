from __future__ import annotations

import argparse
import json

from src.data import build_vocabulary, load_json_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the shared vocabulary distributed to FL clients."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--normalize-tokens", action="store_true")
    args = parser.parse_args()

    records = load_json_records(args.data)
    vocabulary = build_vocabulary(
        records,
        max_vocab_size=args.max_vocab_size,
        min_frequency=args.min_frequency,
        normalize_tokens=args.normalize_tokens,
    )
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(vocabulary, file, indent=2, sort_keys=True)
    print(f"Saved {len(vocabulary)} tokens to {args.output}")


if __name__ == "__main__":
    main()
