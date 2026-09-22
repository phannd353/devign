from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import tree_sitter_c
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tree_sitter import Language, Parser

C_LANGUAGE = Language(tree_sitter_c.language())

KEYWORDS = {
    "alignas",
    "alignof",
    "asm",
    "auto",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "constexpr",
    "continue",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "explicit",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "namespace",
    "new",
    "noexcept",
    "nullptr",
    "operator",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
}

# Keep API names that often carry vulnerability semantics; normalize other
# identifiers so the model learns patterns instead of project-specific names.
SECURITY_APIS = {
    "alloca",
    "calloc",
    "free",
    "malloc",
    "memcpy",
    "memmove",
    "memset",
    "printf",
    "scanf",
    "snprintf",
    "sprintf",
    "sscanf",
    "strcat",
    "strcpy",
    "strlen",
    "strncat",
    "strncmp",
    "strncpy",
    "strstr",
    "system",
}

STRING_LITERAL = re.compile(r'^".*"$', re.DOTALL)
CHAR_LITERAL = re.compile(r"^'.*'$", re.DOTALL)
NUMBER_LITERAL = re.compile(r"^(?:\d+(?:\.\d+)?)$")
COMMENT = re.compile(r"^(?://|/\*).*(?:\*/)?$", re.DOTALL)
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
OPERATOR = {
    "!",
    "%",
    "&",
    "*",
    "+",
    "-",
    "/",
    "<",
    "=",
    ">",
    "^",
    "|",
    "~",
    "==",
    "!=",
    "<=",
    ">=",
    "->",
    "++",
    "--",
    "&&",
    "||",
}
TYPE_KEYWORDS = {
    "auto",
    "bool",
    "char",
    "double",
    "float",
    "int",
    "long",
    "short",
    "signed",
    "size_t",
    "struct",
    "unsigned",
    "void",
}


def tokenize_c_function(
    function_code: str,
    normalize: bool = False,
) -> list[str]:
    """
    Extract C tokens from Tree-sitter leaf nodes.
    """

    if not isinstance(function_code, str):
        raise TypeError("Function code must be a string.")

    tokens, _ = parse_c_function(function_code)
    if normalize:
        tokens = [normalize_token(token) for token in tokens]

    if not tokens:
        tokens = ["<EMPTY>"]

    return tokens


def parse_c_function(function_code: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return Tree-sitter leaf tokens and syntax-derived AST edges."""

    if not isinstance(function_code, str):
        raise TypeError("Function code must be a string.")

    try:
        parser = Parser(C_LANGUAGE)
    except TypeError:
        parser = Parser()
        parser.set_language(C_LANGUAGE)
    source = function_code.encode("utf-8")
    tree = parser.parse(source)
    leaves = []

    def collect_leaves(node) -> None:
        if node.child_count == 0:
            text = source[node.start_byte : node.end_byte].decode(
                "utf-8",
                errors="replace",
            )
            if text:
                leaves.append((node, text))
            return
        for child in node.children:
            collect_leaves(child)

    collect_leaves(tree.root_node)
    if not leaves:
        return ["<EMPTY>"], []

    leaf_tokens = [text for _, text in leaves]
    ast_edges = set()

    def collect_named_nodes(node) -> list:
        nodes = []
        if node.is_named and node.child_count > 0:
            nodes.append(node)
        for child in node.children:
            nodes.extend(collect_named_nodes(child))
        return nodes

    for ast_node in collect_named_nodes(tree.root_node):
        covered = [
            index
            for index, (leaf, _) in enumerate(leaves)
            if leaf.start_byte >= ast_node.start_byte
            and leaf.end_byte <= ast_node.end_byte
        ]
        if len(covered) < 2:
            continue
        first, last = covered[0], covered[-1]
        if first != last:
            ast_edges.add((first, last))
            ast_edges.add((last, first))

    return leaf_tokens, sorted(ast_edges)


def normalize_token(token: str) -> str:
    """Normalize tokens while preserving syntax and security-sensitive APIs."""

    if COMMENT.match(token):
        return "<COMMENT>"
    if STRING_LITERAL.match(token):
        return "<STRING>"
    if CHAR_LITERAL.match(token):
        return "<CHAR>"
    if NUMBER_LITERAL.match(token):
        return "<NUMBER>"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
        if token in KEYWORDS or token in SECURITY_APIS:
            return token
        return "<IDENTIFIER>"
    return token


def build_node_features(tokens: list[str]) -> torch.Tensor:
    """Build numeric features that describe each token's local role."""

    feature_rows = []
    token_count = max(len(tokens), 1)

    for index, token in enumerate(tokens):
        is_identifier = bool(IDENTIFIER.fullmatch(token))
        is_literal = bool(
            STRING_LITERAL.match(token)
            or CHAR_LITERAL.match(token)
            or NUMBER_LITERAL.match(token)
        )
        is_delimiter = token in {"(", ")", "[", "]", "{", "}"}
        feature_rows.append(
            [
                index / max(token_count - 1, 1),
                min(len(token), 32) / 32.0,
                float(is_identifier),
                float(token in KEYWORDS),
                float(is_literal),
                float(token in OPERATOR),
                float(is_delimiter),
                float(token in SECURITY_APIS),
                float(bool(COMMENT.match(token))),
            ]
        )

    return torch.tensor(feature_rows, dtype=torch.float32)


# ============================================================
# Vocabulary
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocabulary(
    records: list[dict],
    max_vocab_size: int = 20_000,
    min_frequency: int = 1,
    normalize_tokens: bool = False,
) -> dict[str, int]:
    counter = Counter()

    for record in records:
        tokens = tokenize_c_function(
            record["func"],
            normalize=normalize_tokens,
        )
        counter.update(tokens)

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    sorted_tokens = sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0]),
    )

    for token, frequency in sorted_tokens:
        if frequency < min_frequency:
            continue

        if token in vocabulary:
            continue

        if len(vocabulary) >= max_vocab_size:
            break

        vocabulary[token] = len(vocabulary)

    return vocabulary


# ============================================================
# JSON loading and validation
# ============================================================


def load_json_records(path: str) -> list[dict]:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path_obj.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("The JSON root must be a list.")

    valid_records = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object.")

        if "func" not in record:
            raise ValueError(f"Record {index} is missing 'func'.")

        if "target" not in record:
            raise ValueError(f"Record {index} is missing 'target'.")

        function_code = record["func"]
        target = record["target"]

        if not isinstance(function_code, str):
            raise ValueError(f"Record {index}: 'func' must be a string.")

        try:
            target = int(target)
        except (TypeError, ValueError):
            raise ValueError(f"Record {index}: 'target' must be 0 or 1.")

        if target not in (0, 1):
            raise ValueError(f"Record {index}: target={target}; expected 0 or 1.")

        if len(function_code.strip()) == 0:
            continue

        valid_records.append(
            {
                "func": function_code,
                "target": target,
            }
        )

    if len(valid_records) == 0:
        raise ValueError("No valid records were found.")

    return valid_records


# ============================================================
# Graph construction
# ============================================================


def make_token_graph(
    function_code: str,
    label: int,
    vocabulary: dict[str, int],
    max_tokens: int = 512,
    context_window: int = 2,
    normalize_tokens: bool = False,
    structural_edges: bool = False,
    ast_edges: bool = False,
    data_flow_edges: bool = False,
) -> Data:
    """
    Convert one function into a token graph.

    Nodes:
        One node per token.

    Edges:
        Nearby tokens and, optionally, matching delimiters connected in both
        directions. Edge types are 0/1 for sequential, 2/3 for delimiter,
        4/5 for heuristic AST, and 6/7 for heuristic data-flow edges.

    Features:
        Token IDs are stored in x. The model converts them into embeddings.
    """

    if context_window < 1:
        raise ValueError("context_window must be at least 1.")

    raw_tokens, parsed_ast_edges = parse_c_function(function_code)
    tokens = raw_tokens[:]
    if normalize_tokens:
        tokens = [normalize_token(token) for token in tokens]
    raw_tokens = raw_tokens[:max_tokens]
    tokens = tokens[:max_tokens]

    token_ids = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokens]

    x = torch.tensor(token_ids, dtype=torch.long).view(-1, 1)

    edges = {}

    for left in range(len(tokens)):
        for right in range(left + 1, min(left + context_window + 1, len(tokens))):
            edges[(left, right)] = 0
            edges[(right, left)] = 1

    if structural_edges:
        matching_delimiters = {")": "(", "]": "[", "}": "{"}
        delimiter_stack = []
        opening_delimiters = set(matching_delimiters.values())
        delimiter_pairs = []

        for index, token in enumerate(tokens):
            if token in opening_delimiters:
                delimiter_stack.append((token, index))
            elif token in matching_delimiters:
                expected = matching_delimiters[token]
                for stack_index in range(len(delimiter_stack) - 1, -1, -1):
                    opening, opening_index = delimiter_stack[stack_index]
                    if opening == expected:
                        delimiter_stack.pop(stack_index)
                        delimiter_pairs.append((opening_index, index))
                        edges[(opening_index, index)] = 2
                        edges[(index, opening_index)] = 3
                        break

    if ast_edges:
        for left, right in parsed_ast_edges:
            if left < len(tokens) and right < len(tokens):
                edges[(left, right)] = 4
                edges[(right, left)] = 5

    if data_flow_edges:
        definition_indices = {}
        for index, token in enumerate(raw_tokens):
            if not IDENTIFIER.fullmatch(token):
                continue

            previous = raw_tokens[index - 1] if index else ""
            next_token = raw_tokens[index + 1] if index + 1 < len(raw_tokens) else ""
            is_definition = (
                previous in TYPE_KEYWORDS
                or next_token == "="
                or (index > 1 and raw_tokens[index - 2] in TYPE_KEYWORDS)
            )
            if is_definition:
                definition_indices[token] = index
                continue

            definition_index = definition_indices.get(token)
            if definition_index is not None:
                edges[(definition_index, index)] = 6
                edges[(index, definition_index)] = 7

    if edges:
        sorted_edges = sorted(edges)
        edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(
            [edges[edge] for edge in sorted_edges],
            dtype=torch.long,
        )
    else:
        # A one-token graph has no sequential edges.
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )
        edge_type = torch.empty((0,), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        node_features=build_node_features(raw_tokens),
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
    )


def records_to_graphs(
    records: list[dict],
    vocabulary: dict[str, int],
    max_tokens: int = 512,
    context_window: int = 2,
    normalize_tokens: bool = False,
    structural_edges: bool = False,
    ast_edges: bool = False,
    data_flow_edges: bool = False,
) -> list[Data]:
    graphs = []

    for record in records:
        graph = make_token_graph(
            function_code=record["func"],
            label=record["target"],
            vocabulary=vocabulary,
            max_tokens=max_tokens,
            context_window=context_window,
            normalize_tokens=normalize_tokens,
            structural_edges=structural_edges,
            ast_edges=ast_edges,
            data_flow_edges=data_flow_edges,
        )
        graphs.append(graph)

    return graphs


# ============================================================


def stratified_record_split(
    records: list[dict],
    seed: int = 42,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    labels = [record["target"] for record in records]
    indices = np.arange(len(records))

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=validation_size + test_size,
        random_state=seed,
        stratify=labels,
    )

    temporary_labels = [labels[index] for index in temporary_indices]

    relative_test_size = test_size / (validation_size + test_size)

    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temporary_labels,
    )

    train_records = [records[index] for index in train_indices]
    validation_records = [records[index] for index in validation_indices]
    test_records = [records[index] for index in test_indices]

    return train_records, validation_records, test_records


# ============================================================
