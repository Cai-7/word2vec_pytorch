"""Word2Vec analogy evaluation for the original Mikolov benchmark.

This script evaluates the official word relationship test set from
"Efficient Estimation of Word Representations in Vector Space".

It supports:
- parsing word-test.v1.txt with semantic / syntactic grouping
- loading word2vec vectors from gensim KeyedVectors .bin or .txt files
- cosine-similarity analogy inference with OOV filtering
- table-style output for overall, semantic, syntactic, and per-relation accuracy
- a few qualitative inference samples
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Project paths: edit these defaults here, or override them from the command line.
# -----------------------------------------------------------------------------
DEFAULT_TEST_PATH = r"word-test.v1.txt"
DEFAULT_EMBEDDING_PATH = r"word_embedding.txt"
DEFAULT_TOPK = 5
DEFAULT_SAMPLE_COUNT = 8


SEMANTIC_RELATIONS = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}

SYNTACTIC_RELATIONS = {
    "adjective-to-adverb",
    "opposite",
    "comparative",
    "superlative",
    "present-participle",
    "nationality-adjective",
    "past-tense",
    "plural",
    "plural-verbs",
}


@dataclass
class AnalogyQuestion:
    relation: str
    category: str
    w1: str
    w2: str
    w3: str
    w4: str


@dataclass
class RelationStats:
    total: int = 0
    valid: int = 0
    correct: int = 0
    oov: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.valid if self.valid else 0.0


@dataclass
class EvaluationReport:
    overall: RelationStats = field(default_factory=RelationStats)
    category_stats: Dict[str, RelationStats] = field(default_factory=dict)
    relation_stats: Dict[str, RelationStats] = field(default_factory=dict)
    samples: List[dict] = field(default_factory=list)
    total_questions: int = 0
    total_oov_questions: int = 0


def classify_relation(relation: str) -> str:
    """Map each official relation to semantic or syntactic."""

    if relation in SEMANTIC_RELATIONS:
        return "semantic"
    if relation in SYNTACTIC_RELATIONS:
        return "syntactic"
    if relation.startswith("gram") and "-" in relation:
        suffix = relation.split("-", 1)[1]
        if suffix in SYNTACTIC_RELATIONS:
            return "syntactic"
    return "unknown"


def normalize_path(path: str) -> str:
    if not path:
        raise ValueError("Empty path is not allowed.")
    return os.path.expanduser(path)


def load_keyed_vectors(path: str):
    """Load word vectors with gensim.

    The loader automatically tries the expected binary/text mode, and then the
    opposite mode if the first attempt fails.
    """

    path = normalize_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")

    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:
        raise RuntimeError(
            "gensim is required. Install it with: pip install gensim"
        ) from exc

    guessed_binary = path.lower().endswith(".bin")
    attempts = [guessed_binary, not guessed_binary]
    errors: List[str] = []

    for binary in dict.fromkeys(attempts):
        try:
            return KeyedVectors.load_word2vec_format(path, binary=binary)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            errors.append(f"binary={binary}: {exc}")

    raise RuntimeError(
        "Failed to load embeddings with gensim. Tried both binary and text modes.\n"
        + "\n".join(errors)
    )


def parse_word_analogy_file(path: str) -> List[AnalogyQuestion]:
    """Parse word-test.v1.txt and attach semantic / syntactic labels."""

    path = normalize_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    questions: List[AnalogyQuestion] = []
    current_relation: Optional[str] = None

    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith(":"):
                relation_name = line[1:].strip()
                if not relation_name:
                    raise ValueError(
                        f"Malformed relation header at line {line_no}: {raw_line.rstrip()}"
                    )
                current_relation = relation_name.split()[0]
                continue

            if current_relation is None:
                raise ValueError(
                    f"Question line appears before any relation header at line {line_no}: {raw_line.rstrip()}"
                )

            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Expected 4 tokens at line {line_no}, got {len(parts)}: {raw_line.rstrip()}"
                )

            w1, w2, w3, w4 = parts
            questions.append(
                AnalogyQuestion(
                    relation=current_relation,
                    category=classify_relation(current_relation),
                    w1=w1,
                    w2=w2,
                    w3=w3,
                    w4=w4,
                )
            )

    if not questions:
        raise ValueError(f"No analogy questions were found in {path}")

    return questions


def _format_percent(value: float) -> str:
    return f"{value * 100:6.2f}"


def _format_int(value: int) -> str:
    return f"{value:d}"


def print_table(title: str, rows: Sequence[Sequence[str]], headers: Sequence[str]) -> None:
    """Print a paper-style aligned table."""

    if not rows:
        print(f"\n{title}")
        print("(no rows)")
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def render_row(row: Sequence[str]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print(f"\n{title}")
    print(render_row(headers))
    print(separator)
    for row in rows:
        print(render_row(row))


def build_vector_cache(kv) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    """Build normalized vectors for cosine search."""

    vectors = np.asarray(kv.vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    norms = np.where(norms == 0.0, 1e-12, norms)
    normalized_vectors = vectors / norms[:, None]

    if hasattr(kv, "key_to_index"):
        word_to_index = dict(kv.key_to_index)
        index_to_word = [None] * len(word_to_index)
        for word, index in word_to_index.items():
            index_to_word[index] = word
    else:  # pragma: no cover - older gensim compatibility
        word_to_index = dict(kv.vocab)  # type: ignore[attr-defined]
        index_to_word = [None] * len(word_to_index)
        for word, vocab_item in word_to_index.items():
            index_to_word[vocab_item.index] = word

    return vectors, normalized_vectors, word_to_index, index_to_word


def top_k_predictions(
    query_vector: np.ndarray,
    normalized_vectors: np.ndarray,
    index_to_word: Sequence[str],
    banned_indices: Sequence[int],
    top_k: int,
) -> List[Tuple[str, float]]:
    """Return top-k words ranked by cosine similarity."""

    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return []

    scores = normalized_vectors @ (query_vector / query_norm)
    if banned_indices:
        scores = scores.copy()
        scores[np.asarray(list(banned_indices), dtype=np.int64)] = -np.inf

    top_k = max(1, min(top_k, scores.shape[0]))
    best_idx = np.argpartition(scores, -top_k)[-top_k:]
    best_idx = best_idx[np.argsort(scores[best_idx])[::-1]]
    return [(index_to_word[idx], float(scores[idx])) for idx in best_idx]


def evaluate_analogies(
    kv,
    questions: Sequence[AnalogyQuestion],
    top_k: int = DEFAULT_TOPK,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    lowercase: bool = False,
) -> EvaluationReport:
    """Evaluate word analogies using the original word2vec rule.

    The paper's vector arithmetic is vec(w2) - vec(w1) + vec(w3).

    Args:
        lowercase: If True, lowercase analogy words before vocab lookup.
            Needed for text8 / most cleaned English dumps (all-lower vocab)
            against the official capitalized word-test.v1.txt.
    """

    vectors, normalized_vectors, word_to_index, index_to_word = build_vector_cache(kv)

    report = EvaluationReport()
    report.total_questions = len(questions)

    relation_order: List[str] = []

    for question in questions:
        if question.category not in report.category_stats:
            report.category_stats[question.category] = RelationStats()

        if question.relation not in report.relation_stats:
            report.relation_stats[question.relation] = RelationStats()
            relation_order.append(question.relation)

        overall_stats = report.overall
        category_stats = report.category_stats[question.category]
        relation_stats = report.relation_stats[question.relation]

        for stats in (overall_stats, category_stats, relation_stats):
            stats.total += 1

        raw_words = (question.w1, question.w2, question.w3, question.w4)
        query_words = tuple(w.lower() if lowercase else w for w in raw_words)

        indices: List[int] = []
        missing_words: List[str] = []
        for raw, query in zip(raw_words, query_words):
            index = word_to_index.get(query)
            if index is None:
                missing_words.append(raw)
            else:
                indices.append(index)

        if missing_words:
            report.total_oov_questions += 1
            for stats in (overall_stats, category_stats, relation_stats):
                stats.oov += 1
            continue

        for stats in (overall_stats, category_stats, relation_stats):
            stats.valid += 1

        # Section 4.1 in the paper: analogy vector arithmetic.
        query_vector = vectors[indices[1]] - vectors[indices[0]] + vectors[indices[2]]
        predictions = top_k_predictions(
            query_vector=query_vector,
            normalized_vectors=normalized_vectors,
            index_to_word=index_to_word,
            banned_indices=indices[:3],
            top_k=top_k,
        )

        predicted_word = predictions[0][0] if predictions else None
        gold = query_words[3]
        if predicted_word is None:
            is_correct = False
        elif lowercase:
            is_correct = predicted_word.lower() == gold
        else:
            is_correct = predicted_word == question.w4
        if is_correct:
            for stats in (overall_stats, category_stats, relation_stats):
                stats.correct += 1

        if len(report.samples) < sample_count:
            report.samples.append(
                {
                    "question": question,
                    "predicted": predicted_word,
                    "correct": is_correct,
                    "top_k": predictions,
                    "missing_words": missing_words,
                }
            )

    # Keep relation insertion order stable for printing.
    report.relation_stats = {relation: report.relation_stats[relation] for relation in relation_order}
    return report


def summarize_report(report: EvaluationReport) -> None:
    """Print overall metrics and paper-style tables."""

    print("\n================ Word Analogy Evaluation ================")
    print(f"Total questions: {report.total_questions}")
    print(f"OOV questions  : {report.total_oov_questions}")
    print(f"Valid questions : {report.overall.valid}")
    print(f"Overall accuracy: {report.overall.accuracy * 100:.2f}%")

    category_rows = []
    for category in ("semantic", "syntactic", "unknown"):
        stats = report.category_stats.get(category)
        if stats is None:
            continue
        category_rows.append(
            [
                category,
                _format_int(stats.total),
                _format_int(stats.valid),
                _format_int(stats.oov),
                _format_int(stats.correct),
                _format_percent(stats.accuracy),
            ]
        )
    print_table(
        "Table 1. Semantic vs. syntactic accuracy",
        category_rows,
        ["Category", "Total", "Valid", "OOV", "Correct", "Acc (%)"],
    )

    relation_rows = []
    for relation, stats in report.relation_stats.items():
        relation_rows.append(
            [
                relation,
                _format_int(stats.total),
                _format_int(stats.valid),
                _format_int(stats.oov),
                _format_int(stats.correct),
                _format_percent(stats.accuracy),
            ]
        )

    semantic_rows = [row for row in relation_rows if classify_relation(row[0]) == "semantic"]
    syntactic_rows = [row for row in relation_rows if classify_relation(row[0]) == "syntactic"]

    print_table(
        "Table 3. Semantic relation accuracy",
        semantic_rows,
        ["Relation", "Total", "Valid", "OOV", "Correct", "Acc (%)"],
    )
    print_table(
        "Table 4. Syntactic relation accuracy",
        syntactic_rows,
        ["Relation", "Total", "Valid", "OOV", "Correct", "Acc (%)"],
    )

    if report.samples:
        print("\nQualitative samples")
        for idx, item in enumerate(report.samples, start=1):
            question = item["question"]
            predicted = item["predicted"]
            status = "correct" if item["correct"] else "wrong"
            print(
                f"[{idx}] {question.relation} ({question.category})  "
                f"{question.w1}:{question.w2} :: {question.w3}:?  "
                f"gold={question.w4}  pred={predicted}  [{status}]"
            )
            if item["top_k"]:
                topk_str = ", ".join(f"{word} ({score:.4f})" for word, score in item["top_k"])
                print(f"    top-{len(item['top_k'])}: {topk_str}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate word2vec embeddings on the official word analogy benchmark."
    )
    parser.add_argument(
        "--test-path",
        default=DEFAULT_TEST_PATH,
        help="Path to word-test.v1.txt (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding-path",
        default=DEFAULT_EMBEDDING_PATH,
        help="Path to a gensim KeyedVectors .txt or .bin file (default: %(default)s)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="How many predictions to show for qualitative samples (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="How many analogy examples to print (default: %(default)s)",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase analogy words before lookup. Required for text8-style "
             "all-lower embeddings against capitalized word-test.v1.txt.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        questions = parse_word_analogy_file(args.test_path)
        kv = load_keyed_vectors(args.embedding_path)
        if args.lowercase:
            print("[Info] lowercase matching enabled (text8 / uncased vocab)")
        report = evaluate_analogies(
            kv=kv,
            questions=questions,
            top_k=args.topk,
            sample_count=args.samples,
            lowercase=args.lowercase,
        )
        summarize_report(report)
    except Exception as exc:
        print(f"[Error] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()