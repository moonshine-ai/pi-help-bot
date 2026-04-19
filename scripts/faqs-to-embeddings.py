"""
faqs-to-embeddings.py — Compute embeddings for FAQ questions using Moonshine Voice.

Walks a directory tree for *.faq.txt files, extracts Q: lines, computes
sentence embeddings via the moonshine-voice IntentRecognizer, and writes
the results to a JSON file.

Setup:
    pip install moonshine-voice

Run:
    python scripts/faqs-to-embeddings.py

Options:
    --faq-dir DIR        Root directory to search for .faq.txt files
                         (default: documentation/documentation)
    --output FILE        Output JSON path (default: embeddings.json)
    --variant VARIANT    Model variant: q4, q8, fp16, fp32 (default: q4)
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

from moonshine_voice import IntentRecognizer, get_embedding_model

_Q_LINE_RE = re.compile(r"^Q:\s*(.+)$")


def extract_questions(faq_path: Path) -> list[dict]:
    """Parse a .faq.txt file and return (question, answer) pairs with metadata."""
    entries = []
    lines = faq_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        m = _Q_LINE_RE.match(line.strip())
        if m:
            question = m.group(1).strip()
            answer = ""
            if i + 1 < len(lines):
                a_line = lines[i + 1].strip()
                if a_line.startswith("A:"):
                    answer = a_line[2:].strip()
            entries.append({
                "question": question,
                "answer": answer,
                "source": str(faq_path),
            })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute embeddings for FAQ questions using Moonshine Voice."
    )
    parser.add_argument(
        "--faq-dir",
        type=str,
        default="documentation/documentation",
        help="Root directory to search for .faq.txt files (default: documentation/documentation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings.json",
        help="Output JSON file path (default: embeddings.json)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="q4",
        choices=["q4", "q8", "fp16", "fp32", "q4f16"],
        help="Embedding model variant (default: q4)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Limit the number of questions to process",
    )
    args = parser.parse_args()

    faq_dir = Path(args.faq_dir)
    if not faq_dir.is_dir():
        parser.error(f"Not a directory: {faq_dir}")

    faq_files = sorted(faq_dir.rglob("*.faq.txt"))
    if not faq_files:
        print(f"No .faq.txt files found under {faq_dir}", file=sys.stderr)
        sys.exit(1)

    all_entries = []
    for faq_path in faq_files:
        entries = extract_questions(faq_path)
        all_entries.extend(entries)
    if args.n is not None:
        all_entries = all_entries[:args.n]
    print(
        f"Found {len(all_entries)} questions in {len(faq_files)} .faq.txt files",
        file=sys.stderr,
    )

    if not all_entries:
        print("No questions found, nothing to do.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model (variant={args.variant})...", file=sys.stderr)
    model_path, model_arch = get_embedding_model(
        "embeddinggemma-300m", args.variant
    )

    recognizer = IntentRecognizer(
        model_path=model_path,
        model_arch=model_arch,
        model_variant=args.variant,
    )

    results = []
    for entry in tqdm(all_entries, desc="Computing embeddings", unit="q"):
        embedding = recognizer.calculate_embedding(entry["question"])
        results.append({
            "question": entry["question"],
            "answer": entry["answer"],
            "source": entry["source"],
            "embedding": embedding,
        })

    recognizer.close()

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} entries to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
