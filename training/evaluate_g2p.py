#!/usr/bin/env python3
"""Compare rule-based, neural, and LLM-phonetician output against the
stress-test set's reference IPA, and log the comparison to Weights & Biases
if it's configured. An evaluation harness, not a training script — pair it
with finetune_g2p.py, which produces the checkpoint this can point at.

Caveat, worth repeating here and not just in the main README:
`draft_ipa` in data/stress_test.csv is reference material, not a verified
ground truth for every row. Treat exact-match accuracy below as a rough
signal to compare sources against each other, not a number to publish
without the human verification pass the README describes.

Usage:
    python training/evaluate_g2p.py
    python training/evaluate_g2p.py --checkpoint training/checkpoints/finetuned.pt
    python training/evaluate_g2p.py --no-llm --no-wandb
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truvox.scorer import phoneme_distance  # noqa: E402
from truvox.sources.llm_phonetician import LLMPhoneticianUnavailable, llm_phonetician  # noqa: E402
from truvox.sources.neural_g2p import NeuralG2PUnavailable, neural_g2p  # noqa: E402
from truvox.sources.rule_g2p import G2PMiss, rule_based_g2p  # noqa: E402


def evaluate_source(name: str, fn, rows: list[dict]) -> dict:
    exact, total_distance, attempted = 0, 0.0, 0
    for row in rows:
        word, locale, reference = row["word"], row["locale"], row["draft_ipa"]
        try:
            result = fn(word, locale)
        except (G2PMiss, NeuralG2PUnavailable, LLMPhoneticianUnavailable):
            continue
        attempted += 1
        if result.ipa == reference:
            exact += 1
        total_distance += phoneme_distance(result.ipa, reference)

    return {
        "source": name,
        "n_attempted": attempted,
        "n_total": len(rows),
        "exact_match_rate": round(exact / attempted, 3) if attempted else None,
        "avg_phoneme_distance": round(total_distance / attempted, 3) if attempted else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stress-set", default=Path("data/stress_test.csv"), type=Path)
    parser.add_argument("--checkpoint", default=None, help="fine-tuned neural G2P checkpoint path or URL")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.stress_set, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = [evaluate_source("rule_based", rule_based_g2p, rows)]
    results.append(
        evaluate_source(
            "neural_g2p",
            lambda word, locale: neural_g2p(word, locale, checkpoint=args.checkpoint),
            rows,
        )
    )
    if not args.no_llm:
        results.append(
            evaluate_source("llm_phonetician", lambda word, locale: llm_phonetician(word, locale), rows)
        )

    for r in results:
        print(r)

    if not args.no_wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed — skipping logging (pip install truvox[train])")
            return

        run = wandb.init(project="truvox-g2p-eval", config={"stress_set": str(args.stress_set)})
        table = wandb.Table(columns=list(results[0].keys()), data=[list(r.values()) for r in results])
        run.log({"source_comparison": table})
        run.finish()


if __name__ == "__main__":
    main()
