#!/usr/bin/env python3
"""Fine-tune a small seq2seq G2P model on CMUdict + WikiPron, logged to W&B.

Status: scaffold. Correct end-to-end shape, not yet run to a finished
checkpoint — see training/README.md for why, and what "done" means here.

Usage (once cmudict.dict and a WikiPron TSV are downloaded — see the
`--cmudict-path` / `--wikipron-path` flags):

    python training/finetune_g2p.py \\
        --cmudict-path data/raw/cmudict.dict \\
        --wikipron-path data/raw/wikipron_multi.tsv \\
        --epochs 3 --batch-size 32

Requires the `train` extra: pip install truvox[train] transformers torch
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from truvox.sources.rule_g2p import _arpabet_token_to_ipa  # reuse the same IPA convention


def load_cmudict_pairs(path: Path) -> list[tuple[str, str]]:
    """word -> IPA pairs from a raw cmudict.dict file (one entry per line,
    `WORD  PH0 PH1 PH2 ...`). Skips CMUdict's alternate-pronunciation
    entries (WORD(1), WORD(2), ...) to keep one pair per surface form.
    """
    pairs = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;") or not line.strip():
                continue
            word, _, phones = line.strip().partition("  ")
            if "(" in word:  # alternate pronunciation variant, e.g. READ(1)
                continue
            tokens = phones.split()
            try:
                ipa = "".join(_arpabet_token_to_ipa(t) for t in tokens)
            except Exception:
                continue
            pairs.append((word.lower(), ipa))
    return pairs


def load_wikipron_pairs(path: Path) -> list[tuple[str, str]]:
    """word -> IPA pairs from a WikiPron TSV export (word\\tipa per line).
    See https://github.com/CUNY-CL/wikipron for per-language exports.
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                pairs.append((row[0].lower(), row[1].strip()))
    return pairs


def build_training_examples(cmudict_path: Path | None, wikipron_path: Path | None):
    examples: list[tuple[str, str]] = []
    if cmudict_path and cmudict_path.exists():
        examples += load_cmudict_pairs(cmudict_path)
    if wikipron_path and wikipron_path.exists():
        examples += load_wikipron_pairs(wikipron_path)
    if not examples:
        raise SystemExit(
            "no training data found — pass --cmudict-path and/or "
            "--wikipron-path pointing at downloaded dataset files "
            "(see training/README.md)"
        )
    return examples


def run_training(examples: list[tuple[str, str]], epochs: int, batch_size: int, use_wandb: bool):
    """The actual fine-tune loop. Kept minimal and swappable — this is the
    part most likely to change once you've picked a concrete base model
    (ByT5-small vs. DeepPhonemizer's own transformer config).
    """
    if use_wandb:
        try:
            import wandb

            wandb.init(project="truvox-g2p-finetune", config={
                "epochs": epochs, "batch_size": batch_size, "n_examples": len(examples),
            })
        except ImportError:
            print("wandb not installed (pip install truvox[train]) — continuing without logging")
            use_wandb = False

    # --- model/tokenizer setup, train/val split, and the training loop
    # itself go here. Left as the explicit next step rather than a fake
    # placeholder loop, since a half-real training loop is worse than an
    # honest TODO: it invites treating fabricated loss numbers as real.
    raise NotImplementedError(
        "model + training loop: pick a base checkpoint (ByT5-small is the "
        "easiest HF starting point), then fill in the HF Trainer/Seq2SeqTrainer "
        "call here. Data loading and the W&B run are already wired up above."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmudict-path", type=Path, default=None)
    parser.add_argument("--wikipron-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    examples = build_training_examples(args.cmudict_path, args.wikipron_path)
    print(f"loaded {len(examples)} word/IPA training pairs")
    run_training(examples, args.epochs, args.batch_size, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
