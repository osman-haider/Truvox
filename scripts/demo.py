#!/usr/bin/env python3
"""Run the pronunciation ensemble on one word from the command line.

Examples:
    python scripts/demo.py "Saoirse" --locale en-IE --no-llm
    python scripts/demo.py "Atorvastatin" --context "prescribed medication, patient chart"

Runs whichever sources are actually available and prints a warning (not a
crash) for any that aren't — that's the same graceful-degradation contract
the pipeline itself relies on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up ANTHROPIC_API_KEY etc. from a local .env, if present
except ImportError:
    pass

from truvox.scorer import score_candidates  # noqa: E402
from truvox.sources.llm_phonetician import (  # noqa: E402
    LLMPhoneticianUnavailable,
    llm_phonetician,
)
from truvox.sources.neural_g2p import NeuralG2PUnavailable, neural_g2p  # noqa: E402
from truvox.sources.rule_g2p import G2PMiss, rule_based_g2p  # noqa: E402
from truvox.store import VerifiedStore  # noqa: E402


def gather_results(word: str, locale: str, context: str, use_llm: bool):
    results, warnings = [], []

    try:
        results.append(rule_based_g2p(word, locale))
    except G2PMiss as exc:
        warnings.append(f"rule-based G2P: {exc}")

    try:
        results.append(neural_g2p(word, locale))
    except (NeuralG2PUnavailable, G2PMiss) as exc:
        warnings.append(f"neural G2P: {exc}")

    if use_llm:
        try:
            results.append(llm_phonetician(word, locale, context))
        except LLMPhoneticianUnavailable as exc:
            warnings.append(f"LLM phonetician: {exc}")

    return results, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word")
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--no-llm", action="store_true", help="skip the LLM phonetician (no API key needed)"
    )
    parser.add_argument("--store", default="data/verified_store.db")
    parser.add_argument("--identity", default=None)
    args = parser.parse_args()

    store = VerifiedStore(args.store)
    verified = store.get(args.word, args.locale, identity_id=args.identity)
    if verified is not None:
        print(
            f"a verified entry is already on file: {verified.ipa}  "
            f"(confirmed by {verified.corrected_by} on {verified.corrected_at.date()})"
        )
        print("showing the fresh ensemble below anyway, for inspection —")
        print("the context-aware ranker is what actually prefers the verified entry.\n")

    results, warnings = gather_results(args.word, args.locale, args.context, not args.no_llm)

    for w in warnings:
        print(f"  (skipped) {w}")

    if not results:
        print("\nEvery source missed or was unavailable for this word.")
        return

    ranked = score_candidates(results)
    print(f"\nranked candidates for {args.word!r} ({args.locale}):")
    print(f"{'ipa':<22}{'score':>7}{'agreement':>11}   contributors")
    for c in ranked:
        contributors = ", ".join(p.value for p in c.contributors)
        print(f"{c.ipa:<22}{c.score:>7.3f}{c.agreement:>11.3f}   {contributors}")


if __name__ == "__main__":
    main()
