#!/usr/bin/env python3
"""Run a slice of the stress-test set through the pipeline and, for whichever
TTS providers have API keys configured, synthesize each word twice — once
with no pronunciation override (baseline) and once with the pipeline's
ranked IPA (corrected).

This script produces audio files and a scoring template. It does not score
anything: listening to each baseline/corrected pair and filling in the two
score columns yourself is the actual benchmark, not a formality it can
skip on your behalf.

Usage:
    python benchmark/run_benchmark.py --limit 20
    python benchmark/run_benchmark.py --limit 20 --no-llm   # skip the LLM phonetician source
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `tts_providers`

from truvox.pipeline import get_pronunciation  # noqa: E402
from tts_providers import TTSProviderUnavailable, azure_speak, elevenlabs_speak  # noqa: E402

FIELDNAMES = [
    "word", "locale", "ranked_ipa", "confidence", "provenance",
    "elevenlabs_baseline", "elevenlabs_corrected",
    "azure_baseline", "azure_corrected",
    "score_baseline_1to5", "score_corrected_1to5", "listener_notes",
]


def _safe_filename(word: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", word).strip("_") or "word"


def _try_elevenlabs(word: str, ipa: str | None, audio_dir: Path, suffix: str) -> str:
    try:
        return elevenlabs_speak(
            word,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            out_path=str(audio_dir / f"{_safe_filename(word)}_elevenlabs_{suffix}.mp3"),
            ipa=ipa,
        )
    except TTSProviderUnavailable as exc:
        return f"skipped: {exc}"


def _try_azure(word: str, ipa: str | None, audio_dir: Path, suffix: str) -> str:
    try:
        return azure_speak(
            word,
            out_path=str(audio_dir / f"{_safe_filename(word)}_azure_{suffix}.wav"),
            ipa=ipa,
        )
    except TTSProviderUnavailable as exc:
        return f"skipped: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stress-set", default=_REPO_ROOT / "data" / "stress_test.csv", type=Path)
    parser.add_argument("--limit", type=int, default=20, help="how many rows to run through real TTS calls")
    parser.add_argument("--audio-dir", default=_REPO_ROOT / "benchmark" / "audio", type=Path)
    parser.add_argument("--out", default=_REPO_ROOT / "benchmark" / "scoring_template.csv", type=Path)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    with open(args.stress_set, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    out_rows = []
    for row in rows:
        word, locale = row["word"], row["locale"]
        try:
            result = get_pronunciation(word, locale=locale, use_llm=not args.no_llm)
        except RuntimeError as exc:
            print(f"  skipping {word!r}: {exc}")
            continue

        ipa = result.ranked.ipa
        record = {
            "word": word,
            "locale": locale,
            "ranked_ipa": ipa,
            "confidence": result.ranked.posterior,
            "provenance": result.ranked.provenance,
            "elevenlabs_baseline": _try_elevenlabs(word, None, args.audio_dir, "baseline"),
            "elevenlabs_corrected": _try_elevenlabs(word, ipa, args.audio_dir, "corrected"),
            "azure_baseline": _try_azure(word, None, args.audio_dir, "baseline"),
            "azure_corrected": _try_azure(word, ipa, args.audio_dir, "corrected"),
            "score_baseline_1to5": "",
            "score_corrected_1to5": "",
            "listener_notes": "",
        }
        out_rows.append(record)
        print(f"  {word!r} ({locale}): ranked /{ipa}/ at {record['confidence']:.2f}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nwrote {len(out_rows)} rows to {args.out}")
    print("listen to each baseline/corrected pair and fill in the two score columns — that pass is the benchmark.")


if __name__ == "__main__":
    main()
