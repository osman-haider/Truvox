"""The single orchestration point: name in, ranked pronunciation out.

Both the MCP tool and the benchmark harness call `get_pronunciation` — the
ensemble, scoring, storage, and ranking logic each live in exactly one
place, so neither caller re-implements or subtly diverges from the other.
"""
from __future__ import annotations

from dataclasses import dataclass

from truvox.ranker import RankedPronunciation, rank
from truvox.schema import SourceResult
from truvox.scorer import score_candidates
from truvox.sources.llm_phonetician import LLMPhoneticianUnavailable, llm_phonetician
from truvox.sources.neural_g2p import NeuralG2PUnavailable, neural_g2p
from truvox.sources.rule_g2p import G2PMiss, rule_based_g2p
from truvox.store import VerifiedStore


@dataclass
class PipelineResult:
    word: str
    locale: str
    ranked: RankedPronunciation
    sources_used: tuple[str, ...]
    sources_skipped: tuple[str, ...]  # human-readable reasons — surfaced, not swallowed


def get_pronunciation(
    word: str,
    locale: str = "en-US",
    context: str = "",
    identity_id: str | None = None,
    store: VerifiedStore | None = None,
    use_llm: bool = True,
    llm_client=None,
) -> PipelineResult:
    """Run the full pipeline for one word and return its best-known
    pronunciation. Raises RuntimeError only when every source missed or was
    unavailable and nothing has ever been verified for this word — a real
    "we have nothing to say" case, not a bug to catch and hide.
    """
    store = store or VerifiedStore()
    results: list[SourceResult] = []
    used: list[str] = []
    skipped: list[str] = []

    try:
        results.append(rule_based_g2p(word, locale))
        used.append("rule_g2p")
    except G2PMiss as exc:
        skipped.append(f"rule_g2p: {exc}")

    try:
        results.append(neural_g2p(word, locale))
        used.append("neural_g2p")
    except (NeuralG2PUnavailable, G2PMiss) as exc:
        skipped.append(f"neural_g2p: {exc}")

    if use_llm:
        try:
            results.append(llm_phonetician(word, locale, context, client=llm_client))
            used.append("llm_phonetician")
        except LLMPhoneticianUnavailable as exc:
            skipped.append(f"llm_phonetician: {exc}")

    scored = score_candidates(results)
    verified = store.get(word, locale, identity_id=identity_id)

    if not scored and verified is None:
        raise RuntimeError(
            f"no pronunciation available for {word!r} ({locale}) — every source "
            "missed or was unavailable, and nothing is verified for it yet"
        )

    ranked = rank(scored, verified_entry=verified)

    return PipelineResult(
        word=word,
        locale=locale,
        ranked=ranked,
        sources_used=tuple(used),
        sources_skipped=tuple(skipped),
    )
