"""Shared data types passed between every stage of the pipeline.

Every source returns a `SourceResult`, never a bare string — the whole
point of the ensemble is that we keep provenance and a confidence signal
attached to a pronunciation all the way through scoring, storage, and
ranking. Losing that at any stage is the bug this schema exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Provenance(str, Enum):
    """Which source produced a candidate. Kept as an explicit enum (not a
    free string) so the scorer's reliability weights can't silently typo
    against the sources that emit them."""

    RULE_CMUDICT = "rule_cmudict"
    RULE_EPITRAN = "rule_epitran"
    NEURAL_G2P = "neural_g2p"
    LLM_PHONETICIAN = "llm_phonetician"
    VERIFIED_STORE = "verified_store"


@dataclass(frozen=True)
class SourceResult:
    """One candidate pronunciation from one source, before scoring."""

    ipa: str
    provenance: Provenance
    raw_confidence: float  # 0..1, self-reported by the source; not comparable across sources
    word: str
    locale: str
    reasoning: str | None = None  # only the LLM source fills this in

    def __post_init__(self) -> None:
        if not 0.0 <= self.raw_confidence <= 1.0:
            raise ValueError(
                f"raw_confidence must be in [0, 1], got {self.raw_confidence!r}"
            )
        if not self.ipa.strip():
            raise ValueError("ipa must be non-empty")


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate after the confidence scorer has reconciled the ensemble."""

    ipa: str
    score: float  # 0..1, the number the ranker and MCP tool actually consume
    agreement: float  # 0..1, how much the other sources corroborated this ipa
    contributors: tuple[Provenance, ...]  # every source that proposed this exact ipa
    top_provenance: Provenance  # the single most-trusted source that proposed it


@dataclass(frozen=True)
class VerifiedEntry:
    """A human-confirmed pronunciation, persisted in the verified store."""

    word: str
    locale: str
    ipa: str
    corrected_by: str
    identity_id: str | None = None
    notes: str = ""
    corrected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
