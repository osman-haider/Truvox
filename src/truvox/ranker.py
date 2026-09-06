"""The context-aware ranker: the step that turns "here's what the ensemble
thinks" into "here's what we're actually going to say," by folding in what
the verified store knows that the ensemble can't see on its own — a prior
confirmation for this exact identity, or for this word in this locale more
generally.

Framed explicitly as a small Bayesian update: posterior ∝ prior × likelihood.
The scorer's `ScoredCandidate.score` stands in for the likelihood — how well
the phonetic evidence supports a candidate. The prior comes from the
verified store: an identity-specific confirmation gets the strongest prior,
a locale-generic confirmation a moderate one, and anything unconfirmed a
neutral prior of 1.0. Priors are further discounted by how long ago the
confirmation happened — a decade-old correction still counts for something
(there's a floor), but a confirmation from last week counts for more than
one from three years ago, since pronunciation conventions and even a
person's own preferred pronunciation can drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from truvox.schema import Provenance, ScoredCandidate, VerifiedEntry

IDENTITY_MATCH_PRIOR = 8.0
LOCALE_GENERIC_MATCH_PRIOR = 3.0
NO_MATCH_PRIOR = 1.0

RECENCY_HALF_LIFE_DAYS = 365.0
MIN_RECENCY_WEIGHT = 0.3  # even a very old confirmation still outweighs a cold guess


@dataclass(frozen=True)
class RankedPronunciation:
    """The ranker's final answer for one word — what an MCP tool or a voice
    agent actually consumes."""

    ipa: str
    posterior: float  # 0..1, renormalized across every candidate considered
    is_verified: bool
    identity_specific: bool
    provenance: str
    runner_up: tuple[str, float] | None  # (ipa, posterior) — kept for transparency, not hidden


def _recency_weight(confirmed_at: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    age_days = max((now - confirmed_at).total_seconds() / 86400.0, 0.0)
    return max(math.exp(-age_days / RECENCY_HALF_LIFE_DAYS), MIN_RECENCY_WEIGHT)


def rank(
    candidates: list[ScoredCandidate],
    verified_entry: VerifiedEntry | None = None,
    now: datetime | None = None,
) -> RankedPronunciation:
    """Combine the scorer's ranked candidates with the verified store's prior
    into one final pronunciation. Raises ValueError only when there is
    nothing at all to rank — every source missed and nothing is verified.
    """
    candidate_scores: dict[str, float] = {c.ipa: max(c.score, 1e-6) for c in candidates}
    candidate_provenance: dict[str, Provenance] = {c.ipa: c.top_provenance for c in candidates}

    if not candidate_scores and verified_entry is None:
        raise ValueError("nothing to rank: no scored candidates and no verified entry")

    if verified_entry is not None:
        candidate_scores.setdefault(verified_entry.ipa, 0.5)  # the store knows it even if the ensemble didn't propose it

    priors = {ipa: NO_MATCH_PRIOR for ipa in candidate_scores}

    if verified_entry is not None:
        recency = _recency_weight(verified_entry.corrected_at, now)
        match_prior = IDENTITY_MATCH_PRIOR if verified_entry.identity_id else LOCALE_GENERIC_MATCH_PRIOR
        priors[verified_entry.ipa] = priors[verified_entry.ipa] * match_prior * recency

    posteriors_raw = {ipa: priors[ipa] * score for ipa, score in candidate_scores.items()}
    total = sum(posteriors_raw.values())
    posteriors = {ipa: value / total for ipa, value in posteriors_raw.items()}

    ranked_ipas = sorted(posteriors, key=posteriors.get, reverse=True)
    top_ipa = ranked_ipas[0]

    is_verified = verified_entry is not None and top_ipa == verified_entry.ipa
    top_provenance_enum = candidate_provenance.get(top_ipa)
    provenance = (
        "verified_store"
        if is_verified or top_provenance_enum is None
        else top_provenance_enum.value
    )

    runner_up = None
    if len(ranked_ipas) > 1:
        runner_up = (ranked_ipas[1], round(posteriors[ranked_ipas[1]], 4))

    return RankedPronunciation(
        ipa=top_ipa,
        posterior=round(posteriors[top_ipa], 4),
        is_verified=is_verified,
        identity_specific=bool(is_verified and verified_entry and verified_entry.identity_id),
        provenance=provenance,
        runner_up=runner_up,
    )
