"""The confidence scorer: reconciles disagreement across the ensemble into
one ranked, scored, provenance-tagged list of candidates.

Design, in plain terms: candidates that (a) come from a source we trust
more, (b) roughly agree with what the *other* sources said, and (c) the
source itself was confident about, score higher. None of those three
signals is trustworthy alone — a confident LLM can be confidently wrong,
a lone rule-based hit can't be cross-checked, and reliability priors are
just priors — which is the actual argument for combining all three instead
of picking one.

Distance metric: panphon's phonological-feature edit distance when it's
installed (it treats /p/ and /b/ as close and /p/ and /a/ as far, which a
plain string edit distance can't). Falls back to a plain Levenshtein
distance over the IPA string otherwise — coarser, but keeps the scorer
usable with zero extra dependencies installed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from truvox.schema import Provenance, ScoredCandidate, SourceResult

# Reliability priors: how much we trust a source *when it has an opinion*,
# independent of what it says this time. Calibrated by hand for now — the
# honest next step (see training/finetune_g2p.py) is fitting these against
# the benchmark's human-scored outcomes instead of eyeballing them.
RELIABILITY_PRIOR: dict[Provenance, float] = {
    Provenance.VERIFIED_STORE: 1.0,
    Provenance.RULE_CMUDICT: 0.90,
    Provenance.RULE_EPITRAN: 0.75,
    Provenance.NEURAL_G2P: 0.75,
    Provenance.LLM_PHONETICIAN: 0.60,
}

# Scorer weights. Must sum to 1.0 — asserted below so a future edit that
# forgets to rebalance them fails loudly instead of silently drifting.
_W_RELIABILITY = 0.45
_W_AGREEMENT = 0.35
_W_RAW_CONFIDENCE = 0.20
assert abs((_W_RELIABILITY + _W_AGREEMENT + _W_RAW_CONFIDENCE) - 1.0) < 1e-9

_NO_CORROBORATION_AGREEMENT = 0.5  # neutral score when a candidate can't be cross-checked


@lru_cache(maxsize=1)
def _panphon_distance():
    """Returns a callable(ipa_a, ipa_b) -> float, or None if panphon isn't
    installed. Cached so we only pay the (small) import + data-load cost
    once per process.
    """
    try:
        from panphon.distance import Distance
    except ImportError:
        return None
    dist = Distance()
    return dist.weighted_feature_edit_distance


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def phoneme_distance(ipa_a: str, ipa_b: str) -> float:
    """Distance between two IPA strings — lower is more similar. Uses
    panphon's feature-weighted edit distance if available, else a plain
    Levenshtein distance over the raw IPA string as a coarser fallback.
    """
    weighted = _panphon_distance()
    if weighted is not None:
        try:
            return float(weighted(ipa_a, ipa_b))
        except Exception:  # noqa: BLE001 - panphon can choke on unrecognized symbols
            pass
    return float(_levenshtein(ipa_a, ipa_b))


def _similarity(distance: float, ipa_a: str, ipa_b: str) -> float:
    """Turn a raw distance into a 0..1 similarity, normalized by transcription
    length so a 1-phoneme difference on a long word doesn't score the same
    as a 1-phoneme difference on a 2-phoneme word.
    """
    scale = max(len(ipa_a), len(ipa_b), 1)
    return max(0.0, 1.0 - (distance / scale))


@dataclass
class _Cluster:
    ipa: str
    members: list[SourceResult]


def _cluster_by_exact_ipa(results: list[SourceResult]) -> list[_Cluster]:
    """Group results that proposed the *identical* IPA string. Deliberately
    conservative — merging near-misses ("close enough") is a real
    improvement but needs a tuned threshold to avoid false-merging two
    genuinely different pronunciations. Exact-match clustering is the
    honest v1; the agreement score below still rewards near-misses even
    when they don't get merged.
    """
    buckets: dict[str, list[SourceResult]] = defaultdict(list)
    for r in results:
        buckets[r.ipa].append(r)
    return [_Cluster(ipa=ipa, members=members) for ipa, members in buckets.items()]


def score_candidates(results: list[SourceResult]) -> list[ScoredCandidate]:
    """Turn raw per-source results for one word into a ranked list of
    ScoredCandidate, highest score first. Empty input returns an empty list
    rather than raising — the pipeline is responsible for deciding what an
    empty ensemble means (all sources missed).
    """
    if not results:
        return []

    clusters = _cluster_by_exact_ipa(results)
    scored: list[ScoredCandidate] = []

    for cluster in clusters:
        others = [r for r in results if r.ipa != cluster.ipa]

        if others:
            sims = [
                _similarity(phoneme_distance(cluster.ipa, o.ipa), cluster.ipa, o.ipa)
                * RELIABILITY_PRIOR.get(o.provenance, 0.5)
                for o in others
            ]
            weights = [RELIABILITY_PRIOR.get(o.provenance, 0.5) for o in others]
            agreement = sum(sims) / sum(weights) if sum(weights) > 0 else _NO_CORROBORATION_AGREEMENT
        else:
            agreement = _NO_CORROBORATION_AGREEMENT

        best_reliability = max(
            RELIABILITY_PRIOR.get(m.provenance, 0.5) for m in cluster.members
        )
        avg_raw_confidence = sum(m.raw_confidence for m in cluster.members) / len(cluster.members)

        score = (
            _W_RELIABILITY * best_reliability
            + _W_AGREEMENT * agreement
            + _W_RAW_CONFIDENCE * avg_raw_confidence
        )

        top_provenance = max(
            cluster.members, key=lambda m: RELIABILITY_PRIOR.get(m.provenance, 0.5)
        ).provenance

        scored.append(
            ScoredCandidate(
                ipa=cluster.ipa,
                score=round(min(max(score, 0.0), 1.0), 4),
                agreement=round(agreement, 4),
                contributors=tuple(m.provenance for m in cluster.members),
                top_provenance=top_provenance,
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored
