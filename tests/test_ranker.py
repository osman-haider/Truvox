from datetime import datetime, timedelta, timezone

import pytest

from truvox.ranker import rank
from truvox.schema import Provenance, ScoredCandidate, VerifiedEntry

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _candidate(ipa, score, provenance=Provenance.RULE_CMUDICT, agreement=0.8):
    return ScoredCandidate(
        ipa=ipa, score=score, agreement=agreement, contributors=(provenance,), top_provenance=provenance
    )


def _verified(ipa, days_ago=1, identity_id=None):
    return VerifiedEntry(
        word="test",
        locale="en-US",
        ipa=ipa,
        corrected_by="usman",
        identity_id=identity_id,
        corrected_at=NOW - timedelta(days=days_ago),
    )


def test_raises_when_nothing_to_rank():
    with pytest.raises(ValueError):
        rank([], verified_entry=None)


def test_no_verified_entry_returns_top_scored_candidate():
    candidates = [_candidate("aaa", 0.9), _candidate("bbb", 0.4)]
    result = rank(candidates, verified_entry=None, now=NOW)
    assert result.ipa == "aaa"
    assert not result.is_verified


def test_fresh_verified_entry_outranks_a_confident_ensemble_disagreement():
    candidates = [_candidate("wrong-guess", 0.95)]
    verified = _verified("verified-answer", days_ago=1)
    result = rank(candidates, verified_entry=verified, now=NOW)
    assert result.ipa == "verified-answer"
    assert result.is_verified


def test_identity_specific_entry_is_flagged_as_such():
    candidates = [_candidate("verified-answer", 0.5)]
    verified = _verified("verified-answer", days_ago=1, identity_id="patient-42")
    result = rank(candidates, verified_entry=verified, now=NOW)
    assert result.identity_specific is True


def test_generic_entry_is_not_flagged_identity_specific():
    candidates = [_candidate("verified-answer", 0.5)]
    verified = _verified("verified-answer", days_ago=1, identity_id=None)
    result = rank(candidates, verified_entry=verified, now=NOW)
    assert result.identity_specific is False


def test_very_old_identity_confirmation_still_wins_via_the_recency_floor():
    # Far past the half-life, so recency decays to the floor — an
    # identity-specific match (the strongest prior) should still win
    # against a fairly confident but unconfirmed ensemble disagreement.
    candidates = [_candidate("wrong-guess", 0.95)]
    verified = _verified("verified-answer", days_ago=365 * 20, identity_id="patient-1")
    result = rank(candidates, verified_entry=verified, now=NOW)
    assert result.ipa == "verified-answer"


def test_runner_up_is_reported_when_more_than_one_candidate():
    candidates = [_candidate("aaa", 0.9), _candidate("bbb", 0.4)]
    result = rank(candidates, verified_entry=None, now=NOW)
    assert result.runner_up is not None
    assert result.runner_up[0] == "bbb"


def test_single_candidate_has_no_runner_up():
    result = rank([_candidate("aaa", 0.9)], verified_entry=None, now=NOW)
    assert result.runner_up is None
