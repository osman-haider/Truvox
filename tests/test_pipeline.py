import pytest

from truvox.pipeline import get_pronunciation
from truvox.store import VerifiedStore


def test_pipeline_runs_end_to_end_on_a_common_word(tmp_path):
    store = VerifiedStore(tmp_path / "store.db")
    result = get_pronunciation("hello", locale="en-US", store=store, use_llm=False)

    assert result.ranked.ipa
    assert "rule_g2p" in result.sources_used


def test_a_verified_entry_takes_priority_over_the_ensemble(tmp_path):
    store = VerifiedStore(tmp_path / "store.db")
    store.correct("hello", "en-US", "custom-verified-ipa", corrected_by="usman")

    result = get_pronunciation("hello", locale="en-US", store=store, use_llm=False)

    assert result.ranked.ipa == "custom-verified-ipa"
    assert result.ranked.is_verified


def test_raises_when_every_source_misses_and_nothing_is_verified(tmp_path):
    store = VerifiedStore(tmp_path / "store.db")
    with pytest.raises(RuntimeError):
        get_pronunciation("zxqvorthblatt", locale="en-US", store=store, use_llm=False)
