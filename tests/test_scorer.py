from truvox.schema import Provenance, SourceResult
from truvox.scorer import phoneme_distance, score_candidates


def _result(ipa, provenance, confidence=0.8, word="test", locale="en-US"):
    return SourceResult(
        ipa=ipa, provenance=provenance, raw_confidence=confidence, word=word, locale=locale
    )


def test_empty_input_returns_empty_output():
    assert score_candidates([]) == []


def test_majority_agreement_wins_over_a_lone_dissenter():
    results = [
        _result("hɛˈloʊ", Provenance.RULE_CMUDICT, confidence=0.9),
        _result("hɛˈloʊ", Provenance.NEURAL_G2P, confidence=0.7),
        _result("hɛˈlaʊ", Provenance.LLM_PHONETICIAN, confidence=0.95),  # confident, but alone
    ]
    ranked = score_candidates(results)
    assert ranked[0].ipa == "hɛˈloʊ"
    assert ranked[0].score > ranked[1].score
    assert set(ranked[0].contributors) == {Provenance.RULE_CMUDICT, Provenance.NEURAL_G2P}


def test_single_source_gets_neutral_agreement_not_zero():
    ranked = score_candidates([_result("sɪŋˈɡəl", Provenance.LLM_PHONETICIAN)])
    assert len(ranked) == 1
    assert ranked[0].agreement == 0.5


def test_verified_store_provenance_outranks_everything_else_given_equal_agreement():
    results = [
        _result("aɪˈpiːeɪ", Provenance.VERIFIED_STORE, confidence=1.0),
        _result("aɪˈpiːeɪ", Provenance.LLM_PHONETICIAN, confidence=0.5),
    ]
    ranked = score_candidates(results)
    # both proposed the identical ipa, so they're one cluster — top_provenance
    # should be the more reliable of the two contributors.
    assert ranked[0].top_provenance == Provenance.VERIFIED_STORE


def test_phoneme_distance_is_zero_for_identical_strings():
    assert phoneme_distance("hɛˈloʊ", "hɛˈloʊ") == 0.0


def test_phoneme_distance_is_positive_for_different_strings():
    assert phoneme_distance("hɛˈloʊ", "wɜːrld") > 0.0
