import pytest

from truvox.schema import Provenance
from truvox.sources.rule_g2p import G2PMiss, cmudict_g2p, rule_based_g2p


def test_cmudict_hit_on_common_word():
    result = cmudict_g2p("hello")
    assert result.provenance == Provenance.RULE_CMUDICT
    assert result.ipa  # non-empty
    assert all(ch not in "0123456789" for ch in result.ipa), "stress digits should be converted to IPA marks"


def test_cmudict_miss_on_invented_name():
    # Not a real word/name — should not be in CMUdict.
    with pytest.raises(G2PMiss):
        cmudict_g2p("zxqvorthblatt")


def test_rule_based_g2p_routes_non_english_to_epitran_and_reports_unavailable():
    # No epitran installed by default in this environment — should raise
    # G2PMiss (not crash with ImportError), same contract as a CMUdict miss.
    with pytest.raises(G2PMiss):
        rule_based_g2p("xochitl", locale="es-MX")


def test_result_rejects_out_of_range_confidence():
    from truvox.schema import SourceResult

    with pytest.raises(ValueError):
        SourceResult(
            ipa="test",
            provenance=Provenance.RULE_CMUDICT,
            raw_confidence=1.5,
            word="test",
            locale="en-US",
        )
