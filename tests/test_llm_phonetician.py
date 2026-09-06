import json

import pytest

from truvox.schema import Provenance
from truvox.sources.llm_phonetician import LLMPhoneticianUnavailable, llm_phonetician


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, text):
        self._text = text

    def create(self, **kwargs):
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def test_parses_well_formed_json_response():
    payload = {"ipa": "ˈkʌmələ", "confidence": 0.8, "reasoning": "common US pronunciation"}
    client = _FakeClient(json.dumps(payload))

    result = llm_phonetician("Kamala", locale="en-US", client=client)

    assert result.ipa == "ˈkʌmələ"
    assert result.provenance == Provenance.LLM_PHONETICIAN
    assert result.raw_confidence == 0.8
    assert "US pronunciation" in result.reasoning


def test_tolerates_chatty_preamble_around_the_json_block():
    payload = {"ipa": "hɛˈloʊ", "confidence": 0.6, "reasoning": "standard"}
    text = f"Sure, here you go:\n{json.dumps(payload)}\nLet me know if that helps!"
    client = _FakeClient(text)

    result = llm_phonetician("hello", client=client)
    assert result.ipa == "hɛˈloʊ"


def test_clamps_out_of_range_confidence():
    payload = {"ipa": "test", "confidence": 1.4, "reasoning": "overconfident"}
    client = _FakeClient(json.dumps(payload))

    result = llm_phonetician("test", client=client)
    assert result.raw_confidence == 1.0


def test_raises_clear_error_on_missing_ipa_field():
    client = _FakeClient(json.dumps({"confidence": 0.5}))
    with pytest.raises(LLMPhoneticianUnavailable):
        llm_phonetician("test", client=client)


def test_raises_clear_error_when_no_client_and_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMPhoneticianUnavailable):
        llm_phonetician("test")
