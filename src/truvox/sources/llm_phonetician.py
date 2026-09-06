"""Source 3: the LLM phonetician.

This is the fallback for names no lexicon has ever seen and no neural G2P
was trained on — exactly the case that breaks every production voice
agent today. The LLM is prompted for an IPA transcription *and* one line
of reasoning, and self-reports a confidence. That self-report is a weak
signal on its own (LLMs are not well calibrated) — the scorer treats it as
one input among several, not as ground truth.

The client is dependency-injected so tests never need a real API key or
network call: pass any object exposing `.messages.create(...)` shaped like
the Anthropic SDK's response (a minimal fake is enough — see
tests/test_llm_phonetician.py).
"""
from __future__ import annotations

import json
import os
import re

from truvox.schema import Provenance, SourceResult

_DEFAULT_MODEL = os.environ.get("TRUVOX_LLM_MODEL", "claude-haiku-4-5-20251001")

_PROMPT_TEMPLATE = """You are a phonetician. Give the IPA (International \
Phonetic Alphabet) pronunciation for the word below.

Word: {word}
Locale / language context: {locale}
Additional context: {context}

Reply with ONLY a JSON object, no other text, in exactly this shape:
{{"ipa": "<IPA transcription, no slashes or brackets>", \
"confidence": <float 0.0-1.0, your honest confidence in this transcription>, \
"reasoning": "<one sentence: why this pronunciation, and what you're unsure about>"}}"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMPhoneticianUnavailable(Exception):
    """Raised when no client was supplied and ANTHROPIC_API_KEY isn't set,
    or the model response couldn't be parsed. The pipeline treats this like
    any other source-unavailable case.
    """


def _build_default_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMPhoneticianUnavailable(
            "no client was passed and ANTHROPIC_API_KEY is not set — "
            "copy .env.example to .env and fill it in, or pass an "
            "injected client for testing"
        )
    try:
        import anthropic
    except ImportError as exc:
        raise LLMPhoneticianUnavailable(
            "the 'anthropic' package isn't installed — "
            "run `pip install -r requirements.txt`"
        ) from exc
    return anthropic.Anthropic()


def _extract_text(response) -> str:
    """Pull the text out of an Anthropic-SDK-shaped response. Kept as its
    own function so a test double only has to satisfy this one contract:
    `response.content[0].text`.
    """
    try:
        return response.content[0].text
    except (AttributeError, IndexError) as exc:
        raise LLMPhoneticianUnavailable(
            f"unexpected response shape from LLM client: {response!r}"
        ) from exc


def _parse_json_payload(raw_text: str) -> dict:
    match = _JSON_BLOCK.search(raw_text)
    if not match:
        raise LLMPhoneticianUnavailable(
            f"no JSON object found in LLM response: {raw_text!r}"
        )
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMPhoneticianUnavailable(
            f"LLM response was not valid JSON: {raw_text!r}"
        ) from exc

    if "ipa" not in payload or not str(payload["ipa"]).strip():
        raise LLMPhoneticianUnavailable(f"LLM response missing 'ipa': {payload!r}")
    return payload


def llm_phonetician(
    word: str,
    locale: str = "en-US",
    context: str = "",
    client=None,
    model: str = _DEFAULT_MODEL,
) -> SourceResult:
    """Entry point the pipeline calls."""
    client = client or _build_default_client()

    prompt = _PROMPT_TEMPLATE.format(
        word=word, locale=locale, context=context or "none given"
    )
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    payload = _parse_json_payload(_extract_text(response))
    confidence = float(payload.get("confidence", 0.5))
    confidence = min(max(confidence, 0.0), 1.0)  # LLM self-reports aren't always well-behaved

    return SourceResult(
        ipa=str(payload["ipa"]).strip(),
        provenance=Provenance.LLM_PHONETICIAN,
        raw_confidence=confidence,
        word=word,
        locale=locale,
        reasoning=payload.get("reasoning"),
    )
