#!/usr/bin/env python3
"""MCP tool wrapper: exposes the pipeline to any MCP-aware agent as two
tools — get a pronunciation, and record a correction — so a voice agent can
drop in Truvox pronunciation quality ahead of a TTS call with a few lines
of client config, instead of embedding pronunciation logic itself.

Try it locally:
    mcp dev src/truvox/mcp_server.py

Wire it into an agent by pointing an MCP client at this script (stdio
transport, the default `mcp.run()` below uses) the same way you would any
other local MCP server.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from truvox.pipeline import get_pronunciation as _get_pronunciation
from truvox.store import VerifiedStore

mcp = FastMCP("truvox")
_store = VerifiedStore()


@mcp.tool()
def get_pronunciation(word: str, locale: str = "en-US", context: str = "") -> dict:
    """Return the best-known IPA pronunciation for a word, with a posterior
    confidence and provenance — call this before handing `word` to a TTS
    engine, and pass the returned `ipa` through as a pronunciation-dictionary
    rule or an SSML <phoneme> override.
    """
    result = _get_pronunciation(word, locale=locale, context=context, store=_store)
    return {
        "word": result.word,
        "locale": result.locale,
        "ipa": result.ranked.ipa,
        "confidence": result.ranked.posterior,
        "provenance": result.ranked.provenance,
        "verified": result.ranked.is_verified,
    }


@mcp.tool()
def correct_pronunciation(
    word: str,
    locale: str,
    ipa: str,
    corrected_by: str,
    identity_id: str | None = None,
) -> dict:
    """Record a human-confirmed pronunciation so every future call for this
    word (and this identity, if given) reuses it instead of re-guessing.
    """
    entry = _store.correct(word, locale, ipa, corrected_by=corrected_by, identity_id=identity_id)
    return {
        "word": entry.word,
        "locale": entry.locale,
        "ipa": entry.ipa,
        "corrected_by": entry.corrected_by,
        "identity_id": entry.identity_id,
    }


if __name__ == "__main__":
    mcp.run()
