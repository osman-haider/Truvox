"""Truvox — a multi-source pronunciation confidence engine.

Pipeline: three independent G2P sources (rule-based, neural, LLM) each
produce a candidate IPA transcription with a provenance tag. The confidence
scorer reconciles them into a ranked list. The verified store persists
human corrections so the same name never has to be re-guessed twice. A
context-aware ranker and an MCP tool sit on top next.
"""

__version__ = "0.1.0"
