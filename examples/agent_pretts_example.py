#!/usr/bin/env python3
"""How a voice agent uses Truvox ahead of a TTS call — the whole point being
that the agent never lets the TTS engine guess a name cold."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truvox.pipeline import get_pronunciation

result = get_pronunciation("Saoirse", locale="en-IE", context="student roster", use_llm=False)
ipa = result.ranked.ipa

# Hand `ipa` to whichever TTS call comes next — an ElevenLabs pronunciation
# dictionary rule, or an SSML <phoneme alphabet="ipa" ph="...">  tag — instead
# of letting the engine guess from the raw text "Saoirse".
print(f"speak 'Saoirse' as /{ipa}/ "
      f"(confidence {result.ranked.posterior:.2f}, source: {result.ranked.provenance})")
