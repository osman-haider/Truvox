"""Thin TTS wrappers for the benchmark harness.

Each function takes an optional `ipa` — omit it for a baseline call (the
provider guesses from raw text, same as it would for any voice agent that
hasn't wired up a pronunciation layer), pass it to get the "corrected" call
that overrides pronunciation the way Truvox's ranked output is meant to be
used. Every function raises `TTSProviderUnavailable` (never a bare
ImportError or a confusing SDK exception) when its API key or SDK isn't
set up, so the harness can run a partial benchmark with whatever's
configured instead of failing outright.
"""
from __future__ import annotations

import os


class TTSProviderUnavailable(Exception):
    """Raised when a provider's API key or SDK isn't available. The harness
    treats this as "skip this provider for this word," not a fatal error.
    """


def elevenlabs_speak(
    text: str,
    voice_id: str,
    out_path: str,
    ipa: str | None = None,
) -> str:
    """Synthesize `text` via ElevenLabs. If `ipa` is given, creates a
    one-off pronunciation-dictionary rule for this exact string and applies
    it to the call — otherwise ElevenLabs pronounces the raw text itself.
    Requires ELEVENLABS_API_KEY.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise TTSProviderUnavailable("ELEVENLABS_API_KEY not set")
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as exc:
        raise TTSProviderUnavailable("pip install elevenlabs (see requirements.txt)") from exc

    client = ElevenLabs(api_key=api_key)
    dictionary_locators = None

    if ipa:
        dictionary = client.pronunciation_dictionaries.create_from_rules(
            name=f"truvox-{text}"[:50],
            rules=[
                {
                    "string_to_replace": text,
                    "type": "phoneme",
                    "phoneme": ipa,
                    "alphabet": "ipa",
                }
            ],
        )
        dictionary_locators = [
            {"pronunciation_dictionary_id": dictionary.id, "version_id": dictionary.version_id}
        ]

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        pronunciation_dictionary_locators=dictionary_locators,
    )
    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    return out_path


def azure_speak(
    text: str,
    out_path: str,
    voice: str = "en-US-JennyNeural",
    ipa: str | None = None,
) -> str:
    """Synthesize `text` via Azure Speech. If `ipa` is given, wraps it in an
    SSML <phoneme> tag; otherwise speaks the raw text. Requires
    AZURE_SPEECH_KEY and AZURE_SPEECH_REGION.
    """
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        raise TTSProviderUnavailable("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set")
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise TTSProviderUnavailable(
            "pip install azure-cognitiveservices-speech (see requirements.txt)"
        ) from exc

    body = f'<phoneme alphabet="ipa" ph="{ipa}">{text}</phoneme>' if ipa else text
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">{body}</voice>'
        "</speak>"
    )

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise TTSProviderUnavailable(f"Azure synthesis did not complete: {result.reason}")
    return out_path
