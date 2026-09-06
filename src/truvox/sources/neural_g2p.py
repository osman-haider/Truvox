"""Source 2: neural G2P, via DeepPhonemizer.

This is intentionally isolated behind lazy imports and a small wrapper
class: DeepPhonemizer pulls in torch, and its pretrained checkpoint
(~20-100MB depending on model) has to be downloaded once. Neither of those
should be a hard requirement to import `truvox` or run the rule-based +
LLM sources — an ensemble source being unavailable is a normal, handled
state, not a crash.

Stretch goal (see training/finetune_g2p.py): fine-tune a small checkpoint
on CMUdict + a WikiPron multilingual subset instead of using the stock
pretrained one. Point NeuralG2PSource at that checkpoint path once it
exists; nothing else in the pipeline needs to change.
"""
from __future__ import annotations

from functools import lru_cache

from truvox.schema import Provenance, SourceResult
from truvox.sources.rule_g2p import G2PMiss

# Public checkpoint DeepPhonemizer ships for US English CMUdict-style G2P.
# See https://github.com/as-ideas/DeepPhonemizer — swap this for a
# fine-tuned checkpoint path once training/finetune_g2p.py has produced one.
_DEFAULT_CHECKPOINT_URL = (
    "https://public-asai-dphon-assets.s3.eu-central-1.amazonaws.com/"
    "checkpoints/en_us_cmudict_ipa_forward.pt"
)


class NeuralG2PUnavailable(Exception):
    """Raised when torch/deep-phonemizer aren't installed, or the checkpoint
    can't be loaded. Callers should catch this the same way they catch
    G2PMiss — it means "skip this source for now", not "pipeline is broken".
    """


@lru_cache(maxsize=1)
def _load_phonemizer(checkpoint: str = _DEFAULT_CHECKPOINT_URL):
    try:
        from dp.phonemizer import Phonemizer  # deep-phonemizer
    except ImportError as exc:
        raise NeuralG2PUnavailable(
            "deep-phonemizer/torch aren't installed — this is an optional "
            "extra: pip install truvox[neural]. The ensemble runs fine "
            "without it (rule-based + LLM sources still contribute)."
        ) from exc

    try:
        return Phonemizer.from_checkpoint(checkpoint)
    except Exception as exc:  # noqa: BLE001 - network/IO errors, deliberately broad
        raise NeuralG2PUnavailable(
            f"could not load DeepPhonemizer checkpoint from {checkpoint!r}: {exc}"
        ) from exc


def neural_g2p(word: str, locale: str = "en_us", checkpoint: str | None = None) -> SourceResult:
    """Entry point the pipeline calls.

    Raises NeuralG2PUnavailable if the optional dependency chain isn't set
    up, or G2PMiss if the model loaded fine but produced nothing usable —
    the pipeline treats both the same way (source contributes nothing for
    this word), it just logs them under different reasons.
    """
    phonemizer = _load_phonemizer(checkpoint or _DEFAULT_CHECKPOINT_URL)

    # DeepPhonemizer's lang codes use underscores (en_us), not the
    # hyphenated BCP-47 style (en-US) the rest of truvox uses.
    dp_lang = locale.replace("-", "_").lower()

    try:
        ipa = phonemizer(word, lang=dp_lang)
    except Exception as exc:  # noqa: BLE001
        raise G2PMiss(f"DeepPhonemizer raised on {word!r}: {exc}") from exc

    if not ipa or not ipa.strip():
        raise G2PMiss(f"DeepPhonemizer returned empty output for {word!r}")

    return SourceResult(
        ipa=ipa,
        provenance=Provenance.NEURAL_G2P,
        raw_confidence=0.7,  # neural G2P generalizes to OOV but is less reliable than a curated lexicon
        word=word,
        locale=locale,
    )
