"""Source 1: rule-based G2P.

Primary path is CMUdict via `pronouncing` — deterministic, offline, and a
useful *negative* signal: CMUdict is exactly the kind of closed lexicon that
breaks on the names this project cares about, so an OOV miss here is not a
bug, it's the reason sources 2 and 3 exist.

Optional secondary path: Epitran, for rule-based G2P on ~100 non-English
languages. It's an optional extra (`pip install truvox[multilingual]`)
because several of its language packs need system data files (flite/festival)
that aren't worth forcing on everyone who just wants to run the English
demo. Import is guarded — the source reports itself unavailable rather than
crashing the pipeline if it isn't installed.
"""
from __future__ import annotations

from truvox.schema import Provenance, SourceResult

# ARPAbet (CMUdict) -> IPA. This is the standard simplified mapping used by
# most CMUdict-to-IPA converters: it strips CMUdict's stress digits off each
# vowel and re-attaches them as IPA stress marks (ˈ primary, ˌ secondary)
# immediately before the phoneme they were attached to. It is *not* a real
# syllabifier — CMUdict doesn't give us syllable boundaries, only per-phone
# stress — so multi-syllable stress placement is an approximation. Good
# enough for a rule-based baseline; call this out if it ever matters in the
# benchmark write-up.
_ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}

_STRESS_MARK = {"1": "ˈ", "2": "ˌ", "0": ""}


class G2PMiss(Exception):
    """Raised when a source has no candidate at all for this word/locale.

    The pipeline catches this per-source and simply omits that source from
    the ensemble for this word — a miss from one source is expected and
    handled, not a pipeline failure.
    """


def _arpabet_token_to_ipa(token: str) -> str:
    phoneme = token.rstrip("012")
    stress_digit = token[len(phoneme):] or "0"
    ipa = _ARPABET_TO_IPA.get(phoneme)
    if ipa is None:
        raise G2PMiss(f"unmapped ARPAbet phoneme: {token!r}")
    return _STRESS_MARK.get(stress_digit, "") + ipa


def cmudict_g2p(word: str, locale: str = "en-US") -> SourceResult:
    """Look up `word` in CMUdict and convert its top pronunciation to IPA.

    Raises G2PMiss if the word isn't in CMUdict (case-insensitive, and
    `pronouncing` also fails on words containing anything but letters and
    a handful of punctuation marks CMUdict itself uses).
    """
    try:
        import pronouncing
    except ImportError as exc:  # pragma: no cover - exercised via requirements
        raise G2PMiss(
            "the 'pronouncing' package isn't installed — "
            "run `pip install -r requirements.txt`"
        ) from exc

    phones_options = pronouncing.phones_for_word(word.lower())
    if not phones_options:
        raise G2PMiss(f"{word!r} is not in CMUdict")

    # pronouncing returns every listed pronunciation variant; take the first
    # (CMUdict lists the most common pronunciation first by convention).
    tokens = phones_options[0].split()
    ipa = "".join(_arpabet_token_to_ipa(t) for t in tokens)

    return SourceResult(
        ipa=ipa,
        provenance=Provenance.RULE_CMUDICT,
        raw_confidence=0.9,  # CMUdict entries are human-curated; high trust *when present*
        word=word,
        locale=locale,
    )


def epitran_g2p(word: str, locale: str) -> SourceResult:
    """Rule-based G2P for non-English locales via Epitran.

    `locale` must be an Epitran language code, e.g. "spa-Latn" (Spanish),
    "vie-Latn" (Vietnamese). See https://github.com/dmort27/epitran for the
    full table. Raises G2PMiss (not ImportError) if epitran isn't installed,
    so callers can treat every source uniformly.
    """
    try:
        import epitran
    except ImportError as exc:
        raise G2PMiss(
            "epitran isn't installed — this is an optional extra: "
            "pip install truvox[multilingual]"
        ) from exc

    epi = epitran.Epitran(locale)
    ipa = epi.transliterate(word)
    if not ipa.strip():
        raise G2PMiss(f"epitran produced no output for {word!r} in {locale!r}")

    return SourceResult(
        ipa=ipa,
        provenance=Provenance.RULE_EPITRAN,
        raw_confidence=0.75,  # rule-based transliteration, not a curated lexicon
        word=word,
        locale=locale,
    )


def rule_based_g2p(word: str, locale: str = "en-US") -> SourceResult:
    """Entry point the pipeline calls. English locales try CMUdict first;
    everything else goes straight to Epitran if it's installed.
    """
    if locale.startswith("en"):
        return cmudict_g2p(word, locale)
    return epitran_g2p(word, locale)
