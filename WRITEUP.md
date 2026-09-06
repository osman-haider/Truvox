# Why voice agents keep getting names wrong, and what a fix looks like

Voice agents keep shipping the same failure mode: the model is fluent
right up until it hits a name, a drug, or a place it hasn't seen before,
and then it guesses — confidently, and often wrong. A patient-relations
bot mangles "Atorvastatin." A university enrollment assistant butchers
"Saoirse." The rest of the interaction can be flawless; the mispronounced
word is what the person remembers.

This is a measurement problem as much as a modeling one. There's no
widely-used benchmark for pronunciation reliability across TTS providers,
locales, and word categories — which makes it hard to even state, in a
falsifiable way, how bad the problem is or whether a given fix helps.
Truvox is a small attempt at both halves: a working system that produces a
pronunciation with a confidence score instead of a silent guess, and the
start of a benchmark harness to measure whether that confidence is
earned.

## The shape of the fix

No single pronunciation source is trustworthy across the board. A
rule-based lexicon (CMUdict, Epitran) is precise when a word is in it and
useless the moment it isn't — which is exactly the failure mode this
project cares about, since the hard names are the ones no lexicon has
ever seen. A neural G2P model generalizes better to unseen words but is
less reliable than a curated entry. An LLM prompted for IPA covers the
long tail no lexicon or model has training data for, but self-reports
confidence that isn't well calibrated.

So Truvox doesn't pick one. It runs all three, reconciles disagreement
with a confidence scorer that weighs phonetic distance between candidates
against how much each source is trusted, and — critically — remembers.
A human correction, once made, is stored and treated as a strong prior:
the next time that name comes up, for that person or in that locale, the
system doesn't re-guess. A context-aware ranker folds that prior in
explicitly as a Bayesian update (posterior ∝ prior × likelihood), so a
fresh identity-specific confirmation outweighs a confident but
unconfirmed ensemble guess, and an old confirmation still counts for
something without dominating forever.

The whole pipeline is exposed as a single MCP tool, `get_pronunciation`,
so a voice agent calls it the same way it'd call any other tool — pass a
word and a locale, get back an IPA transcription, a confidence score, and
which source or verified entry it came from — before the TTS call ever
happens.

## A finding worth stating plainly

Not every TTS provider can even take a correction. ElevenLabs supports
pronunciation dictionaries in IPA or CMU-phoneme form via its API. Azure
and Google Cloud TTS support full SSML, including `<phoneme>` tags. OpenAI's
TTS API supports neither — no SSML, no phoneme-level override of any
kind; pacing and tone are steerable through punctuation and natural-language
instructions, but pronunciation itself is not. That's not a criticism of
OpenAI's product so much as a precise statement of the gap this kind of
pronunciation layer exists to fill: for a growing share of the TTS
ecosystem, there is currently no lower-level hook to correct a
mispronunciation at all, which makes an upstream layer that gets the
input right the first time more valuable, not less.

## What's actually measured here, and what isn't yet

Being precise about this matters more than the numbers themselves would.
The stress-test set (`data/stress_test.csv`) — fifty names, drug names,
places, and contested pronunciations spanning a dozen-plus languages — ships
with a reference IPA transcription drawn from general phonetic knowledge,
not from listening to native-speaker audio. Every row is flagged
`needs_human_check`. The benchmark harness (`benchmark/run_benchmark.py`)
is built to synthesize each word with and without the pipeline's
corrected pronunciation, across whichever real TTS providers have API keys
configured, and produce a scoring template — but scoring a baseline
against a corrected clip by ear is a human judgment this code
deliberately doesn't automate away. Reporting a mispronunciation rate
before that listening pass has actually happened would be exactly the
kind of unearned confidence this whole project argues against.

## What's next

A fine-tuned G2P model trained on CMUdict plus WikiPron's multilingual
word/IPA pairs, evaluated against the LLM phonetician and the rule-based
baseline on held-out words. A completed listening pass across the
stress set, with real accuracy numbers attached to real audio. And,
longer term, exactly the kind of public benchmark and leaderboard the
field currently lacks — not because no one could build one, but because
it takes sustained, unglamorous verification work that's easy to skip and
easy to notice when it's been skipped.
