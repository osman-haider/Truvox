# Truvox

A multi-source pronunciation confidence engine: an ensemble of G2P sources
reconciled into a scored, provenance-tagged pronunciation, a store that
persists human corrections, a context-aware ranker, and an MCP tool that
puts the whole thing in front of a voice agent before it calls TTS.

Built as a portfolio project for NameCoach's Founding Voice AI Engineer
role — scoped as a small, honest version of the exact system their job
post describes (see "Why it's built this way" below).

## Architecture

```
name + locale + context
        │
        ├──► rule-based G2P (CMUdict / Epitran)   ─┐
        ├──► neural G2P (DeepPhonemizer)            ├──► confidence scorer ◄──► verified store
        └──► LLM phonetician (prompted)            ─┘         │
                                                                ▼
                                                        context ranker  (planned)
                                                                │
                                                                ▼
                                                    MCP tool → voice agent → TTS  (planned)
```

## Status

**Built:** the three-source ensemble, the confidence scorer, and the
verified store. **Not yet built:** the context-aware ranker, the MCP tool
wrapper, and the TTS benchmark. See `training/` for the stretch goal (a
fine-tune scaffold, not a finished model).

Tests are written for every module (`tests/`) but have **not been
executed from this environment** — the sandbox this was built in hit
infrastructure issues partway through the session. Run them yourself
before trusting the claims above:

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest -v
```

If something fails, it's a real bug to fix, not a formality — treat a
clean `pytest` run as part of "done," not this README.

## Setup

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY at minimum
pip install -r requirements.txt
pytest -v
python scripts/demo.py "Saoirse" --locale en-IE --no-llm
```

`--no-llm` skips the LLM phonetician source so you can try the demo before
setting up an API key — the rule-based source alone is enough to see the
scorer and store work.

### Optional extras

```bash
pip install -e ".[multilingual]"   # Epitran — rule-based G2P for ~100 non-English locales
pip install -e ".[neural]"         # DeepPhonemizer + torch — the neural G2P source
pip install -e ".[train]"          # wandb — for the stretch fine-tune
```

Nothing above is required to run the core pipeline. Sources that aren't
installed report themselves as unavailable and are skipped — the ensemble
degrades gracefully rather than crashing.

## Layout

```
src/truvox/
  schema.py              shared types: SourceResult, ScoredCandidate, VerifiedEntry
  sources/
    rule_g2p.py           Source 1 — CMUdict (primary) / Epitran (optional, multilingual)
    neural_g2p.py          Source 2 — DeepPhonemizer (optional extra)
    llm_phonetician.py     Source 3 — prompted LLM, dependency-injected client
  scorer.py               confidence scorer: phoneme-distance + reliability + agreement
  store.py                SQLite verified-pronunciation store (get / correct)
data/
  stress_test.csv         50-word stress set — draft IPA, all flagged needs_human_check
tests/                    one test file per module above
scripts/demo.py           CLI: run the ensemble on one word
training/                 G2P fine-tune scaffold, stretch goal (CMUdict + WikiPron → W&B)
```

## An honest caveat on the data

`data/stress_test.csv` ships with a `draft_ipa` column populated from
general phonetic reference knowledge, not from listening to native-speaker
audio. Every row is marked `needs_human_check` on purpose. Verifying a
chunk of these against real audio (Forvo, Wiktionary, YouGlish) and
updating `verification_status` / `verified_by` is real work, not
busywork — it's the difference between a pipeline that *looks* like it
handles pronunciation and one that's actually been checked against reality.

## Why it's built this way

Every module above maps to a specific line in NameCoach's job post:
confidence scores + model provenance → the scorer; verified entries that
persist and get reused → the store; context-aware ranking → the ranker
(planned); MCP-style integration → the tool wrapper (planned). That
mapping is deliberate, not a coincidence.
