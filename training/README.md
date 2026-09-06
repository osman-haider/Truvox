# Stretch goal: fine-tuning a small G2P model

This is the "if the schedule is holding" item from the build plan. It is a
scaffold, not a finished, benchmarked model — the honest goal here is
a training run that's *started* and logged, not a checkpoint you'd ship.

## What it fine-tunes

A small ByT5 (or DeepPhonemizer's own transformer) checkpoint, on:

- **CMUdict** — ~130k English word → ARPAbet pairs (converted to IPA with
  the same mapping table `src/truvox/sources/rule_g2p.py` uses, so the
  fine-tuned model and the rule-based source are trained on a consistent
  IPA convention).
- **WikiPron** (https://github.com/CUNY-CL/wikipron) — a real, citable,
  multilingual word → IPA dataset scraped from Wiktionary pronunciation
  entries, covering dozens of languages. This is the dataset to reach for
  instead of hand-collecting multilingual pairs from scratch.

## Why this isn't wired to auto-run

Both datasets need a first-time download, and a real fine-tune needs GPU
time this environment doesn't have guaranteed access to. `finetune_g2p.py`
is written to be correct and runnable on a machine with the data and a GPU
(or a few hours of patience on CPU for the ByT5-small config) — not to
silently no-op if you run it here.

## What "done" looks like for the stretch goal

Not a state-of-the-art G2P model. A Weights & Biases run comparing:
rule-based (CMUdict) vs. this fine-tune vs. the LLM phonetician, on a
held-out slice of the stress-test set — one chart, linked from the README.
That comparison, even with a middling fine-tune, is a genuine data point
for the write-up: does a small fine-tuned model beat prompting an LLM for
IPA? Worth knowing either way.
