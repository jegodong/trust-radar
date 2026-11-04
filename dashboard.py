"""trust_radar.py -- single-question pipeline behind the dashboard.

All confidence and consistency maths lives in radar_core.py so that the
dashboard and the batch evaluation (trust_eval.py) can never disagree.

Public functions
----------------
    analyze(model, prompt)               one greedy answer + per-entity scores
    multi_sample_check(model, prompt, n) N sampled answers + one verdict
"""

import json
import math
import os

import ollama
import spacy

import radar_core as core

DEFAULT_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

nlp = spacy.load("en_core_web_sm")

FACTUAL_ENTITIES = {
    "PERSON", "GPE", "LOC", "ORG", "DATE",
    "CARDINAL", "MONEY", "PERCENT", "QUANTITY",
    "EVENT", "FAC", "NORP", "LAW",
}

MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def is_factual_date(text: str) -> bool:
    """Keep "1912" and "March 2020"; drop "yesterday" and "last week"."""
    if any(ch.isdigit() for ch in text):
        return True
    low = text.lower()
    return any(month in low for month in MONTHS)


def analyze(model: str, prompt: str) -> dict:
    """One deterministic answer, with every factual entity scored."""
    response = ollama.generate(model=model, prompt=prompt, logprobs=True, top_logprobs=3)
    text = response["response"]

    tokens = []
    for item in response.get("logprobs") or []:
        if item.get("logprob") is None:
            continue
        tokens.append({"token": item.get("token", ""), "p": math.exp(float(item["logprob"]))})

    spans, status = core.align_tokens(text, tokens)
    if spans is None:
        return {"text": text, "entities": [], "alignment": status}

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ not in FACTUAL_ENTITIES:
            continue
        if ent.label_ == "DATE" and not is_factual_date(ent.text):
            continue
        probs = [p for start, end, p in spans if start < ent.end_char and end > ent.start_char]
        if not probs:
            continue
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": int(ent.start_char),
            "end": int(ent.end_char),
            "avg": sum(probs) / len(probs),
            "min": min(probs),
            "tokens": [text[start:end] for start, end, _ in spans
                       if start < ent.end_char and end > ent.start_char],
        })

    return {"text": text, "entities": entities, "alignment": status}


def get_embedder():
    return core.get_embedder()


def compute_sindex(answers, distance_threshold=None) -> dict:
    """
    Semantic inconsistency of a bag of answers.

    Thin wrapper over radar_core.consistency_report so the dashboard and
    results.csv report the same score.  The old version mixed cluster
    coherence into the entropy and never normalised the result, which made a
    50/50 split score only 0.69 - below the "high" threshold of 0.8.
    """
    report = core.consistency_report(answers, embedder=core.get_embedder(),
                                     threshold=distance_threshold)
    return {
        "sindex": report["sindex"],
        "n_clusters": report["n_clusters_sem"],
        "clusters": report["clusters"],
        "level": report["level"],
        "verdict": report["verdict"],
    }


def multi_sample_check(model: str, prompt: str, n: int = 5) -> dict:
    """Sample the same question n times and describe how much the answers vary."""
    from collections import Counter

    # No JSON/grammar constraint here: constrained decoding changes the answer
    # distribution, and trust_eval.py samples without it.  Same generation
    # settings in both files keeps the two measurements comparable.
    short_prompt = f"{prompt}\n\nAnswer with only the direct entity."
    answers = []
    for i in range(n):
        try:
            resp = ollama.generate(
                model=model,
                prompt=short_prompt,
                options={"temperature": 0.8, "seed": 42 + 17 * i, "num_predict": 32},
            )
            raw = (resp.get("response") or "").strip().split("\n")[0].strip()
            answers.append({"raw": raw, "key": core.normalize_answer(raw)})
        except Exception:
            answers.append({"raw": "", "key": ""})

    report = core.consistency_report([a["raw"] for a in answers])

    counter = Counter(a["key"] for a in answers if a["key"])
    top_key = counter.most_common(1)[0][0] if counter else ""
    top_raw = next((a["raw"] for a in answers if a["key"] == top_key), top_key)

    return {
        "samples": [a["raw"] for a in answers],
        # exact-string agreement: a different question from SINdex, reported
        # next to it, not turned into a second competing verdict
        "consistency": report["consistency_exact"],
        "unique_count": report["n_unique"],
        "top_entity": top_raw,
        # one verdict, produced by radar_core
        "verdict": report["verdict"],
        "level": report["level"],
        "sindex": report["sindex"],
        "sindex_n_clusters": report["n_clusters_sem"],
        "sindex_clusters": report["clusters"],
        "sindex_verdict": report["verdict"],
        "sindex_level": report["level"],
    }
