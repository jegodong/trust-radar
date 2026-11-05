"""trust_radar.py -- the full pipeline for one question.

    report = analyze("Who painted the Mona Lisa?", model="qwen2.5:3b")
    report.decision      -> "PASS" | "VERIFY" | "ABSTAIN"
    report.final_answer  -> what the user should be shown

Steps
-----
1. Answer      one deterministic answer (temperature 0) with token logprobs
2. Locate      spaCy entities in that answer, minus words copied from the
               question; plain content words if spaCy finds nothing
3. Score       weakest-token confidence per entity + what the model nearly
               said instead (top-k alternatives)
4. Resample    N answers at temperature > 0, same prompt, then count how many
               agree with the shown answer
5. Fuse        risk = 1 - (0.4 * confidence + 0.6 * support)
6. Act         PASS shows the answer, VERIFY checks it against the local
               knowledge base / Wikipedia and lets the model re-answer from
               the evidence, ABSTAIN refuses
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import llm
import radar_core as core
import retrieval

try:
    import spacy

    try:
        NLP = spacy.load("en_core_web_sm")
    except OSError:
        NLP = None
except ImportError:                                     # pragma: no cover
    NLP = None

FACTUAL_LABELS = {
    "PERSON", "GPE", "LOC", "ORG", "DATE", "TIME", "CARDINAL", "ORDINAL",
    "MONEY", "PERCENT", "QUANTITY", "EVENT", "FAC", "NORP", "LAW",
    "WORK_OF_ART", "PRODUCT", "LANGUAGE",
}
MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")

PROMPTS = {
    "short": (
        "You are a careful factual assistant. Answer the question with the "
        "shortest complete answer - a name, number, date or short phrase - in "
        "a single sentence. No explanation."
    ),
    "sentence": (
        "You are a careful factual assistant. Answer the question in one or "
        "two plain sentences. Be specific and do not hedge."
    ),
}

RAG_PROMPT = (
    "Answer the question using ONLY the context below. Give the shortest "
    "complete answer in one sentence. If the context does not contain the "
    "answer, reply exactly: NOT IN CONTEXT\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

ABSTAIN_MESSAGE = "I can't verify this with enough certainty, so I won't guess."


@dataclass
class Report:
    question: str
    model: str
    answer: str
    decision: str
    final_answer: str
    final_status: str           # answered | verified | corrected | unverified | abstained
    risk: float
    confidence: float
    support: float
    entropy: float
    entities: list = field(default_factory=list)
    tokens: list = field(default_factory=list)       # [{text, p, top}]
    key_source: str = "entities"
    samples: list = field(default_factory=list)
    clusters: list = field(default_factory=list)
    verification: dict | None = None
    alignment: str = "exact"
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# step 2-3: find and score the answer-bearing parts
# --------------------------------------------------------------------------

def _is_factual_date(text: str) -> bool:
    low = text.lower()
    return any(ch.isdigit() for ch in text) or any(m in low for m in MONTHS)


def _primary_entity(entities: list, question: str):
    """
    The entity that actually answers the question.

    An answer often volunteers details nobody asked for ("the Pacific Ocean,
    covering 63 million square kilometres").  Those extras should not be able
    to veto an answer that is itself confident, so the decision is driven by
    the entity whose type matches the question word.
    """
    q = question.lower()
    if "how many" in q or "how much" in q or "percentage" in q:
        prefer = {"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "ORDINAL"}
    elif "what year" in q or q.startswith("when"):
        prefer = {"DATE", "TIME"}
    elif q.startswith("who"):
        prefer = {"PERSON", "ORG", "NORP"}
    elif q.startswith("where"):
        prefer = {"GPE", "LOC", "FAC"}
    else:
        prefer = {"PERSON", "GPE", "LOC", "ORG", "EVENT", "PRODUCT",
                  "WORK_OF_ART", "NORP"}
    for e in entities:
        if e["label"] in prefer:
            return e
    return max(entities, key=lambda e: e["min"])


def _key_spans(answer: str, question: str) -> tuple:
    """Character spans worth scoring, plus where they came from."""
    spans = []
    if NLP is not None:
        for ent in NLP(answer).ents:
            if ent.label_ not in FACTUAL_LABELS:
                continue
            if ent.label_ == "DATE" and not _is_factual_date(ent.text):
                continue
            if not core.content_tokens(ent.text, question):
                continue                    # entity only repeats the question
            spans.append((ent.start_char, ent.end_char, ent.text, ent.label_))
    if spans:
        return spans, "entities"

    # fallback: every content word that is not in the question
    import re

    q = core.content_tokens(question) | {
        t for t in core.normalize_answer(question).split()}
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*[A-Za-z0-9]|[A-Za-z0-9]", answer):
        word = m.group(0)
        toks = core.content_tokens(word)
        if toks and not toks <= q:
            spans.append((m.start(), m.end(), word, "KEY"))
    return spans, "content words"


def score_answer(answer: str, tokens: list, question: str):
    spans, status = core.align_tokens(answer, tokens)
    heat = []
    if spans:
        heat = [{"text": s[4], "p": s[2],
                 "top": [{"token": t["token"], "p": round(t["p"], 4)} for t in s[3]]}
                for s in spans]
    entities = []
    if spans:
        keys, source = _key_spans(answer, question)
        for start, end, text, label in keys:
            sc = core.score_span(spans, start, end, question)
            if sc is None:
                continue
            entities.append({
                "text": text, "label": label, "start": start, "end": end,
                **sc, "level": core.level_for(sc["min"]),
            })
    else:
        source = "none"

    if entities:
        confidence = min(e["min"] for e in entities)
    elif spans:
        word_ps = [s[2] for s in spans if any(ch.isalnum() for ch in s[4])]
        confidence = min(word_ps) if word_ps else 0.0
    else:
        confidence = 0.5                    # no logprobs: neutral, rely on support
    return entities, heat, confidence, source, status


# --------------------------------------------------------------------------
# step 4: equivalence back-ends
# --------------------------------------------------------------------------

def make_equivalence(kind: str, question: str, model: str):
    """'lexical' (fast, default) or 'llm' (model judges paraphrases)."""
    if kind != "llm":
        return None
    cache: dict = {}

    def judge(a: str, b: str) -> bool:
        if core.normalize_answer(a) == core.normalize_answer(b):
            return True
        key = tuple(sorted((a, b)))
        if key not in cache:
            prompt = (f"Question: {question}\nAnswer A: {a}\nAnswer B: {b}\n"
                      "Do A and B give the same answer to the question? "
                      "Reply with exactly one word: yes or no.")
            out = llm.generate(model, prompt, temperature=0.0, num_predict=3).text
            cache[key] = out.strip().lower().startswith("yes")
        return cache[key]

    return judge


# --------------------------------------------------------------------------
# step 6: verification
# --------------------------------------------------------------------------

def verify(question: str, answer: str, model: str, use_local: bool = True,
           use_wikipedia: bool = True, equivalent=None) -> dict:
    t0 = time.time()
    passages = retrieval.retrieve(question, use_local, use_wikipedia)
    result = {"passages": [p.__dict__ for p in passages], "status": "no_evidence",
              "rag_answer": "", "coverage": 0.0, "seconds": 0.0}
    if not passages:
        result["seconds"] = time.time() - t0
        return result

    context = "\n\n".join(f"[{p.title}] {p.text}" for p in passages)[:6000]
    rag = llm.generate(model, RAG_PROMPT.format(context=context, question=question),
                       temperature=0.0, num_predict=64).text.strip()
    result["rag_answer"] = rag
    result["coverage"] = retrieval.evidence_coverage(answer, passages, question)
    rag_coverage = retrieval.evidence_coverage(rag, passages, question)
    eq = equivalent or (lambda x, y: core.answers_equivalent(x, y, question))

    if "not in context" in rag.lower() or not rag:
        result["status"] = "no_evidence"
    elif eq(answer, rag) and result["coverage"] >= 0.5:
        result["status"] = "supported"
    elif rag_coverage >= 0.5:
        result["status"] = "contradicted"
    else:
        result["status"] = "inconclusive"
    result["seconds"] = time.time() - t0
    return result


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------

def analyze(question: str, model: str = llm.DEFAULT_MODEL, n_samples: int = 5,
            temperature: float = 0.8, mode: str = "short", do_verify: bool = True,
            use_local: bool = True, use_wikipedia: bool = True,
            equivalence: str = "lexical", risk_low: float = core.RISK_LOW,
            risk_high: float = core.RISK_HIGH, w_conf: float = core.W_CONFIDENCE,
            w_sup: float = core.W_SUPPORT, verify_all: bool = False,
            progress=None) -> Report:
    system = PROMPTS.get(mode, PROMPTS["short"])
    timings = {}

    def step(msg):
        if progress:
            progress(msg)

    step("Generating answer")
    t0 = time.time()
    g = llm.generate(model, question, system=system, temperature=0.0, seed=0,
                     num_predict=96, logprobs=True, top_logprobs=5)
    timings["answer"] = time.time() - t0
    answer_raw = g.text
    answer = answer_raw.strip()

    entities, heat, confidence, source, status = score_answer(answer_raw, g.tokens, question)

    # The headline confidence belongs to the entity that answers the question.
    # A volunteered detail ("covering 63 million km2") must not make a correct
    # answer look unsure; it is still highlighted, just not decisive.
    primary = _primary_entity(entities, question) if entities else None
    if primary is not None:
        confidence = primary["min"]
    weakest = min((e["min"] for e in entities), default=confidence)

    step(f"Resampling {n_samples} answers")
    t0 = time.time()
    samples = []
    for i in range(n_samples):
        try:
            s = llm.generate(model, question, system=system, temperature=temperature,
                             seed=1000 + 17 * i, num_predict=96).text.strip()
        except Exception:                               # noqa: BLE001
            s = ""
        samples.append(s)
    timings["samples"] = time.time() - t0

    eq = make_equivalence(equivalence, question, model)
    cons = core.consistency(answer, samples, question, eq)

    # Two different questions, two different numbers:
    #   confidence  what to show  - the fact that answers the question
    #   weakest     what to score - the shakiest fact anywhere in the answer,
    #               so a volunteered detail still lowers trust (sensitivity)
    risk = core.risk_score(weakest, cons.support, w_conf, w_sup)
    decision = core.triage(risk, risk_low, risk_high)

    # ...but a weak detail may only downgrade a confident answer to VERIFY.
    # It must never veto an answer whose own fact is solid.
    if decision == core.ABSTAIN and primary and primary["min"] >= core.ENTITY_SAFE:
        decision = core.VERIFY

    verification = None
    final_answer, final_status = answer, "answered"
    if decision == core.ABSTAIN:
        final_answer, final_status = ABSTAIN_MESSAGE, "abstained"
    elif decision == core.VERIFY or (verify_all and decision == core.PASS):
        if do_verify:
            step("Checking evidence")
            verification = verify(question, answer, model, use_local, use_wikipedia, eq)
            timings["verify"] = verification["seconds"]
            vs = verification["status"]
            if vs == "supported":
                final_status = "verified"
            elif vs == "contradicted":
                final_answer, final_status = verification["rag_answer"], "corrected"
            else:
                final_status = "answered" if decision == core.PASS else "unverified"
        else:
            final_status = "unverified"

    return Report(
        question=question, model=model, answer=answer, decision=decision,
        final_answer=final_answer, final_status=final_status,
        risk=risk, confidence=confidence, support=cons.support,
        entropy=cons.entropy, entities=entities, tokens=heat, key_source=source,
        samples=cons.samples, clusters=cons.clusters, verification=verification,
        alignment=status, timings=timings,
    )
