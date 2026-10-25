import ollama
import math
import spacy
import requests

nlp = spacy.load("en_core_web_sm")

FACTUAL_ENTITIES = {
    "PERSON", "GPE", "LOC", "ORG", "DATE",
    "CARDINAL", "MONEY", "PERCENT", "QUANTITY",
    "EVENT", "FAC", "NORP", "LAW",
}

MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}

def is_factual_date(text: str) -> bool:
    if any(ch.isdigit() for ch in text):
        return True
    return any(m in text for m in MONTHS)

def analyze(model: str, prompt: str):
    response = ollama.generate(
        model=model, prompt=prompt, logprobs=True, top_logprobs=3,
    )
    text = response["response"]
    logprobs_list = response["logprobs"]

    char_pos = 0
    token_spans = []
    for item in logprobs_list:
        token_text = item["token"]
        prob = math.exp(item["logprob"])
        start = char_pos
        end = char_pos + len(token_text)
        token_spans.append((start, end, prob, token_text))
        char_pos = end

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ not in FACTUAL_ENTITIES:
            continue
        if ent.label_ == "DATE" and not is_factual_date(ent.text):
            continue
        matching = [
            (tok, prob) for cs, ce, prob, tok in token_spans
            if cs < ent.end_char and ce > ent.start_char
        ]
        if not matching:
            continue
        probs = [p for _, p in matching]
        entities.append({
            "text": ent.text, "label": ent.label_,
            "start": ent.start_char, "end": ent.end_char,
            "avg": sum(probs) / len(probs), "min": min(probs),
            "tokens": [t for t, _ in matching],
        })
    return {"text": text, "entities": entities}

def multi_sample_check(model: str, prompt: str, n: int = 5):
    from collections import Counter
    short_prompt = (
        f"{prompt}\n\nAnswer with ONLY the name or entity that directly answers "
        "the question. Do not explain. Do not write a full sentence."
    )
    answers = []
    for _ in range(n):
        try:
            resp = ollama.generate(model=model, prompt=short_prompt)
            raw = resp["response"].strip().strip('."\'')
            answers.append({"raw": raw, "key": " ".join(raw.lower().split())})
        except Exception:
            answers.append({"raw": "", "key": "(empty)"})
    counter = Counter(a["key"] for a in answers)
    top_key, top_count = counter.most_common(1)[0]
    consistency = top_count / n
    unique_count = len(counter)
    top_raw = next((a["raw"] for a in answers if a["key"] == top_key), top_key)
    if consistency >= 0.8 and unique_count <= 2:
        verdict, level = "Stable", "success"
    elif consistency < 0.6 or unique_count >= 3:
        verdict, level = "High semantic divergence — consider abstaining", "error"
    else:
        verdict, level = "Moderate — worth external verification", "warning"
    return {
        "samples": [a["raw"] for a in answers],
        "consistency": consistency, "unique_count": unique_count,
        "top_entity": top_raw, "verdict": verdict, "level": level,
    }