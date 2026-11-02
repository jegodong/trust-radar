"""radar_core.py -- single source of truth for confidence and consistency.

Two entry points import this module:

    dashboard.py / trust_radar.py   interactive: one question at a time
    trust_eval.py                   batch: 72 questions

so a number shown on screen and the same number in results.csv come from
identical code.  Before this module existed the two files had separate
implementations, different clustering thresholds and different verdict rules,
which meant the demo and the poster could disagree about the same answer.

Vocabulary used throughout:

    entity confidence   mean token probability of one factual entity
    exact agreement     fraction of the N samples that gave the identical
                        answer string (after normalisation)
    semantic clusters   samples grouped by meaning instead of by spelling
    SINdex              normalised entropy of those clusters, in [0, 1].
                        0 = every sample agreed, 1 = maximum disagreement for
                        the number of clusters observed.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
EMBEDDER_DIR = PROJECT_DIR / "all-MiniLM-L6-v2"

# ---- thresholds ----------------------------------------------------------
# Cosine-distance cut for semantic clustering, chosen from measured pairs
# (see README, "Why numbers are special-cased"):
#     "Canberra" vs "Canberra, 1908"      0.297   <- must merge
#     "Paris"    vs "Lyon"                0.503   <- must not merge
#     "Neil Armstrong" vs "Buzz Aldrin"   0.490
# so 0.35 is the only band that separates paraphrases from different facts.
# Numbers break this rule on their own ("26" vs "27" is only 0.114 apart),
# which is why numeric answers are clustered by exact match instead.
SEMANTIC_THRESHOLD = 0.35

# Entity confidence bands, applied to mean token probability.
ENTITY_SAFE = 0.80
ENTITY_UNSURE = 0.50

# SINdex bands.  The score is normalised to [0, 1], so these numbers mean the
# same thing whether the samples split into 2 clusters or 10.  Without the
# normalisation a 50/50 split scored 0.693 and could never reach 0.8, i.e.
# the worst possible two-way disagreement was reported as "moderate".
SINDEX_STABLE = 0.50
SINDEX_HIGH = 0.80

SUBSCRIPT_MAP = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2212",
    "01234567890123456789-",
)

_EMBEDDER = None
_EMBEDDER_TRIED = False


# ---- text normalisation --------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip diacritics and punctuation, unify number commas."""
    text = (text or "").strip().lower()
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace("\u2019", "'").replace("\u00b0", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff. ]+", " ", text)
    text = re.sub(r"\.(?=\s|$)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---- token alignment -----------------------------------------------------

def align_tokens(text: str, tokens: list):
    """
    Map (token, probability) pairs onto character spans of `text`.

    Ollama usually returns tokens that concatenate back to the response, but
    not always: `response` can be cut at a stop boundary while the logprob
    list keeps going.  Returning None in that case silently deleted rows, so
    we clip instead and report which path was taken.
    """
    if not tokens:
        return None, "empty"

    spans = []
    cursor = 0
    pieces = []
    for tok in tokens:
        piece = tok.get("token", "")
        pieces.append(piece)
        spans.append((cursor, cursor + len(piece), float(tok.get("p", float("nan")))))
        cursor += len(piece)

    joined = "".join(pieces)
    if joined == text:
        return spans, "exact"
    if joined.startswith(text):
        inside = [s for s in spans if s[0] < len(text)]
        if inside:
            return inside, "clipped"
    if text.startswith(joined):
        return spans, "partial"

    common = 0
    for a, b in zip(joined, text):
        if a != b:
            break
        common += 1
    if common and common >= 0.9 * min(len(joined), len(text)):
        inside = [s for s in spans if s[0] < common]
        if inside:
            return inside, "clipped"
    return None, "failed"


# ---- clustering ----------------------------------------------------------

def exact_clusters(answers: list) -> list:
    """Group by normalised string identity."""
    seen = {}
    labels = []
    for answer in answers:
        key = normalize_answer(answer)
        labels.append(seen.setdefault(key, len(seen)))
    return labels


def _semantic_labels(answers, embedder, threshold):
    """Agglomerative clustering on MiniLM embeddings; (labels, embeddings)."""
    if embedder is None or len(answers) < 2:
        return None, None
    try:
        from sklearn.cluster import AgglomerativeClustering

        emb = embedder.encode(answers, normalize_embeddings=True)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )
        return model.fit_predict(emb).tolist(), emb
    except Exception:                                  # noqa: BLE001
        return None, None


NUMBER_RE = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}

# Unit words are removed before reading the number, so "100 C", "100.0 C" and
# "100 degrees celsius" all collapse onto 100.0 while "26" and "27" stay apart.
_UNIT_WORDS = {
    "c", "f", "k", "celsius", "fahrenheit", "kelvin", "degree", "degrees",
    "percent", "km", "m", "cm", "mm", "kg", "g", "meters", "metres",
    "feet", "ft", "miles", "mile", "minutes", "minute", "seconds", "second",
    "hours", "hour", "days", "day", "years", "year", "moons", "moon",
    "sides", "legs", "strings", "keys", "players", "bones", "teeth",
    "chambers", "hearts", "states", "countries", "zones",
}


def numeric_key(answer: str):
    """
    The number an answer is really about, or None if it is not a number.

    Embeddings put "26" and "27" only 0.114 apart - closer than two
    paraphrases of the same fact (0.297) - so numeric answers are never
    clustered by similarity.  They are compared by value instead.
    """
    norm = normalize_answer(answer)
    if not norm:
        return None
    if norm in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[norm])
    tokens = [t for t in norm.split() if t not in _UNIT_WORDS]
    joined = " ".join(tokens)
    if NUMBER_RE.match(joined):
        try:
            return float(joined)
        except ValueError:
            return None
    return None


def isolate_numeric(labels, answers):
    """
    Split any embedding cluster that contains different numbers.

    Non-numeric answers keep their semantic cluster; numeric ones get a key
    derived from their value, so "100 C" and "100.0 C" agree while "26" and
    "27" never do.
    """
    keys = []
    for label, answer in zip(labels, answers):
        value = numeric_key(answer)
        keys.append(("num", value) if value is not None else ("sem", label))
    index = {}
    return [index.setdefault(key, len(index)) for key in keys]


def cluster_masses(labels) -> list:
    """Cluster sizes as fractions of the sample count, largest first."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = float(sum(counts.values())) or 1.0
    return sorted((c / total for c in counts.values()), reverse=True)


def normalized_entropy(masses) -> float:
    """Shannon entropy divided by log(k); 0 when everything agrees."""
    k = len(masses)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in masses if p > 0)
    return float(h / math.log(k))


def _coherence(emb, members) -> float:
    """Mean pairwise cosine similarity inside one cluster."""
    if len(members) < 2:
        return 1.0                       # nothing to disagree with
    total = 0.0
    pairs = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            total += float(sum(a * b for a, b in zip(emb[members[i]], emb[members[j]])))
            pairs += 1
    return total / pairs if pairs else 1.0


# ---- embedder ------------------------------------------------------------

def get_embedder():
    """Load MiniLM once per process; None if the folder is absent."""
    global _EMBEDDER, _EMBEDDER_TRIED
    if _EMBEDDER_TRIED:
        return _EMBEDDER
    _EMBEDDER_TRIED = True
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    if not EMBEDDER_DIR.exists():
        print("[radar_core] all-MiniLM-L6-v2 not found; using exact matching only")
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(str(EMBEDDER_DIR))
    except Exception as exc:                           # noqa: BLE001
        print(f"[radar_core] cannot load MiniLM ({exc}); using exact matching only")
        _EMBEDDER = None
    return _EMBEDDER


# ---- the shared measurement ---------------------------------------------

def consistency_report(answers, embedder=None, threshold=None) -> dict:
    """
    Everything both files need to know about a bag of sampled answers.

    Returns
    -------
    consistency_exact  share of samples with the modal answer string
    consistency_sem    share of samples in the largest semantic cluster
    sindex             normalised cluster entropy in [0, 1]
    level / verdict    'success' | 'warning' | 'error' + a sentence to display
    clusters           per-cluster view for the dashboard table
    """
    threshold = SEMANTIC_THRESHOLD if threshold is None else threshold
    answers = [a for a in (answers or []) if a and a.strip()]

    empty = {
        "n_samples": 0, "n_unique": 0, "n_clusters_sem": 0,
        "consistency_exact": 0.0, "consistency_sem": 0.0,
        "entropy_exact": 0.0, "sindex": 0.0,
        "level": "warning", "verdict": "No usable samples",
        "clusters": [],
    }
    if not answers:
        return empty

    exact = exact_clusters(answers)
    masses_exact = cluster_masses(exact)

    sem_labels, emb = _semantic_labels(answers, embedder, threshold)
    if sem_labels is None:
        sem_labels = exact
    sem_labels = isolate_numeric(sem_labels, answers)
    masses_sem = cluster_masses(sem_labels)

    sindex = normalized_entropy(masses_sem)
    level, verdict = sindex_verdict(sindex)

    # per-cluster detail for the dashboard
    groups = {}
    for idx, label in enumerate(sem_labels):
        groups.setdefault(label, []).append(idx)
    clusters = []
    for label, members in groups.items():
        size = len(members)
        coherence = _coherence(emb, members) if emb is not None else 1.0
        p = size / float(len(answers))
        clusters.append({
            "answer": answers[members[0]],
            "size": size,
            "coherence": round(coherence, 4),
            "p": round(p, 4),
            "weight": round(p * coherence, 4),
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)

    report = {
        "n_samples": len(answers),
        "n_unique": len(set(exact)),
        "n_clusters_sem": len(masses_sem),
        "consistency_exact": float(masses_exact[0]),
        "consistency_sem": float(masses_sem[0]),
        "entropy_exact": normalized_entropy(masses_exact),
        "sindex": round(sindex, 4),
        "level": level,
        "verdict": verdict,
        "clusters": clusters,
    }
    return report


def sindex_verdict(sindex: float):
    """Map a normalised SINdex onto a colour level and a sentence."""
    if sindex >= SINDEX_HIGH:
        return "error", "Answers disagree across samples - strong hallucination signal"
    if sindex >= SINDEX_STABLE:
        return "warning", "Partial disagreement - worth verifying before trusting"
    return "success", "Samples agree - low semantic inconsistency"


def entity_color(mean_prob: float) -> str:
    """Background colour for one entity, shared by the dashboard and figures."""
    if mean_prob >= ENTITY_SAFE:
        return "#c8e6c9"
    if mean_prob >= ENTITY_UNSURE:
        return "#fff9c4"
    return "#ffcdd2"


def entity_level(mean_prob: float) -> str:
    if mean_prob >= ENTITY_SAFE:
        return "success"
    if mean_prob >= ENTITY_UNSURE:
        return "warning"
    return "error"
