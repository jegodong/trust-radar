"""radar_core.py -- the maths behind Trust Radar.

Pure Python, no model calls, no third-party imports, so every number the
dashboard, the CLI and the benchmark print comes from the same functions and
can be unit-tested without Ollama.

Four ideas live here
--------------------
1. Answer-bearing tokens.  Grammar tokens ("the", "was") and words copied from
   the question ("Moon" in "Who walked on the Moon?") carry no new claim.  We
   score only the tokens that carry the answer.

2. Weakest-link confidence.  Inside an entity, the first token is the
   decision; the rest follows almost for free ("Neil" p=0.57 -> " Armstrong"
   p=0.996).  Averaging hides the doubt, so an entity's confidence is the
   minimum over its word tokens, and the claim's confidence is its weakest
   entity.

3. Support.  We resample the same question N times and count how many samples
   say the same thing as the answer the user is actually shown.  Two answers
   are "the same" when their answer-bearing content overlaps and their numbers
   do not contradict each other.

4. Triage.  Confidence and support are fused into one risk score in [0, 1],
   which maps to PASS / VERIFY / ABSTAIN.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# thresholds (all overridable from the dashboard / CLI)
# --------------------------------------------------------------------------

ENTITY_SAFE = 0.80          # green at or above
ENTITY_UNSURE = 0.50        # yellow at or above, red below

RISK_LOW = 0.35             # below -> PASS
RISK_HIGH = 0.65            # at or above -> ABSTAIN, in between -> VERIFY

W_CONFIDENCE = 0.4          # weight of token confidence in the risk score
W_SUPPORT = 0.6             # weight of cross-sample support

STOPWORDS = frozenset("""
a an the and or but if of at by for with about against between into through
during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t can will just don
should now is are was were be been being have has had having do does did doing
i me my myself we our ours you your yours he him his she her hers it its they
them their theirs what which who whom this that these those am would could
also as yes called known named answer approximately about around roughly
""".split())

_SUBSCRIPTS = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉"
    "⁰¹²³⁴⁵⁶⁷⁸⁹−",
    "01234567890123456789-",
)

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip accents/punctuation, unify '1,000' -> '1000'."""
    text = (text or "").strip().lower().translate(_SUBSCRIPTS)
    text = text.replace("’", "'").replace("°", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(?<=\d),[ ]*(?=\d{3}\b)", "", text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^0-9a-z一-鿿. ]+", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canon(token: str) -> str:
    """Number words -> digits, '100.0' -> '100', light plural stripping."""
    if token in NUMBER_WORDS:
        return str(NUMBER_WORDS[token])
    if _NUM_RE.match(token):
        value = float(token)
        return str(int(value)) if value.is_integer() else str(value)
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def is_number(token: str) -> bool:
    return bool(_NUM_RE.match(token))


def content_tokens(text: str, question: str = "") -> set:
    """
    The words that carry the answer: no stopwords, nothing copied from the
    question, numbers in canonical form.

        content_tokens("The first man on the Moon was Neil Armstrong.",
                       "Who was the first man on the Moon?")
        -> {"neil", "armstrong"}
    """
    q = {_canon(t) for t in normalize_answer(question).split()}
    out = set()
    for tok in normalize_answer(text).split():
        c = _canon(tok)
        if tok in STOPWORDS or c in STOPWORDS:
            continue
        if c in q:
            continue                       # echoes the question
        out.add(c)
    return out


# --------------------------------------------------------------------------
# answer equivalence + clustering
# --------------------------------------------------------------------------

def answers_equivalent(a: str, b: str, question: str = "",
                       overlap: float = 0.5) -> bool:
    """
    True when two answers make the same claim.

    * numbers must not contradict ("1969" ~ "July 20, 1969", "26" != "27")
    * the smaller answer's content must be at least `overlap` contained in
      the larger one ("Armstrong" ~ "Neil Armstrong", "Paris" != "Lyon")
    """
    ta, tb = content_tokens(a, question), content_tokens(b, question)
    if not ta or not tb:
        return normalize_answer(a) == normalize_answer(b)

    na = {t for t in ta if is_number(t)}
    nb = {t for t in tb if is_number(t)}
    if na and nb and not (na <= nb or nb <= na):
        return False
    if bool(na) != bool(nb):
        # one answer is a number, the other is words only: compare the words
        wa, wb = ta - na, tb - nb
        if not wa or not wb:
            return False
        ta, tb = wa, wb

    shared = len(ta & tb)
    return shared > 0 and shared / min(len(ta), len(tb)) >= overlap


def cluster_answers(answers: list, question: str = "", equivalent=None) -> list:
    """
    Group answers by meaning. Returns a label per answer.

    Each new answer joins the first cluster whose representative (its first
    member) it is equivalent to - the same greedy procedure used for
    semantic entropy (Kuhn et al., 2023).  `equivalent` lets callers plug in
    an embedding or NLI check instead of the lexical default.
    """
    eq = equivalent or (lambda x, y: answers_equivalent(x, y, question))
    reps: list = []
    labels = []
    for ans in answers:
        for idx, rep in enumerate(reps):
            if eq(ans, rep):
                labels.append(idx)
                break
        else:
            reps.append(ans)
            labels.append(len(reps) - 1)
    return labels


def semantic_entropy(labels: list) -> float:
    """
    Entropy of the cluster distribution divided by log(N).

    Dividing by the sample count (not the number of clusters) keeps the score
    monotone: 6/4 scores lower than 6/1/1/1/1, which scores lower than
    ten different answers.  0 = all agree, 1 = every sample different.
    """
    n = len(labels)
    if n <= 1:
        return 0.0
    counts: dict = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return max(0.0, h / math.log(n))


@dataclass
class Consistency:
    samples: list
    labels: list
    support: float            # share of samples that agree with the shown answer
    majority_share: float     # share of the largest cluster
    entropy: float            # normalised semantic entropy
    n_clusters: int
    clusters: list = field(default_factory=list)   # [{answer, size, share, agrees}]
    shown_in_majority: bool = True


def consistency(shown: str, samples: list, question: str = "",
                equivalent=None) -> Consistency:
    """How many resampled answers back up the answer the user sees."""
    samples = [s for s in samples if s and s.strip()]
    if not samples:
        return Consistency([], [], 0.0, 0.0, 0.0, 0, [], False)

    eq = equivalent or (lambda x, y: answers_equivalent(x, y, question))
    labels = cluster_answers(samples, question, eq)
    n = len(samples)

    groups: dict = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)

    agrees = [eq(shown, s) for s in samples]
    support = sum(agrees) / n

    clusters = []
    for lab, members in groups.items():
        clusters.append({
            "answer": samples[members[0]],
            "size": len(members),
            "share": len(members) / n,
            "agrees": any(agrees[i] for i in members),
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)

    return Consistency(
        samples=samples,
        labels=labels,
        support=support,
        majority_share=clusters[0]["share"],
        entropy=semantic_entropy(labels),
        n_clusters=len(clusters),
        clusters=clusters,
        shown_in_majority=clusters[0]["agrees"],
    )


# --------------------------------------------------------------------------
# token / entity confidence
# --------------------------------------------------------------------------

def align_tokens(text: str, tokens: list):
    """
    Map [{token, p, top}] onto character spans of `text`.

    Returns (spans, status).  Each span is (start, end, p, top, piece) and
    status is 'exact', 'clipped' (token list longer than text), 'partial'
    (shorter) or 'failed'.
    """
    if not tokens:
        return None, "empty"
    spans = []
    cursor = 0
    for tok in tokens:
        piece = tok.get("token", "")
        spans.append((cursor, cursor + len(piece), float(tok.get("p", float("nan"))),
                      tok.get("top") or [], piece))
        cursor += len(piece)
    joined = "".join(s[4] for s in spans)

    if joined == text:
        return spans, "exact"
    if joined.startswith(text):
        return [s for s in spans if s[0] < len(text)], "clipped"
    if text.startswith(joined):
        return spans, "partial"
    # Ollama sometimes strips leading whitespace from `response`
    stripped = joined.lstrip()
    if stripped and (stripped == text or stripped.startswith(text)):
        shift = len(joined) - len(stripped)
        moved = [(max(0, s - shift), e - shift, p, top, piece)
                 for s, e, p, top, piece in spans if e > shift and s - shift < len(text)]
        return moved, "exact" if stripped == text else "clipped"
    return None, "failed"


def _has_word(piece: str) -> bool:
    return any(ch.isalnum() for ch in piece)


def score_span(spans: list, start: int, end: int, question: str = "") -> dict | None:
    """
    Confidence of one character range (an entity or a key word).

    first      raw probability of the first token, where the model chose
    first_adj  the same, renormalised over competing *answers* only: top-k
               alternatives that are stopwords, punctuation or words from the
               question ("The", "Moon") are phrasing choices, not rival facts
    min        weakest word token, using first_adj for the first one - the
               number used for colours and risk
    margin     first_adj minus the strongest rival answer
    alternatives  rival answers the model nearly produced at that token
    """
    words = [s for s in spans if s[0] < end and s[1] > start and _has_word(s[4])]
    # Skip a leading article.  In "the Pacific Ocean" the decision the model
    # made is "Pacific", not "the", and its rivals are the interesting part.
    while len(words) > 1 and words[0][4].strip().lower() in {"the", "a", "an"}:
        words = words[1:]
    if not words:
        return None
    first = words[0]
    top = first[3] or []
    chosen = first[4].strip().lower()
    q_words = set(normalize_answer(question).split())

    rivals = []
    for t in top:
        tok = t["token"].strip()
        low = normalize_answer(tok)
        if not tok or tok.lower() == chosen or not low:
            continue
        if low in STOPWORDS or low in q_words:
            continue
        rivals.append(t)

    # Fragments such as "Sy" (for Sydney) are hidden from the display, but they
    # still count as competing probability mass - otherwise filtering them out
    # would silently inflate the confidence score.
    shown = [t for t in rivals
             if normalize_answer(t["token"]).isdigit() or len(normalize_answer(t["token"])) >= 3]

    rival_mass = sum(t["p"] for t in rivals)
    phrasing_mass = sum(t["p"] for t in top if t["token"].strip().lower() != chosen) - rival_mass
    # probability outside the top-k is split between rivals and phrasing in
    # the same ratio we observed inside it, so a flat distribution stays low
    unseen = max(0.0, 1.0 - first[2] - rival_mass - phrasing_mass) if top else 0.0
    seen_alt = rival_mass + phrasing_mass
    rival_share = rival_mass / seen_alt if seen_alt > 0 else 1.0
    denom = first[2] + rival_mass + unseen * rival_share
    first_adj = first[2] / denom if denom > 0 else first[2]
    probs = [first_adj] + [s[2] for s in words[1:]]
    return {
        "min": min(probs),
        "mean": sum(probs) / len(probs),
        "first": first[2],
        "first_adj": first_adj,
        "margin": first_adj - rivals[0]["p"] / denom if rivals and denom > 0 else None,
        "alternatives": shown[:3],
        "n_tokens": len(words),
    }


def color_for(p: float) -> str:
    if p is None or math.isnan(p):
        return "#e0e0e0"
    if p >= ENTITY_SAFE:
        return "#c8e6c9"
    if p >= ENTITY_UNSURE:
        return "#fff3b0"
    return "#ffcdd2"


def heat_color(p: float) -> str:
    """Continuous red -> yellow -> green background for the token heatmap."""
    if p is None or math.isnan(p):
        return "rgba(200,200,200,0.3)"
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        r, g, b = 239, int(83 + (p / 0.5) * (200 - 83)), 80
    else:
        t = (p - 0.5) / 0.5
        r, g, b = int(239 - t * (239 - 102)), int(200 - t * (200 - 187)), int(80 + t * (106 - 80))
    alpha = 0.15 + 0.55 * (1 - p)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def level_for(p: float) -> str:
    if p >= ENTITY_SAFE:
        return "high"
    if p >= ENTITY_UNSURE:
        return "medium"
    return "low"


# --------------------------------------------------------------------------
# risk fusion + triage
# --------------------------------------------------------------------------

PASS, VERIFY, ABSTAIN = "PASS", "VERIFY", "ABSTAIN"


def risk_score(confidence: float, support: float,
               w_conf: float = W_CONFIDENCE, w_sup: float = W_SUPPORT) -> float:
    """1 - weighted trust.  0 = safe, 1 = almost certainly unreliable."""
    total = w_conf + w_sup
    trust = (w_conf * confidence + w_sup * support) / total
    return max(0.0, min(1.0, 1.0 - trust))


def triage(risk: float, low: float = RISK_LOW, high: float = RISK_HIGH) -> str:
    if risk < low:
        return PASS
    if risk >= high:
        return ABSTAIN
    return VERIFY


# --------------------------------------------------------------------------
# grading + calibration (benchmark)
# --------------------------------------------------------------------------

def is_correct(answer: str, gold: list) -> bool:
    """
    True if any gold alias is contained in the answer.

    Aliases are matched on normalised content words; numbers must match
    exactly, so "100" accepts "100 degrees Celsius" but not "1000".
    """
    ans = content_tokens(answer)
    ans_norm = " " + normalize_answer(answer) + " "
    for alias in gold:
        alias_tokens = content_tokens(alias)
        if alias_tokens and alias_tokens <= ans:
            return True
        norm = normalize_answer(alias)
        if norm and f" {norm} " in ans_norm:
            return True
    return False


def auroc(scores: list, labels: list) -> float | None:
    """Probability a random correct answer outranks a random wrong one."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def ece(confidences: list, labels: list, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(labels[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


def reliability_bins(confidences: list, labels: list, bins: int = 10) -> list:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            out.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "confidence": sum(confidences[i] for i in idx) / len(idx),
                "accuracy": sum(labels[i] for i in idx) / len(idx),
                "count": len(idx),
            })
    return out


def brier(confidences: list, labels: list) -> float:
    if not confidences:
        return 0.0
    return sum((c - y) ** 2 for c, y in zip(confidences, labels)) / len(confidences)


def coverage_curve(trust: list, labels: list) -> list:
    """
    Selective answering: keep the k most trusted answers, k = 1..N.
    Returns [{coverage, accuracy}] - the curve a triage system lives on.
    """
    order = sorted(range(len(trust)), key=lambda i: trust[i], reverse=True)
    out = []
    correct = 0
    for k, i in enumerate(order, 1):
        correct += labels[i]
        out.append({"coverage": k / len(order), "accuracy": correct / k,
                    "threshold": trust[i]})
    return out


def aurc(trust: list, labels: list) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    curve = coverage_curve(trust, labels)
    if not curve:
        return 0.0
    return sum(1 - c["accuracy"] for c in curve) / len(curve)
