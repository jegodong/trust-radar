"""retrieval.py -- evidence for the VERIFY tier, from Wikipedia.

Searching Wikipedia with the raw question finds the wrong pages ("Who was the
first person to walk on the Moon?" returns the film "A Walk on the Moon";
"...Basil II die?" returns "Basil I").  So we search two ways:

1. keywords   the question without filler words
              ("Byzantine emperor Basil II die" -> the Basil II article)
2. candidates every answer in play - the model's answer, the other answers
              its resamples gave, the second model's answer - searched by
              name ("Canberra", "Sydney" -> their own articles)

Searching all candidates, not just the model's answer, keeps the evidence fair:
the model can be confirmed or contradicted.

From each article intro we keep only the sentences that mention the question's
keywords or a candidate, so the context stays short enough for small models.
Needs internet; offline it returns [] and the answer is marked unverified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import radar_core as core

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TrustRadar/2.0 (hallucination-detection demo; python-requests)"
TIMEOUT = 6.0


@dataclass
class Passage:
    title: str
    text: str
    source: str        # "wikipedia"
    score: float = 0.0
    url: str = ""


def _get(params: dict) -> dict:
    """One MediaWiki API call; {} on any network error."""
    try:
        import requests

        r = requests.get(WIKI_API, params={**params, "format": "json"},
                         headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:                                   # noqa: BLE001
        return {}


_VERBS = frozenset("""died born wrote written painted composed won awarded winner
discovered founded located signed invented directed sung released built became won""".split())


def keyword_query(question: str) -> str:
    """'In what year did Basil II die?' -> 'year Basil II die' (case kept)."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", question)
    return " ".join(w for w in words
                    if w.lower() not in core.STOPWORDS and w.lower() not in _VERBS)


def candidate_query(answer: str, question: str) -> str:
    """
    The name a candidate answer points at, for a title search.

    'Australia's capital is Sydney.'        -> 'Sydney'
    'Gertrude Steuern was awarded the prize' -> 'Gertrude Steuern'
    'Nitrogen.'                              -> 'Nitrogen'

    Capitalised words that are not in the question are taken in their
    original spelling.  Without any, short answers fall back to their plain
    words.  Pure numbers are skipped: searching '1081' finds the year
    article, which is about other events and only confuses the reader model.
    """
    q_words = set(core.normalize_answer(question).split())
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", answer)

    def useful(w):
        low = core.normalize_answer(w)
        return low and low not in core.STOPWORDS and low not in q_words and low not in _VERBS

    names = [w for w in words if w[0].isupper() and useful(w)]
    if names:
        return " ".join(names[:4])
    plain = [w for w in words if useful(w)]
    return " ".join(plain) if 0 < len(plain) <= 3 else ""


def search_titles(query: str, k: int) -> list:
    if not query.strip():
        return []
    data = _get({"action": "query", "list": "search", "srsearch": query, "srlimit": k})
    return [h["title"] for h in data.get("query", {}).get("search", [])]


def fetch_intros(titles: list) -> dict:
    """{title: intro text} for up to 20 titles in one call."""
    if not titles:
        return {}
    data = _get({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                 "redirects": 1, "titles": "|".join(titles[:20])})
    out = {}
    for page in data.get("query", {}).get("pages", {}).values():
        text = (page.get("extract") or "").strip()
        if text:
            out[page.get("title", "")] = text
    return out


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]


def search_wikipedia(question: str, candidates: tuple = (), k: int = 4,
                     per_page: int = 3) -> list:
    """Top-k passages: the most relevant sentences of the most relevant articles."""
    titles = search_titles(keyword_query(question), 3)
    seen_queries = set()
    for cand in candidates:
        q = candidate_query(cand, question)
        if q and q not in seen_queries:
            seen_queries.add(q)
            titles += search_titles(q, 1)
        if len(seen_queries) >= 4:
            break
    titles = list(dict.fromkeys(titles))            # dedupe, keep order
    intros = fetch_intros(titles)

    q_terms = core.content_tokens(question)
    cand_terms = [core.content_tokens(c, question) for c in candidates]
    cand_terms = [t for t in cand_terms if t]

    passages = []
    for title, text in intros.items():
        scored = []
        for sent in _sentences(text):
            toks = core.content_tokens(sent) | core.content_tokens(title)
            score = 2 * len(q_terms & toks) + sum(1 for t in cand_terms if t <= toks)
            if score > 0:
                scored.append((score, sent))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        best = scored[:per_page]
        passages.append(Passage(
            title=title, text=" ".join(s for _, s in best), source="wikipedia",
            score=float(best[0][0]),
            url="https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
        ))
    passages.sort(key=lambda p: -p.score)
    return passages[:k]


def retrieve(question: str, use_wikipedia: bool = True, candidates: tuple = (),
             k: int = 4) -> list:
    return search_wikipedia(question, candidates, k) if use_wikipedia else []


def evidence_coverage(claim: str, passages: list, question: str = "") -> float:
    """Share of the claim's answer-bearing words that appear in the evidence."""
    claim_tokens = core.content_tokens(claim, question)
    if not claim_tokens or not passages:
        return 0.0
    evidence = set()
    for p in passages:
        evidence |= core.content_tokens(p.title + " " + p.text)
    return len(claim_tokens & evidence) / len(claim_tokens)
