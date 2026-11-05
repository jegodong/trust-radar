"""dashboard.py -- Trust Radar web app.

    streamlit run dashboard.py
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import benchmark
import llm
import radar_core as core
import trust_radar as tr

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Trust Radar", page_icon="📡", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.verdict {border-radius: 14px; padding: 18px 22px; margin: 4px 0 14px 0; border: 1px solid;}
.verdict .head {font-size: 1.05rem; font-weight: 700; line-height: 1.2;}
.verdict .tech {font-size: .7rem; letter-spacing: .09em; text-transform: uppercase;
               opacity: .55; margin-top: 2px;}
.verdict .ans {font-size: 1.55rem; font-weight: 600; margin: 4px 0 2px 0; line-height: 1.3;}
.verdict .why {font-size: .92rem; opacity: .85;}
.v-PASS {background: #e8f5e9; border-color: #a5d6a7; color: #1b5e20;}
.v-VERIFY {background: #fff8e1; border-color: #ffe082; color: #7a4f01;}
.v-ABSTAIN {background: #ffebee; border-color: #ef9a9a; color: #8e1b1b;}
.tokens {font-size: 1.15rem; line-height: 2.3; font-family: ui-monospace, Menlo, monospace;}
.tok {padding: 3px 1px; border-radius: 4px; white-space: pre-wrap;}
.ent {font-size: 1.35rem; line-height: 2;}
.ent span.e {padding: 3px 7px; border-radius: 6px; font-weight: 600;}
.small {font-size: .85rem; opacity: .75;}
.legend span {padding: 2px 8px; border-radius: 4px; margin-right: 6px; font-size: .8rem;}
</style>
""", unsafe_allow_html=True)

SAMPLES = {
    "Confident & right": [
        "Who was the first person to walk on the Moon?",
        "Who wrote the novel 1984?",
        "What is the chemical symbol for gold?",
    ],
    "Torn between answers": [
        "What is the capital of Australia?",
        "What is the capital of Switzerland?",
        "What is the largest ocean on Earth?",
    ],
    "Long-tail facts": [
        "Who won the 1987 Nobel Prize in Literature?",
        "Who was the first Chief Executive of Hong Kong?",
        "What is the capital of Burkina Faso?",
    ],
    "Confidently wrong (try Verify-all)": [
        "In what year did the Byzantine emperor Basil II die?",
        "At what temperature in Celsius does water boil at sea level?",
        "How many hearts does an octopus have?",
    ],
}

DECISION_TEXT = {
    "PASS": ("🟢", "Safe to use"),
    "VERIFY": ("🟡", "Check this before you trust it"),
    "ABSTAIN": ("🔴", "Not reliable enough to answer"),
}

STATUS_TEXT = {
    "answered": "Low risk – shown as generated.",
    "verified": "Checked against retrieved evidence – supported.",
    "corrected": "The model's first answer conflicted with the evidence; this is the evidence-based answer.",
    "unverified": "Medium risk and no evidence found – treat with caution.",
    "abstained": "Too uncertain – the radar refused to answer.",
}

if "history" not in st.session_state:
    st.session_state.history = []
if "question" not in st.session_state:
    st.session_state.question = SAMPLES["Torn between answers"][0]


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 Trust Radar")
    models = llm.list_models()
    if models:
        st.success(f"Ollama connected · {len(models)} model(s)", icon="🟢")
        default = llm.DEFAULT_MODEL if llm.DEFAULT_MODEL in models else models[0]
        model = st.selectbox("Model", models, index=models.index(default))
    else:
        st.error(f"Ollama not reachable at {llm.HOST}", icon="🔴")
        model = st.text_input("Model", llm.DEFAULT_MODEL)

    mode = st.radio("Answer style", ["short", "sentence"], horizontal=True,
                    help="Short answers give the cleanest signal.")
    n_samples = st.slider("Resamples", 3, 12, 5,
                          help="How many extra answers to draw to test consistency.")
    temperature = st.slider("Resample temperature", 0.3, 1.2, 0.8, 0.1)

    st.markdown("**Display**")
    colour_answer = st.toggle("Colour the answer", True,
                              help="Shade each fact by how sure the model was.")

    st.markdown("**Verification**")
    use_local = st.toggle("Local knowledge base", True,
                          help="Searches .md/.txt files in ./knowledge")
    use_wiki = st.toggle("Wikipedia", True)
    verify_all = st.toggle("Verify-all (paranoid mode)", False,
                           help="Also fact-check low-risk answers. Catches confident errors.")

    with st.expander("Advanced"):
        equivalence = st.selectbox("Answer matching", ["lexical", "llm"],
                                   help="lexical = fast word/number overlap; "
                                        "llm = the model judges if two answers mean the same")
        risk_low, risk_high = st.slider("Risk bands (PASS | VERIFY | ABSTAIN)",
                                        0.0, 1.0, (core.RISK_LOW, core.RISK_HIGH), 0.05)
        w_sup = st.slider("Weight on sample support", 0.0, 1.0, core.W_SUPPORT, 0.05)
        w_conf = 1.0 - w_sup
        st.caption(f"risk = 1 − ({w_conf:.2f}·confidence + {w_sup:.2f}·support)")

SETTINGS = dict(model=model, n_samples=n_samples, temperature=temperature, mode=mode,
                use_local=use_local, use_wikipedia=use_wiki, equivalence=equivalence,
                risk_low=risk_low, risk_high=risk_high, w_conf=w_conf, w_sup=w_sup,
                verify_all=verify_all)

# Display-only flags: never passed to analyze().
DISPLAY = dict(colour=colour_answer)


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------

def entity_html(answer: str, entities: list) -> str:
    parts, cursor = [], 0
    for e in sorted(entities, key=lambda e: e["start"]):
        if e["start"] < cursor:
            continue
        parts.append(escape(answer[cursor:e["start"]]))
        rivals = ", ".join(f"{a['token'].strip()} {a['p']:.0%}" for a in e["alternatives"])
        tip = f"{e['label']} · confidence {e['min']:.2f}" + (f" · rivals: {rivals}" if rivals else "")
        parts.append(f'<span class="e" style="background:{core.color_for(e["min"])}" '
                     f'title="{escape(tip)}">{escape(answer[e["start"]:e["end"]])}</span>')
        cursor = e["end"]
    parts.append(escape(answer[cursor:]))
    return f'<div class="ent">{"".join(parts)}</div>'


def was_torn(entity: dict, floor: float = 0.05) -> bool:
    """True when a rival answer was genuinely in the running at that token."""
    alts = entity.get("alternatives") or []
    return bool(alts) and alts[0]["p"] >= floor


def plain_html(text: str) -> str:
    return f'<div class="ent">{escape(text)}</div>'


def rivals_chart(entity: dict) -> go.Figure:
    """Chosen answer plus the alternatives that actually mattered, biggest first."""
    bars = [(entity["text"], entity["first"], "#43a047")]
    for alt in entity["alternatives"]:
        if alt["p"] >= 0.01:                       # 0% rows are noise, not rivals
            bars.append((alt["token"].strip(), alt["p"], "#ef9a9a"))
    bars.sort(key=lambda b: b[1])                  # ascending: largest ends up on top
    fig = go.Figure(go.Bar(
        x=[b[1] for b in bars], y=[b[0] for b in bars], orientation="h",
        marker_color=[b[2] for b in bars],
        text=[f"{b[1]:.0%}" for b in bars], textposition="outside"))
    fig.update_layout(height=50 + 40 * len(bars), margin=dict(l=10, r=40, t=36, b=10),
                      title=f"What the model nearly said instead of “{entity['text']}”",
                      xaxis=dict(range=[0, 1.1], tickformat=".0%"))
    return fig


def render_samples(r: tr.Report) -> None:
    """Every resample, marked against the answer the user is shown."""
    flags = [core.answers_equivalent(r.answer, s, r.question) for s in r.samples]
    st.markdown(f"**{sum(flags)} of {len(flags)} resamples say the same thing.**")
    for i, (sample, ok) in enumerate(zip(r.samples, flags), 1):
        st.write(("✅" if ok else "❌"), f"{i}.", sample)


def render_report(r: tr.Report) -> None:
    colour = DISPLAY.get("colour", True)

    # ---- one card: the verdict in plain words, then the answer, then why
    if r.final_answer != r.answer:
        body = plain_html(r.final_answer)
    elif colour:
        body = entity_html(r.answer, r.entities)
    else:
        body = plain_html(r.answer)

    weak = min(r.entities, key=lambda e: e["min"]) if r.entities else None
    n = len(r.samples)
    bits = [f"{round(r.support * n)}/{n} resamples agree"]
    if weak:
        bits.append(f"weakest fact “{weak['text'][:40]}” at {weak['min']:.0%}")
    bits.append(f"risk {r.risk:.2f}")

    icon, headline = DECISION_TEXT.get(r.decision, ("", r.decision))
    st.markdown(
        f'<div class="verdict v-{r.decision}">'
        f'<div class="head">{icon} {headline}</div>'
        f'<div class="tech">{r.decision} · {r.final_status.upper()}</div>'
        f'{body}'
        f'<div class="why">{escape(" · ".join(bits))}</div>'
        f'</div>', unsafe_allow_html=True)

    if r.final_answer != r.answer:
        st.caption(f"Model's original answer: “{r.answer}”")

    # ---- the numbers behind the verdict
    m1, m2, m3 = st.columns(3)
    m1.metric("Answer confidence", f"{r.confidence:.0%}",
              help="How sure the model was about the fact that answers the question")
    m2.metric("Sample support", f"{r.support:.0%}",
              help="Share of resamples that agree with the answer shown")
    m3.metric("Semantic entropy", f"{r.entropy:.2f}",
              help=(f"0 = every resample said the same thing, 1 = all different. "
                    f"With {n} samples only a few values are possible: a 4+1 split scores "
                    f"0.31, 3+2 scores 0.42, 3+1+1 scores 0.59. Read it as coarse evidence, "
                    f"not a precise score."))

    total = sum(r.timings.values())
    st.caption(f"Scored on {r.key_source} · {total:.1f}s "
               f"({', '.join(f'{k} {v:.1f}s' for k, v in r.timings.items())})")

    if colour and r.entities:
        st.markdown(
            '<div class="legend small">'
            f'<span style="background:{core.color_for(0.9)}">≥ {core.ENTITY_SAFE:.0%} sure</span>'
            f'<span style="background:{core.color_for(0.6)}">≥ {core.ENTITY_UNSURE:.0%}</span>'
            f'<span style="background:{core.color_for(0.1)}">&lt; {core.ENTITY_UNSURE:.0%}</span>'
            '</div>', unsafe_allow_html=True)
        st.caption("Colour = how sure the model was about that word, not whether it is correct.")

    tab_a, tab_c, tab_d = st.tabs(["🎯 Facts", "🔁 Resamples", "📚 Evidence"])

    with tab_a:
        if r.entities:
            contested = max(r.entities,
                            key=lambda e: (e["alternatives"][0]["p"]
                                           if e.get("alternatives") else 0.0))
            if was_torn(contested):
                st.plotly_chart(rivals_chart(contested), width="stretch")
            else:
                st.caption("The model was not torn here — everything else it considered "
                           "had less than 5% probability.")
            st.dataframe(pd.DataFrame([{
                "fact": e["text"], "type": e["label"],
                "confidence": round(e["min"], 3),
                "nearly said": ", ".join(f"{a['token'].strip()} ({a['p']:.0%})"
                                         for a in e["alternatives"]) or "—",
            } for e in r.entities]), width="stretch", hide_index=True)
        else:
            st.info("No answer-bearing tokens found (or the model returned no logprobs).")

    with tab_c:
        if r.samples:
            render_samples(r)

    with tab_d:
        v = r.verification
        if not v:
            st.info("No verification run – the answer was routed to "
                    f"**{r.decision}**. Turn on *Verify-all* to check every answer.")
        else:
            label = {"supported": "✅ supported", "contradicted": "❌ contradicted",
                     "inconclusive": "❔ inconclusive", "no_evidence": "∅ no evidence found"}
            st.markdown(f"**Result:** {label.get(v['status'], v['status'])} · "
                        f"answer words found in evidence: {v['coverage']:.0%} · {v['seconds']:.1f}s")
            if v["rag_answer"]:
                st.markdown(f"**Answer re-derived from evidence:** {escape(v['rag_answer'])}")
            for p in v["passages"]:
                src_label = "📁 local" if p["source"] == "local" else "🌐 Wikipedia"
                with st.expander(f"{src_label} · {p['title']}"):
                    st.write(p["text"][:1200])
                    if p.get("url", "").startswith("http"):
                        st.markdown(f"[{p['url']}]({p['url']})")

    st.download_button("⬇ Download report (JSON)",
                       json.dumps(r.to_dict(), indent=2, default=str),
                       file_name="trust_radar_report.json", mime="application/json")


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
st.title("📡 Trust Radar")
st.caption("Spot hallucinations before you trust them: token-level fact confidence, "
           "cross-sample consistency, and automatic PASS / VERIFY / ABSTAIN triage.")

page_ask, page_bench, page_hist, page_how = st.tabs(
    ["🔍 Ask", "📊 Benchmark", "🕘 History", "ℹ️ How it works"])

def _use_sample(q: str) -> None:
    """Button callback: runs before the widgets are built on the next run."""
    st.session_state.question = q
    st.session_state.autorun = True


with page_ask:
    with st.form("ask"):
        question = st.text_input("Your question", key="question",
                                 placeholder="e.g. What is the capital of Australia?")
        submitted = st.form_submit_button("Analyze", type="primary")

    with st.expander("Or try a sample question"):
        cols = st.columns(2)
        for i, (group, qs) in enumerate(SAMPLES.items()):
            with cols[i % 2]:
                st.markdown(f"<span class='small'>{group}</span>", unsafe_allow_html=True)
                for q in qs:
                    st.button(q, key=f"s-{q}", width="stretch",
                              on_click=_use_sample, args=(q,))

    if submitted or st.session_state.pop("autorun", False):
        status = st.status("Running the radar…", expanded=False)
        try:
            report = tr.analyze(st.session_state.question,
                                progress=lambda m: status.update(label=m), **SETTINGS)
            status.update(label="Done", state="complete")
            st.session_state.history.insert(0, report)
            st.session_state.current = report
        except Exception as exc:                                  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")

    if st.session_state.get("current"):
        render_report(st.session_state.current)

with page_bench:
    st.markdown("Runs the labelled questions in `questions.json` **without verification**, "
                "then checks whether the radar's risk score separates right from wrong answers.")
    saved = HERE / "results" / "results.csv"
    c1, c2, c3 = st.columns([2, 1, 1])
    tiers = c1.multiselect("Question tiers", ["easy", "medium", "hard", "trick"],
                           ["easy", "medium", "hard", "trick"])
    limit = c2.number_input("Max questions", 1, 500, 48)
    run_bench = c3.button("Run benchmark", type="primary")

    rows = None
    if run_bench:
        qs = benchmark.load_questions(tiers=tiers, limit=int(limit))
        bar = st.progress(0.0, text="starting")

        def prog(i, total, row):
            bar.progress(i / total, text=f"{i}/{total} · {row['question'][:60]}")

        kw = {k: v for k, v in SETTINGS.items()
              if k not in ("model", "n_samples", "use_local", "use_wikipedia", "verify_all")}
        rows = benchmark.run(model, qs, n_samples=n_samples, progress=prog, **kw)
        benchmark.write_outputs(rows, benchmark.summarize(rows, risk_low, risk_high), model)
    elif saved.exists():
        rows = pd.read_csv(saved).fillna("").to_dict("records")
        meta = json.loads((HERE / "results" / "summary.json").read_text())
        st.caption(f"Showing saved results for **{meta.get('model', '?')}** "
                   f"from results/results.csv")

    if rows:
        s = benchmark.summarize(rows, risk_low, risk_high)
        m = st.columns(5)
        m[0].metric("Raw accuracy", f"{s['accuracy_raw']:.0%}", help="Model alone")
        pa = s["triage"]["PASS"]["accuracy"]
        m[1].metric("Accuracy of PASSed answers", f"{pa:.0%}" if pa is not None else "–")
        m[2].metric("AUROC (fused trust)", s["auroc"]["fused_trust"] or "–",
                    help="0.5 = random, 1.0 = perfectly separates right from wrong")
        m[3].metric("Wrong answers caught", f"{s['wrong_answers_caught']:.0%}",
                    help="Wrong answers routed to VERIFY or ABSTAIN")
        m[4].metric("Right answers passed", f"{s['right_answers_passed']:.0%}")
        st.markdown(
            f"AUROC by signal — token confidence **{s['auroc']['token_confidence']}**, "
            f"sample support **{s['auroc']['sample_support']}**, "
            f"fused **{s['auroc']['fused_trust']}** · ECE (fused) {s['ece']['fused_trust']} · "
            f"Brier {s['brier_fused_trust']}")
        figs = benchmark.figures(rows)
        a, b = st.columns(2)
        a.plotly_chart(figs[0], width="stretch")
        b.plotly_chart(figs[3], width="stretch")
        a.plotly_chart(figs[2], width="stretch")
        b.plotly_chart(figs[1], width="stretch")
        df = pd.DataFrame(rows)[["tier", "question", "answer", "gold", "correct",
                                 "decision", "risk", "confidence", "support", "rivals"]]
        df["correct"] = df["correct"].map({1: "✅", 0: "❌"})
        st.dataframe(df, width="stretch", hide_index=True, height=420)

with page_hist:
    if not st.session_state.history:
        st.info("Questions you analyze appear here.")
    for i, r in enumerate(st.session_state.history):
        icon = {"PASS": "🟢", "VERIFY": "🟡", "ABSTAIN": "🔴"}[r.decision]
        with st.expander(f"{icon} {r.question} → {r.final_answer[:60]}  (risk {r.risk:.2f})"):
            st.json({k: v for k, v in r.to_dict().items() if k not in ("tokens",)},
                    expanded=False)

with page_how:
    st.markdown(f"""
### Two signals, one decision

| | asks | catches | misses |
|---|---|---|---|
| **Fact confidence** | how sure was the model about the *answer-bearing* tokens? | guesses, near-ties (Sydney 55% vs Canberra 26%) | fluent, confident errors |
| **Sample support** | ask {n_samples}× more – do they say the same thing? | unstable answers, made-up names | errors the model repeats every time |

**1. Answer-bearing tokens only.** spaCy finds names, places, dates and numbers. Words copied from the
question and grammar tokens are ignored, so “*The capital of Australia is* **Sydney**” is scored on *Sydney* alone.

**2. Weakest link, rivals only.** An entity's confidence is its weakest token. At the first token we
renormalise over *competing answers* only – alternatives like “The” are phrasing, not a rival fact.

**3. Support.** Resampled answers are grouped by meaning (word overlap + numbers must agree, or an LLM judge),
and we count how many agree with the answer actually shown.

**4. Triage.** `risk = 1 − ({w_conf:.2f}·confidence + {w_sup:.2f}·support)`

* **PASS** (risk < {risk_low:.2f}) – shown as is
* **VERIFY** ({risk_low:.2f}–{risk_high:.2f}) – retrieve evidence (local `knowledge/` folder, then Wikipedia),
  let the model re-answer from it; keep, correct, or flag
* **ABSTAIN** (≥ {risk_high:.2f}) – refuse instead of guessing

**Limitation:** a model that is consistently and confidently wrong passes both signals.
That is what *Verify-all* is for.
""")
