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
.verdict .tag {font-weight: 700; letter-spacing: .06em; font-size: .8rem; opacity: .85;}
.verdict .ans {font-size: 1.55rem; font-weight: 600; margin: 4px 0 2px 0; line-height: 1.3;}
.verdict .why {font-size: .92rem; opacity: .85;}
.v-PASS {background: #e8f5e9; border-color: #a5d6a7; color: #1b5e20;}
.v-VERIFY {background: #fff8e1; border-color: #ffe082; color: #7a4f01;}
.v-ABSTAIN {background: #ffebee; border-color: #ef9a9a; color: #8e1b1b;}
.tokens {font-size: 1.15rem; line-height: 2.3; font-family: ui-monospace, Menlo, monospace;}
.tok {padding: 3px 1px; border-radius: 4px; white-space: pre-wrap; color: #111;}
.ent {font-size: 1.35rem; line-height: 2;}
.ent span.e {padding: 3px 7px; border-radius: 6px; font-weight: 600; color: #111;}
.small {font-size: .85rem; opacity: .75;}
.legend span {padding: 2px 8px; border-radius: 4px; margin-right: 6px; font-size: .8rem; color: #111;}
</style>
""", unsafe_allow_html=True)

STATUS_TEXT = {
    "answered": "Low risk – shown as generated.",
    "verified": "Checked against retrieved evidence – supported.",
    "corrected": "The model's first answer conflicted with the evidence; this is the evidence-based answer.",
    "unverified": "Medium risk and no evidence found – treat with caution.",
    "abstained": "Too uncertain and no supporting evidence found – the radar refused to answer.",
}

SAMPLE_QUESTIONS = [
    "Who was the first person to walk on the Moon?",
    "What is the capital of Australia?",
    "Who wrote the novel 1984?",
    "What is the capital of Switzerland?",
    "What is the chemical symbol for gold?",
    "What is the largest ocean on Earth?",
    "Who won the 1987 Nobel Prize in Literature?",
    "In what year did the Byzantine emperor Basil II die?",
    "Who was the first Chief Executive of Hong Kong?",
    "At what temperature in Celsius does water boil at sea level?",
    "What is the capital of Burkina Faso?",
    "How many hearts does an octopus have?",
]

if "question" not in st.session_state:
    st.session_state.question = "What is the capital of Australia?"


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

    st.markdown("**Second opinion**")
    second_choice = st.selectbox(
        "Second model", ["Off"] + [m for m in models if m != model],
        help="A different model answers the same question without seeing the first "
             "answer. If it disagrees, a PASS is escalated to VERIFY. Pick a bigger "
             "model from a different family for the best check.")
    second_model = None if second_choice == "Off" else second_choice

    st.markdown("**Verification**")
    use_wiki = st.toggle("Wikipedia", True,
                         help="Fact-check VERIFY answers against Wikipedia (needs internet).")
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
                use_wikipedia=use_wiki, verify_all=verify_all, equivalence=equivalence,
                risk_low=risk_low, risk_high=risk_high, w_conf=w_conf, w_sup=w_sup,
                second_model=second_model)


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


def token_html(tokens: list) -> str:
    parts = []
    for t in tokens:
        alts = " | ".join(f"{a['token']!r} {a['p']:.2f}" for a in t["top"][:5])
        tip = f"p = {t['p']:.3f}\n{alts}"
        c = core.heat_color(t["p"])
        parts.append(f'<span class="tok" style="background:linear-gradient({c},{c}),#fff" '
                     f'title="{escape(tip)}">{escape(t["text"])}</span>')
    return f'<div class="tokens">{"".join(parts)}</div>'


def rivals_chart(entity: dict) -> go.Figure:
    labels = [entity["text"]] + [a["token"].strip() for a in entity["alternatives"]]
    values = [entity["first"]] + [a["p"] for a in entity["alternatives"]]
    colors = ["#43a047"] + ["#ef9a9a"] * len(entity["alternatives"])
    fig = go.Figure(go.Bar(x=values[::-1], y=labels[::-1], orientation="h",
                           marker_color=colors[::-1],
                           text=[f"{v:.0%}" for v in values[::-1]], textposition="outside"))
    fig.update_layout(height=60 + 42 * len(labels), margin=dict(l=10, r=40, t=36, b=10),
                      title=f"What the model nearly said instead of “{entity['text']}”",
                      xaxis=dict(range=[0, 1.1], tickformat=".0%"))
    return fig


def clusters_chart(clusters: list, n: int) -> go.Figure:
    clusters = clusters[:8]
    fig = go.Figure(go.Bar(
        x=[c["size"] for c in clusters][::-1],
        y=[c["answer"] if len(c["answer"]) <= 50 else "…" + c["answer"][-49:]
           for c in clusters][::-1],
        orientation="h",
        marker_color=["#43a047" if c["agrees"] else "#e57373" for c in clusters][::-1],
        text=[f"{c['size']}/{n}" for c in clusters][::-1], textposition="outside"))
    fig.update_layout(height=70 + 40 * len(clusters), margin=dict(l=10, r=30, t=36, b=10),
                      title="Resampled answers grouped by meaning (green = agrees with shown answer)",
                      xaxis=dict(range=[0, n + 0.8], dtick=1))
    return fig


def render_report(r: tr.Report) -> None:
    why = (f"risk {r.risk:.2f} · weakest-fact confidence {r.confidence:.2f} · "
           f"{round(r.support * len(r.samples))}/{len(r.samples)} resamples agree")
    st.markdown(
        f'<div class="verdict v-{r.decision}"><div class="tag">{r.decision} · '
        f'{r.final_status.upper()}</div><div class="ans">{escape(r.final_answer)}</div>'
        f'<div class="why">{STATUS_TEXT.get(r.final_status, "")}<br>{why}</div></div>',
        unsafe_allow_html=True)
    if r.final_answer != r.answer:
        st.caption(f"Model's original answer: “{r.answer}”")

    so = r.second_opinion
    if so:
        if so["agrees"] is None:
            st.warning(f"🤖 Second opinion ({so['model']}) failed: {so.get('error', '')}")
        else:
            verdict = "✅ agrees" if so["agrees"] else "⚠️ disagrees"
            note = " · escalated PASS → VERIFY" if r.escalated else ""
            box = st.info if so["agrees"] else st.warning
            box(f"🤖 **Second opinion** ({so['model']}): {so['answer']}  —  {verdict}{note}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Fact confidence", f"{r.confidence:.0%}",
              help="Weakest answer-bearing token, rival-normalised")
    m2.metric("Sample support", f"{r.support:.0%}",
              help="Share of resamples that say the same thing")
    m3.metric("Semantic entropy", f"{r.entropy:.2f}",
              help="0 = all resamples agree, 1 = all different")
    total = sum(r.timings.values())
    st.caption(f"Scored on: {r.key_source} · alignment: {r.alignment} · "
               f"{total:.1f}s ({', '.join(f'{k} {v:.1f}s' for k, v in r.timings.items())})")
    st.markdown(
        '<div class="legend">'
        f'<span style="background:{core.color_for(0.9)}">≥ {core.ENTITY_SAFE:.0%} sure</span>'
        f'<span style="background:{core.color_for(0.6)}">≥ {core.ENTITY_UNSURE:.0%}</span>'
        f'<span style="background:{core.color_for(0.1)}">&lt; {core.ENTITY_UNSURE:.0%}</span>'
        '</div>', unsafe_allow_html=True)
    st.markdown(entity_html(r.answer, r.entities), unsafe_allow_html=True)

    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["🎯 Facts & rivals", "🔥 Token heatmap", "🔁 Consistency", "📚 Evidence"])

    with tab_a:
        if r.entities:
            weakest = min(r.entities, key=lambda e: e["min"])
            if weakest["alternatives"]:
                st.plotly_chart(rivals_chart(weakest), width="stretch")
            st.dataframe(pd.DataFrame([{
                "fact": e["text"], "type": e["label"],
                "confidence": round(e["min"], 3),
                "raw first-token p": round(e["first"], 3),
                "nearly said": ", ".join(f"{a['token'].strip()} ({a['p']:.0%})"
                                         for a in e["alternatives"]) or "—",
            } for e in r.entities]), width="stretch", hide_index=True)
        else:
            st.info("No answer-bearing tokens found (or the model returned no logprobs).")

    with tab_b:
        st.caption("Every generated token, coloured by probability. Hover for the top-5 "
                   "alternatives. Grammar tokens are shown but not scored.")
        if r.tokens:
            st.markdown(token_html(r.tokens), unsafe_allow_html=True)
            df = pd.DataFrame({"token": [t["text"] for t in r.tokens],
                               "p": [t["p"] for t in r.tokens]})
            df["position"] = range(len(df))
            fig = go.Figure(go.Scatter(x=df["position"], y=df["p"], mode="lines+markers",
                                       text=df["token"], hovertemplate="%{text!r}<br>p=%{y:.3f}"))
            fig.add_hline(y=core.ENTITY_SAFE, line_dash="dash", line_color="green")
            fig.add_hline(y=core.ENTITY_UNSURE, line_dash="dash", line_color="orange")
            fig.update_layout(title="Confidence trajectory", height=280,
                              yaxis=dict(range=[0, 1.05]), xaxis_title="token #",
                              margin=dict(l=10, r=10, t=40, b=30))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("This model/server returned no token logprobs.")

    with tab_c:
        if r.samples:
            st.plotly_chart(clusters_chart(r.clusters, len(r.samples)), width="stretch")
            for i, s in enumerate(r.samples, 1):
                st.markdown(f"**{i}.** {escape(s)}")

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
                src = "🌐 Wikipedia"
                with st.expander(f"{src} · {p['title']}"):
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

page_ask, page_bench = st.tabs(["🔍 Ask", "📊 Benchmark"])

with page_ask:
    with st.expander("Or try a sample question"):
        cols = st.columns(2)
        for k, q in enumerate(SAMPLE_QUESTIONS):
            if cols[k % 2].button(q, key=f"s-{k}", width="stretch"):
                st.session_state.question = q
                st.session_state.autorun = True

    with st.form("ask"):
        st.text_input("Your question", key="question")
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted or st.session_state.pop("autorun", False):
        status = st.status("Running the radar…", expanded=False)
        try:
            report = tr.analyze(st.session_state.question,
                                progress=lambda m: status.update(label=m), **SETTINGS)
            status.update(label="Done", state="complete")
            st.session_state.current = report
        except Exception as exc:                                  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")

    if st.session_state.get("current"):
        render_report(st.session_state.current)

with page_bench:
    st.markdown("Runs the labelled questions in `questions.json` **without verification** "
                "and shows where right and wrong answers were routed.")
    saved = HERE / "results" / "results.csv"
    c1, c2, c3 = st.columns([2, 1, 1])
    tiers = c1.multiselect("Question tiers", ["easy", "medium", "hard", "trick"],
                           ["easy", "medium", "hard", "trick"])
    limit = c2.number_input("Max questions", 1, 500, 84)
    run_bench = c3.button("Run benchmark", type="primary")

    rows = None
    if run_bench:
        qs = benchmark.load_questions(tiers=tiers, limit=int(limit))
        bar = st.progress(0.0, text="starting")

        def prog(i, total, row):
            bar.progress(i / total, text=f"{i}/{total} · {row['question'][:60]}")

        kw = dict(temperature=temperature, mode=mode, equivalence=equivalence,
                  risk_low=risk_low, risk_high=risk_high, w_conf=w_conf, w_sup=w_sup,
                second_model=second_model)
        rows = benchmark.run(model, qs, n_samples=n_samples, progress=prog, **kw)
        benchmark.write_outputs(rows, benchmark.summarize(rows, risk_low, risk_high), model)
    elif saved.exists():
        rows = pd.read_csv(saved).fillna("").to_dict("records")
        meta = json.loads((HERE / "results" / "summary.json").read_text())
        st.caption(f"Showing saved results for **{meta.get('model', '?')}** "
                   "from results/results.csv")
    else:
        st.info("No results yet. Press **Run benchmark** to test the selected model "
                "(a few seconds per question; larger models are slower).")

    if rows:
        summary = benchmark.summarize(rows, risk_low, risk_high)
        auc = summary["auroc"]["fused_trust"]
        m1, m2, _ = st.columns([1, 1, 2])
        m1.metric("Model accuracy", f"{summary['accuracy_raw']:.0%}",
                  help="Share of questions the model answered correctly on its own, "
                       "before the radar does anything.")
        m2.metric("AUROC", f"{auc:.2f}" if auc is not None else "–",
                  help="How well the radar's trust score separates right from wrong "
                       "answers: the chance a correct answer gets more trust than a "
                       "wrong one. 0.5 = coin flip, 1.0 = perfect. Needs at least one "
                       "right and one wrong answer.")

        figs = benchmark.figures(rows)
        a, b = st.columns(2)
        a.plotly_chart(figs[3], width="stretch")      # where answers were routed
        b.plotly_chart(figs[1], width="stretch")      # predicted trust vs accuracy
        cols = ["tier", "question", "answer", "gold", "correct",
                "decision", "risk", "confidence", "support", "rivals"]
        if any(r.get("second_answer") for r in rows):
            cols += ["second_answer", "second_agrees"]
        df = pd.DataFrame(rows)[cols]
        df["correct"] = df["correct"].map({1: "✅", 0: "❌"})
        st.dataframe(df, width="stretch", hide_index=True, height=420)
