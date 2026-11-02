"""dashboard.py -- interactive Trust Radar demo.

Run with:  streamlit run dashboard.py

The maths lives in radar_core.py (shared with the batch evaluation) and the
single-question pipeline lives in trust_radar.py.  This file only renders.
"""

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

import radar_core as core
import trust_radar


def render_annotated(text: str, entities: list) -> None:
    """Print the answer with each factual entity coloured by its confidence."""
    entities = sorted(entities, key=lambda e: e["start"])
    parts = []
    cursor = 0
    for ent in entities:
        if ent["start"] > cursor:
            parts.append(escape(text[cursor:ent["start"]]))
        colour = core.entity_color(ent["avg"])
        tooltip = f"{ent['label']} | avg={ent['avg']:.2f} min={ent['min']:.2f}"
        parts.append(
            f'<span style="background-color:{colour}; padding:2px 5px; '
            f'border-radius:4px;" title="{tooltip}">{escape(ent["text"])}</span>'
        )
        cursor = ent["end"]
    if cursor < len(text):
        parts.append(escape(text[cursor:]))
    st.markdown("".join(parts), unsafe_allow_html=True)


def level_box(level: str, message: str) -> None:
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


# ---------- page ----------
st.set_page_config(page_title="Trust Radar", layout="wide")
st.title("Trust Radar")
st.caption(
    f"Entity colour = mean token probability "
    f"(green >= {core.ENTITY_SAFE:.2f}, yellow >= {core.ENTITY_UNSURE:.2f}, red below). "
    f"SINdex is a normalised semantic-inconsistency score: "
    f"below {core.SINDEX_STABLE:.2f} samples agree, above {core.SINDEX_HIGH:.2f} they disagree."
)

with st.sidebar:
    st.header("Input")
    model = st.text_input("Model", value="qwen2.5:3b")
    prompt = st.text_area(
        "Prompt",
        value="The first person to walk on the Moon was",
        height=100,
    )
    n_samples = st.slider("Multi-sample count", min_value=3, max_value=10, value=5)
    run = st.button("Generate & Analyze", type="primary")

if run:
    with st.spinner("Generating and analyzing single sample..."):
        try:
            result = trust_radar.analyze(model, prompt)
            text = result["text"]
            entities = result["entities"]
        except Exception as exc:                      # noqa: BLE001
            st.error(f"Error in single-sample analysis: {exc}")
            st.stop()

    if result.get("alignment") not in (None, "exact"):
        st.caption(f"note: token alignment fell back to '{result['alignment']}'")

    st.subheader("Generated Text")
    if entities:
        render_annotated(text, entities)
    else:
        st.write(text)

    st.subheader("Entity Confidence")
    if entities:
        rows = [
            {
                "Entity": e["text"],
                "Type": e["label"],
                "Mean prob": round(e["avg"], 3),
                "Min prob": round(e["min"], 3),
            }
            for e in entities
        ]
        st.dataframe(rows, use_container_width=True)

        df = pd.DataFrame(rows)
        fig = px.bar(
            df, x="Entity", y="Mean prob", color="Type",
            range_y=[0, 1], title="Entity confidence (mean token probability)",
        )
        fig.add_hline(y=core.ENTITY_SAFE, line_dash="dash", line_color="green")
        fig.add_hline(y=core.ENTITY_UNSURE, line_dash="dash", line_color="orange")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No factual entities detected in the generated text.")

    # ---------- one consistency verdict, not two ----------
    st.subheader("Cross-Sample Consistency")
    with st.spinner(f"Sampling {n_samples} answers, please wait..."):
        try:
            ms = trust_radar.multi_sample_check(model, prompt, n=n_samples)
        except Exception as exc:                      # noqa: BLE001
            st.error(f"Error in multi-sample analysis: {exc}")
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Exact agreement", f"{ms['consistency']:.0%}")
    col2.metric("Semantic clusters", ms["sindex_n_clusters"])
    col3.metric("Top answer", (ms["top_entity"] or "-")[:20])

    level_box(ms["level"], f"SINdex {ms['sindex']:.2f} — {ms['verdict']}")

    with st.expander(f"Show {n_samples} sampled answers"):
        for i, ans in enumerate(ms["samples"], 1):
            st.write(f"**{i}.** {(ans or '(empty)')[:200]}")

    with st.expander("Semantic cluster details"):
        st.caption(
            "Coherence = mean pairwise similarity inside a cluster. "
            "Singletons have no internal disagreement, so coherence is 1.0 by convention."
        )
        for c in ms["sindex_clusters"]:
            st.write(
                f"**{(c['answer'] or '(empty)')[:60]}** — "
                f"size={c['size']}, coherence={c['coherence']}, "
                f"share={c['p']}, weight={c['weight']}"
            )
