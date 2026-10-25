import streamlit as st
import pandas as pd
import plotly.express as px
import trust_radar
from html import escape

#three color green yellow red
def score_to_color(avg: float) -> str:
    if avg >= 0.80:
        return "#c8e6c9"
    if avg >= 0.50:
        return "#fff9c4"
    return "#ffcdd2"

def render_annotated(text: str, entities: list):
    entities = sorted(entities, key=lambda e: e["start"])
    parts = []
    cursor = 0
    for ent in entities:
        if ent["start"] > cursor:
            parts.append(escape(text[cursor:ent["start"]]))
        color = score_to_color(ent["avg"])
        label = f"{ent['label']} | avg={ent['avg']:.2f} min={ent['min']:.2f}"
        parts.append(
            f'<span style="background-color:{color}; padding:2px 5px; '
            f'border-radius:4px;" title="{label}">{escape(ent["text"])}</span>'
        )
        cursor = ent["end"]
    if cursor < len(text):
        parts.append(escape(text[cursor:]))
    st.markdown("".join(parts), unsafe_allow_html=True)

# ---------- UI  ----------
st.set_page_config(page_title="Trust Radar", layout="wide")
st.title("Trust Radar")
st.caption(
    "Token logprobs are mapped to factual entities. "
    "Green = high (>=0.80), Yellow = medium (0.50-0.80), Red = low (<0.50)."
)

with st.sidebar:
    st.header("Input")
    model = st.text_input("Model", value="qwen2.5:3b")
    prompt = st.text_area(
        "Prompt",
        value="The first person to walk on the Moon was",
        height=100,
    )
    n_samples = st.slider("Multi-sample count", min_value=3, max_value=5, value=5)
    run = st.button("Generate & Analyze", type="primary")

if run:
    with st.spinner("Generating and analyzing single sample..."):
        try:
            result = trust_radar.analyze(model, prompt)
            text = result["text"]
            entities = result["entities"]
        except Exception as e:
            st.error(f"Error in single-sample analysis: {e}")
            st.stop()

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
                "Avg Prob": round(e["avg"], 3),
                "Min Prob": round(e["min"], 3),
                "Tokens": " ".join(e["tokens"]),
            }
            for e in entities
        ]
        st.dataframe(rows, use_container_width=True)

        df = pd.DataFrame(rows)
        fig = px.bar(
            df, x="Entity", y="Avg Prob", color="Type",
            range_y=[0, 1], title="Entity Confidence Scores",
        )
        fig.add_hline(y=0.80, line_dash="dash", line_color="green")
        fig.add_hline(y=0.50, line_dash="dash", line_color="orange")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No factual entities detected in the generated text.")

    st.subheader("Multi-Sample Consistency")
    with st.spinner(f"Running {n_samples} samples, please wait..."):
        try:
            ms = trust_radar.multi_sample_check(model, prompt, n=n_samples)
        except Exception as e:
            st.error(f"Error in multi-sample analysis: {e}")
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Consistency", f"{ms['consistency']:.2f}")
    col2.metric("Unique answers", ms["unique_count"])
    col3.metric("Top answer", ms["top_entity"][:20])

    if ms["level"] == "success":
        st.success(ms["verdict"])
    elif ms["level"] == "warning":
        st.warning(ms["verdict"])
    else:
        st.error(ms["verdict"])

    with st.expander(f"Show {n_samples} sampled answers"):
        for i, ans in enumerate(ms["samples"], 1):
            st.write(f"**{i}.** {ans[:150]}")