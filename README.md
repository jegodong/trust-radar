# 📡 Trust Radar

**Spot LLM hallucinations before you trust them.** Ask a local Ollama model a factual
question. Trust Radar colours the facts in the answer by how sure the model really was,
re-asks the question several times to see if the answer holds, fuses both into one risk
score, and then **passes**, **fact-checks** or **refuses** the answer.

![Australia question: the model says Sydney, the radar flags it, checks the evidence and corrects it to Canberra](docs/1_ask_australia.png)

> The screenshot above is from an earlier build: the risk gauge and the separate token
> heatmap tab have since been removed, and the verdict is now shown in plain language
> (*Safe to use* / *Check this before you trust it* / *Not reliable enough to answer*).

## What it does

| Feature | What you see |
|---|---|
| **Fact confidence** | Names, places, dates and numbers in the answer are highlighted green / yellow / red. Grammar words and words copied from the question are ignored. Colouring can be switched off in the sidebar. |
| **"What it nearly said"** | The runner-up answers the model was considering, shown only when it was genuinely torn (a rival above 5%). |
| **Cross-sample consistency** | The question is re-asked N times. Answers are grouped by meaning, and we count how many agree with the answer you are shown. |
| **One plain-language verdict** | 🟢 *Safe to use* / 🟡 *Check this before you trust it* / 🔴 *Not reliable enough to answer*, with the technical label and the numbers underneath. |
| **Evidence check (RAG)** | VERIFY answers are checked against your own `knowledge/` folder (offline) and then Wikipedia. The model re-answers from the evidence, and the answer is kept, **corrected** or flagged. |
| **Verify-all (paranoid) mode** | Also fact-checks low-risk answers. This catches errors the model makes confidently and consistently. |
| **Benchmark tab** | Runs 48 labelled questions and reports AUROC per signal, calibration (ECE/Brier), a coverage-vs-accuracy curve and where right and wrong answers were routed. |
| **CLI + JSON** | `python cli.py "question"`, with a colour terminal output or `--json` |

## Quick start

```bash
# 1. Install Ollama (https://ollama.com) and pull a model
ollama pull qwen2.5:3b            # any Ollama model works; bigger = more accurate

# 2. Python deps + spaCy English model
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run
streamlit run dashboard.py        # web app at http://localhost:8501
python cli.py --demo              # or: terminal showcase
python benchmark.py --model qwen2.5:3b   # full evaluation -> results/report.html
python -m pytest -q               # 21 tests, no Ollama needed
```

Options (environment variables):

* `TRUST_RADAR_MODEL`: default model (default `qwen2.5:3b`)
* `OLLAMA_HOST`: point at Ollama on another machine, e.g. `http://192.168.1.50:11434`
* `TRUST_RADAR_KB`: a different knowledge-base folder

Thinking models (Qwen3, DeepSeek-R1) are handled automatically: thinking is switched off
so the token budget goes to the answer.

## Sample questions to try

| Group | Questions | What usually happens |
|---|---|---|
| Confident & right | *Who was the first person to walk on the Moon?* · *Who wrote the novel 1984?* | 🟢 PASS: high confidence, all samples agree |
| Torn between answers | *What is the capital of Switzerland?* · *What is the largest ocean on Earth?* | 🟡 VERIFY: rivals visible, samples split, evidence corrects it |
| Long-tail facts | *Who won the 1987 Nobel Prize in Literature?* · *What is the capital of Burkina Faso?* | 🟡 VERIFY or 🔴 ABSTAIN: the model invents a different name each time |
| Confidently wrong | *In what year did the Byzantine emperor Basil II die?* · *What color is a polar bear's skin?* | 🟢 PASS with a wrong answer. Turn on **Verify-all** to catch it |

All 48 labelled questions (easy / medium / hard / trick, with gold answers) are in
`questions.json`.

## Real results

`python benchmark.py --model qwen2.5:3b`, 48 questions, verification off so the knowledge
base cannot leak answers into the score:

| Metric | Value |
|---|---|
| Model accuracy on its own | **87.5%** |
| Accuracy of answers the radar PASSed | **93.2%** (44 questions) |
| Right answers allowed through | **97.6%** |
| Wrong answers routed to VERIFY or ABSTAIN | **50%** (3 of 6) |
| AUROC: token confidence alone | 0.841 |
| AUROC: sample support alone | 0.724 |
| AUROC: **fused risk score** | 0.835 |
| ECE: token confidence / fused | 0.061 / 0.098 |

![Benchmark tab](docs/6_benchmark.png)

**The fusion only helps when the model is unstable.** An earlier run used Qwen3 0.6B, a
deliberately weak model that got 52% right and disagreed with itself constantly. There,
sample support reached AUROC 0.854 and the fused score 0.868 — clearly better than
confidence alone (0.793). On a capable 3B model the resamples almost always agree, so
support carries little information and the fused score merely matches token confidence.
That is a property of the model, not a bug in the fusion.

**Honest limitation.** A model that is *confidently and consistently* wrong passes both
signals. On qwen2.5:3b the radar let through *Sahara* for "largest desert", *Yangon* for
"capital of Myanmar" and *White* for "polar bear's skin". Neither signal can see those;
only external evidence can, which is what **Verify-all** mode does.

## How the scoring works

1. **Answer-bearing tokens only.** spaCy finds entities. Words already in the question and
   stopwords are dropped, so *"The capital of Australia is **Sydney**"* is scored on
   *Sydney* alone.
2. **Weakest link, rivals only.** An entity's confidence is its weakest token. Inside a
   name, the first token is where the choice happens ("Neil" 0.61 → "Armstrong" 0.996), so
   averaging would hide the doubt. At that first token, alternatives such as *"The"* are
   phrasing choices, not rival facts, so the probability is renormalised over competing
   answers only.
3. **Two different numbers, on purpose.** The **answer confidence** shown at the top is the
   confidence of the fact that answers the question. The **risk score** uses the weakest
   fact anywhere in the answer, so a volunteered detail still lowers trust. A weak detail
   can push a PASS down to VERIFY, but it can never turn a confident answer into a refusal.
4. **Support.** N resamples are taken at temperature 0.8 and grouped by meaning. Two
   answers match if their answer-bearing words overlap (≥ 50% of the shorter answer) and
   their numbers don't contradict each other: *Armstrong* matches *Neil Armstrong*, *1969*
   matches *July 20, 1969*, *26* does not match *27*.
5. **Triage.** PASS if risk < 0.35, ABSTAIN if risk ≥ 0.65, VERIFY in between. Both
   thresholds and the weights can be adjusted in the sidebar.

### Reading the numbers

* **Colour is not correctness.** It is how sure the model was about that word. Rare
  spellings (*Ouagadougou*, *Naypyidaw*) score low even when the model gets them right.
* **Semantic entropy is coarse.** With 5 samples it can only take a handful of values — a
  4+1 split scores 0.31, 3+2 scores 0.42, 3+1+1 scores 0.59. Read it as evidence, not a
  precise score.
* **Support asks a different question from entropy.** Support asks "do the resamples back
  up the answer you are shown"; entropy asks "how spread out are they". An answer that no
  sample supports can still have entropy 0 if the samples all agree with each other.

## Your own knowledge base

Put `.md` or `.txt` files in `knowledge/`, such as course notes, company policies or a
product manual. They're searched first, with BM25, fully offline, before Wikipedia. Use `#`
headings to split a file into passages. The included files cover the demo questions.

Wikipedia needs network access to `en.wikipedia.org`; if it is unreachable the check
silently falls back to the local knowledge base.

## Project layout

| File | Role |
|---|---|
| `dashboard.py` | Streamlit app: Ask / Benchmark / History / How it works |
| `cli.py` | Terminal interface (`--demo`, `--verify-all`, `--json`) |
| `trust_radar.py` | Pipeline: answer → score → resample → fuse → triage → verify |
| `radar_core.py` | Pure maths: normalisation, equivalence, clustering, entropy, risk, AUROC/ECE |
| `retrieval.py` | Local BM25 knowledge base + Wikipedia search |
| `llm.py` | Ollama client wrapper (logprobs, top-k, thinking off, remote host) |
| `benchmark.py` | Labelled evaluation → `results/` (CSV, JSON, HTML report) |
| `questions.json` | 48 questions with gold answers |
| `knowledge/` | Local evidence for the VERIFY tier |
| `tests/` | 21 unit tests (fake model, no server needed) |
