# Trust Radar: Confidence Radar for LLMs

LLMs state false facts with the same confidence as true ones. Trust Radar sits between a
local Ollama model and the user. It measures how sure the model really was about each fact,
checks whether the model gives the same answer when asked again, and then **passes**,
**fact-checks** or **refuses** the answer.

---

## Setup

1. Install [Python 3.10+](https://www.python.org/downloads/) and [Ollama](https://ollama.com/).
2. Pull a model:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
3. Install the Python packages and the spaCy English model:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. Start the dashboard:
   ```bash
   streamlit run dashboard.py
   ```
   Open http://localhost:8501, pick a model in the sidebar, and ask a question.

Wikipedia fact-checking needs an internet connection. Everything else runs offline.

---

## How it works

For every question, `trust_radar.analyze()` runs these steps:

| Step | What happens |
|---|---|
| 1. Answer | The model answers once at temperature 0, and Ollama returns each token's probability (logprobs) and its top-5 alternatives. |
| 2. Find the facts | spaCy finds the names, places, dates and numbers in the answer. Words copied from the question are ignored. |
| 3. Fact confidence | Each fact's confidence is its **weakest** token. Alternatives such as "The" are phrasing, not rival answers, so they are not counted against it. |
| 4. Sample support | The question is asked 5 more times at temperature 0.8. **Support** is the share of those answers that say the same thing as the shown answer. |
| 5. Risk score | `risk = 1 − (0.4 × confidence + 0.6 × support)` → **PASS** (< 0.35) / **VERIFY** (0.35–0.65) / **ABSTAIN** (≥ 0.65) |
| 6. Second opinion *(optional)* | A different model answers the same question without seeing the first answer. If it disagrees with a PASS, the answer is escalated to VERIFY. |
| 7. Wikipedia check | VERIFY and ABSTAIN answers are checked on Wikipedia. The model re-answers using only the retrieved sentences. The answer is then **verified**, **corrected**, or (for ABSTAIN with no evidence) **refused**. |

**Wikipedia search.** Searching with the raw question returns wrong pages (e.g. the film
*A Walk on the Moon*). Instead, the radar searches with the question's keywords
("Byzantine emperor Basil II die") and with every candidate answer by name ("Canberra",
"Sydney"), then keeps only the sentences that mention the question or a candidate.

---

## Using the dashboard

**Ask tab**
- Type a question, or pick one from **Or try a sample question**.
- The banner shows the decision, the final answer and why.
- 🎯 **Facts & rivals**: what the model nearly said instead (e.g. Sydney 55% vs Canberra 26%)
- 🔥 **Token heatmap**: every token coloured by probability; hover for alternatives
- 🔁 **Consistency**: the 5 re-asked answers, grouped by meaning
- 📚 **Evidence**: the Wikipedia sentences used and the evidence-based answer

**Sidebar**
- **Model**: any model installed in Ollama
- **Second model**: a second-opinion model, or Off
- **Wikipedia**: turn fact-checking on or off
- **Verify-all**: also fact-check PASS answers (catches confident, repeated mistakes)
- **Advanced**: answer matching, risk thresholds, support weight

**Benchmark tab**
- Press **Run benchmark** to run the labelled questions in `questions.json` on the selected model.
- Shows model accuracy, AUROC, where right and wrong answers were routed, and predicted trust vs actual accuracy.
- Results are saved to `results/` (created on the first run).

---

## Command line

```bash
python cli.py "What is the capital of Australia?"
python cli.py --demo                                  # six showcase questions
python cli.py --verify-all "In what year did Basil II die?"
python cli.py --second gemma4:12b "Who won the 1987 Nobel Prize in Literature?"
python cli.py --json "Who wrote 1984?"                # machine-readable output

python benchmark.py --model qwen2.5:1.5b              # all 84 questions
python benchmark.py --model qwen2.5:1.5b --tier hard trick --limit 20
python benchmark.py --model qwen2.5:1.5b --second gemma4:12b
```

| Option | `cli.py` | `benchmark.py` |
|---|---|---|
| `--model` | model to test | model to test |
| `--n` | number of resamples (default 5) | number of resamples (default 5) |
| `--second` | second-opinion model | second-opinion model |
| `--verify-all` | also fact-check PASS answers | – |
| `--no-wiki` | skip Wikipedia | – (benchmark never uses Wikipedia) |
| `--tier`, `--limit` | – | choose question tiers / how many |

Environment variables:
- `OLLAMA_HOST`: Ollama on another machine, e.g. `http://192.168.1.50:11434`
- `TRUST_RADAR_MODEL`: default model (otherwise `qwen2.5:3b`)

---

## Sample questions

| Question | What usually happens |
|---|---|
| Who was the first person to walk on the Moon? | PASS: confident, all samples agree |
| What is the capital of Australia? | VERIFY: Sydney vs Canberra; Wikipedia corrects it |
| Who won the 1987 Nobel Prize in Literature? | ABSTAIN on small models; Wikipedia finds Joseph Brodsky |
| In what year did the Byzantine emperor Basil II die? | PASS with a wrong year; caught by Verify-all or a second model |

`questions.json` has 84 questions with accepted answers, in four tiers: easy, medium,
hard (obscure facts) and trick (common misconceptions).

---

## Files

| File | Role |
|---|---|
| `dashboard.py` | Streamlit web app (Ask and Benchmark tabs) |
| `trust_radar.py` | The pipeline: answer → score → resample → risk → second opinion → Wikipedia |
| `radar_core.py` | The maths: answer matching, support, risk score, triage, grading, AUROC |
| `retrieval.py` | Wikipedia search and sentence selection |
| `llm.py` | Ollama client: logprobs, top-5 alternatives, thinking turned off |
| `benchmark.py` | Runs the labelled questions and draws the benchmark charts |
| `cli.py` | Terminal version |
| `questions.json` | 84 questions with accepted answers |
| `tests/` | 25 automated tests (no Ollama needed): `python -m pytest -q` |
