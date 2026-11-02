# trust-radar-backup

A confidence radar for LLM answers. Ask a local model a factual question; the
dashboard colours every name, number and date by how sure the model was about
that exact word, and asks the same question several times to see whether the
answer stays the same.

### 1. Prerequisites (Install these first)
- Install [Python](https://www.python.org/downloads/) (Make sure to check "Add Python to PATH").
- Install [Ollama](https://ollama.com/) (Download the Windows/Mac installer and run it).

### 2. Download the Model
Open your terminal (PowerShell/CMD) and pull the default model:
- ollama pull qwen2.5:3b

### 3. Install Python Dependencies
Clone this repository or download the ZIP. Open your terminal in the project folder, and run:
- pip install -r requirements.txt

### 4. Download the spaCy Language Model
This step is mandatory (requirements.txt cannot install it directly):
- python -m spacy download en_core_web_sm

### 5. Run the Dashboard
- streamlit run dashboard.py

Then open http://localhost:8501, type a question and press **Generate & Analyze**.

## What you see

- The answer with every factual entity coloured by the model's mean token
  probability: green (>= 0.80), yellow (>= 0.50), red (below).
- A table and a bar chart of per-entity confidence.
- **Cross-sample consistency**: the question is asked 5-10 times at
  temperature 0.8, the answers are grouped by meaning, and one verdict is shown.

## The two signals

| signal | asks | misses |
|---|---|---|
| entity confidence | "is the model sure about this word?" | fluent fabrication |
| SINdex | "does the model keep saying the same thing?" | consistent errors |

SINdex is a normalised entropy over the answer groups: 0.00 means every sample
agreed, 1.00 means maximum disagreement. **0.00 means consistent, not correct** —
asked for the boiling point of water on Everest the model answers 100 C every
time (wrong) and still scores 0.00.

## Optional: semantic grouping

The folder `all-MiniLM-L6-v2/` (930 MB) is not included. Without it the app
still runs, but falls back to exact string matching and prints a warning, so
"Canberra" and "Canberra, 1908" count as different answers. To enable it:

    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir all-MiniLM-L6-v2

## Project layout

| file | role |
|---|---|
| `dashboard.py` | renders the page |
| `trust_radar.py` | asks the model, extracts entities, scores them |
| `radar_core.py` | shared maths: text normalisation, grouping, SINdex, colour bands |
