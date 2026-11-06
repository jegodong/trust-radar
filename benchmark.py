"""benchmark.py -- does the radar actually separate right from wrong answers?

    python benchmark.py --model qwen2.5:3b            # all questions
    python benchmark.py --model qwen2.5:3b --limit 10 # quick check
    python benchmark.py --tier hard trick             # subsets

Every question in questions.json has gold aliases.  For each question we run
the radar WITHOUT verification (so retrieved evidence cannot leak answers into
the score) and record whether the model's raw answer was correct.

Outputs (in ./results):
    results.csv     one row per question
    summary.json    accuracy, AUROC per signal, ECE, Brier, triage outcome
    report.html     interactive charts (open in a browser)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import radar_core as core
import trust_radar as tr

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load_questions(path=HERE / "questions.json", tiers=None, limit=None) -> list:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if tiers:
        items = [x for x in items if x.get("tier") in tiers]
    return items[:limit] if limit else items


def run(model: str, questions: list, n_samples: int = 5, progress=None, **kw) -> list:
    rows = []
    for i, item in enumerate(questions):
        t0 = time.time()
        r = tr.analyze(item["q"], model=model, n_samples=n_samples, do_verify=False, **kw)
        correct = core.is_correct(r.answer, item["gold"])
        rows.append({
            "question": item["q"],
            "tier": item.get("tier", ""),
            "gold": " | ".join(item["gold"]),
            "answer": r.answer,
            "correct": int(correct),
            "confidence": round(r.confidence, 4),
            "support": round(r.support, 4),
            "entropy": round(r.entropy, 4),
            "risk": round(r.risk, 4),
            "trust": round(1 - r.risk, 4),
            "decision": r.decision,
            "weakest_entity": min(r.entities, key=lambda e: e["min"])["text"] if r.entities else "",
            "rivals": ", ".join(f"{a['token'].strip()} {a['p']:.0%}"
                                for a in (min(r.entities, key=lambda e: e["min"])["alternatives"]
                                          if r.entities else [])),
            "samples": " || ".join(r.samples),
            "second_answer": (r.second_opinion or {}).get("answer", ""),
            "second_agrees": {True: "yes", False: "no", None: ""}[(r.second_opinion or {}).get("agrees")],
            "escalated": int(r.escalated),
            "seconds": round(time.time() - t0, 2),
        })
        if progress:
            progress(i + 1, len(questions), rows[-1])
    return rows


def summarize(rows: list, risk_low=core.RISK_LOW, risk_high=core.RISK_HIGH) -> dict:
    y = [r["correct"] for r in rows]
    n = len(rows)

    def auc(key):
        v = core.auroc([r[key] for r in rows], y)
        return None if v is None else round(v, 3)

    passed = [r for r in rows if r["decision"] == core.PASS]
    verify = [r for r in rows if r["decision"] == core.VERIFY]
    abstain = [r for r in rows if r["decision"] == core.ABSTAIN]

    def acc(group):
        return round(sum(r["correct"] for r in group) / len(group), 3) if group else None

    trust = [r["trust"] for r in rows]
    conf = [r["confidence"] for r in rows]
    return {
        "n": n,
        "accuracy_raw": acc(rows),
        "auroc": {
            "token_confidence": auc("confidence"),
            "sample_support": auc("support"),
            "fused_trust": auc("trust"),
        },
        "ece": {
            "token_confidence": round(core.ece(conf, y), 3),
            "fused_trust": round(core.ece(trust, y), 3),
        },
        "brier_fused_trust": round(core.brier(trust, y), 3),
        "aurc_fused_trust": round(core.aurc(trust, y), 3),
        "triage": {
            "PASS": {"count": len(passed), "accuracy": acc(passed)},
            "VERIFY": {"count": len(verify), "accuracy": acc(verify)},
            "ABSTAIN": {"count": len(abstain), "accuracy": acc(abstain)},
        },
        "wrong_answers_caught": (
            sum(1 for r in rows if not r["correct"] and r["decision"] != core.PASS)
            / max(1, sum(1 for r in rows if not r["correct"]))
        ),
        "right_answers_passed": (
            sum(1 for r in rows if r["correct"] and r["decision"] == core.PASS)
            / max(1, sum(1 for r in rows if r["correct"]))
        ),
        "thresholds": {"risk_low": risk_low, "risk_high": risk_high},
    }


def figures(rows: list) -> list:
    """Plotly figures shared by report.html and the dashboard."""
    import plotly.graph_objects as go

    y = [r["correct"] for r in rows]
    figs = []

    # 1. risk vs correctness
    fig = go.Figure()
    for ok, name, color in [(1, "correct", "#2e7d32"), (0, "wrong", "#c62828")]:
        pts = [r for r in rows if r["correct"] == ok]
        fig.add_trace(go.Scatter(
            x=[r["confidence"] for r in pts], y=[r["support"] for r in pts],
            mode="markers", name=name, marker=dict(size=11, color=color, opacity=0.75),
            text=[f"{r['question']}<br>→ {r['answer']}" for r in pts], hoverinfo="text"))
    fig.update_layout(title="Every answer: token confidence vs sample support",
                      xaxis_title="weakest-fact token confidence",
                      yaxis_title="share of samples that agree",
                      xaxis_range=[-0.03, 1.03], yaxis_range=[-0.05, 1.05], height=420)
    figs.append(fig)

    # 2. reliability diagram
    bins = core.reliability_bins([r["trust"] for r in rows], y, bins=5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(dash="dash", color="#999")))
    fig.add_trace(go.Scatter(x=[b["confidence"] for b in bins], y=[b["accuracy"] for b in bins],
                             mode="lines+markers", name="Trust Radar",
                             marker=dict(size=[8 + 2 * b["count"] for b in bins]),
                             text=[f"{b['count']} answers" for b in bins]))
    fig.update_layout(title="Reliability: predicted trust vs actual accuracy",
                      xaxis_title="trust (1 - risk)", yaxis_title="accuracy",
                      xaxis_range=[0, 1], yaxis_range=[0, 1.05], height=380)
    figs.append(fig)

    # 3. coverage vs accuracy
    curve = core.coverage_curve([r["trust"] for r in rows], y)
    fig = go.Figure(go.Scatter(x=[c["coverage"] for c in curve], y=[c["accuracy"] for c in curve],
                               mode="lines", fill="tozeroy", name="answered"))
    fig.add_hline(y=sum(y) / len(y), line_dash="dot",
                  annotation_text="accuracy if the model answers everything")
    fig.update_layout(title="Answer less, be right more: accuracy vs share answered",
                      xaxis_title="share of questions answered (most trusted first)",
                      yaxis_title="accuracy of answered questions",
                      yaxis_range=[0, 1.05], height=380)
    figs.append(fig)

    # 4. triage outcome
    fig = go.Figure()
    for ok, name, color in [(1, "correct", "#66bb6a"), (0, "wrong", "#ef5350")]:
        fig.add_trace(go.Bar(
            x=[core.PASS, core.VERIFY, core.ABSTAIN],
            y=[sum(1 for r in rows if r["decision"] == d and r["correct"] == ok)
               for d in (core.PASS, core.VERIFY, core.ABSTAIN)],
            name=name, marker_color=color))
    fig.update_layout(barmode="stack", title="Where right and wrong answers were routed",
                      yaxis_title="questions", height=380)
    figs.append(fig)
    return figs


def write_outputs(rows: list, summary: dict, model: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "summary.json").write_text(json.dumps({"model": model, **summary}, indent=2))

    figs = figures(rows)
    html = [f"<html><head><meta charset='utf-8'><title>Trust Radar benchmark - {model}</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:1000px;margin:auto;padding:20px}"
            "pre{background:#f5f5f5;padding:12px;border-radius:6px}</style></head><body>",
            f"<h1>Trust Radar benchmark</h1><p>Model: <b>{model}</b> &middot; "
            f"{summary['n']} questions</p><pre>{json.dumps(summary, indent=2)}</pre>"]
    for i, fig in enumerate(figs):
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html.append("</body></html>")
    (RESULTS / "report.html").write_text("\n".join(html), encoding="utf-8")
    return RESULTS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=tr.llm.DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=5, help="samples per question")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tier", nargs="*", default=None, help="easy medium hard trick")
    ap.add_argument("--second", default=None, help="second-opinion model, e.g. gemma4:12b")
    args = ap.parse_args()

    questions = load_questions(tiers=args.tier, limit=args.limit)
    extra = f", second opinion: {args.second}" if args.second else ""
    print(f"Running {len(questions)} questions on {args.model} (n={args.n}{extra})\n")

    def show(i, total, row):
        mark = "✓" if row["correct"] else "✗"
        print(f"[{i:>2}/{total}] {mark} {row['decision']:<7} risk={row['risk']:.2f}  "
              f"{row['question'][:48]:<48} → {row['answer'][:40]}")

    rows = run(args.model, questions, n_samples=args.n, progress=show, second_model=args.second)
    summary = summarize(rows)
    out = write_outputs(rows, summary, args.model)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {out/'results.csv'}, {out/'summary.json'}, {out/'report.html'}")


if __name__ == "__main__":
    main()
