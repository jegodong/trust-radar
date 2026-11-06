"""cli.py -- Trust Radar in the terminal.

    python cli.py "What is the capital of Australia?"
    python cli.py --demo                       # curated showcase questions
    python cli.py --verify-all "In what year did Basil II die?"
    python cli.py --json "Who wrote 1984?"     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys

import llm
import radar_core as core
import trust_radar as tr

DEMO = [
    "Who was the first person to walk on the Moon?",
    "What is the capital of Australia?",
    "Who won the 1987 Nobel Prize in Literature?",
    "What is the capital of Switzerland?",
    "In what year did the Byzantine emperor Basil II die?",
    "Who was the first Chief Executive of Hong Kong?",
]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {"high": "\033[42;30m", "medium": "\033[43;30m", "low": "\033[41;97m"}
DECISION = {core.PASS: "\033[1;32m", core.VERIFY: "\033[1;33m", core.ABSTAIN: "\033[1;31m"}


def paint_answer(answer: str, entities: list) -> str:
    out, cursor = [], 0
    for e in sorted(entities, key=lambda e: e["start"]):
        if e["start"] < cursor:
            continue
        out.append(answer[cursor:e["start"]])
        out.append(f"{COLORS[e['level']]}{answer[e['start']:e['end']]}{RESET}")
        cursor = e["end"]
    out.append(answer[cursor:])
    return "".join(out)


def bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(r: tr.Report) -> None:
    print(f"\n{BOLD}Q:{RESET} {r.question}")
    print(f"{BOLD}Model said:{RESET} {paint_answer(r.answer, r.entities)}")
    for e in r.entities:
        rivals = ", ".join(f"{a['token'].strip()} {a['p']:.0%}" for a in e["alternatives"]) or "-"
        print(f"   {DIM}{e['label']:<9}{RESET} {e['text'][:30]:<30} "
              f"conf {e['min']:.2f}  {DIM}nearly said: {rivals}{RESET}")
    print(f"   confidence {bar(r.confidence)} {r.confidence:.2f}")
    print(f"   support    {bar(r.support)} {r.support:.2f}  "
          f"({round(r.support * len(r.samples))}/{len(r.samples)} samples agree)")
    print(f"   risk       {bar(r.risk)} {r.risk:.2f}")
    short = [s if len(s) <= 42 else "…" + s[-41:] for s in r.samples]
    print(f"   {DIM}samples: {' | '.join(short)}{RESET}")
    if r.second_opinion:
        so = r.second_opinion
        mark = {True: "agrees", False: "DISAGREES", None: "failed"}[so["agrees"]]
        esc = " → escalated PASS to VERIFY" if r.escalated else ""
        print(f"   second opinion ({so['model']}): {so['answer'][:60]}  [{mark}]{esc}")
    if r.verification:
        v = r.verification
        src = ", ".join(sorted({p['source'] for p in v['passages']})) or "none"
        print(f"   verify: {v['status']} (sources: {src}) → {v['rag_answer'][:80]}")
    print(f"{DECISION[r.decision]}{r.decision}{RESET} → {BOLD}{r.final_answer}{RESET} "
          f"{DIM}[{r.final_status}]{RESET}")


def main():
    ap = argparse.ArgumentParser(description="Trust Radar CLI")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=5, help="number of resamples")
    ap.add_argument("--demo", action="store_true", help="run the showcase questions")
    ap.add_argument("--no-wiki", action="store_true", help="skip Wikipedia fact-checking")
    ap.add_argument("--verify-all", action="store_true", help="also fact-check PASS answers")
    ap.add_argument("--second", default=None, help="second-opinion model, e.g. gemma4:12b")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not llm.server_ok():
        sys.exit(f"Cannot reach Ollama at {llm.HOST}. Start it with `ollama serve`.")

    questions = DEMO if args.demo else [" ".join(args.question)]
    if not questions[0]:
        ap.error("give a question or --demo")

    for q in questions:
        r = tr.analyze(q, model=args.model, n_samples=args.n,
                       use_wikipedia=not args.no_wiki, verify_all=args.verify_all,
                       second_model=args.second)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2, default=str))
        else:
            print_report(r)


if __name__ == "__main__":
    main()
