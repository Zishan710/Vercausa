"""Phase 8 evaluation: GraphRAG (Reasoning Layer) with the local LLM.

Runs the 40 hand-written questions in data/graphrag_questions.json -- worded
the way an operator might ask, not copied from the prompts -- against the
knowledge graph after replaying all of AI4I. Needs Ollama running with the
model from vercausa/reasoning.py.

Measures:
  Understanding   is the question mapped to the right intent and details
                  (reading number, failure type, signal, change)? Scored for
                  the rules alone, the LLM alone, and both combined (what the
                  dashboard uses).
  Right event     for questions naming a reading, did retrieval fetch it?
  Right evidence  does the answer cite the kind of fact that answers the
                  question (root cause, simulation result, recommendation...)?
  Faithfulness    share of LLM answers that pass verification (every number
                  and citation found in the retrieved facts); the rest are
                  replaced by an answer built from the facts. Final answers
                  are re-verified.

    python scripts/evaluate_graphrag.py
"""

import collections
import json
import pathlib
import sys
import time
import warnings

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from vercausa.agents import ActionAgents, operating_levels
from vercausa.causal import CausalLayer
from vercausa.explain import typical_values
from vercausa.kg import InMemoryGraphStore
from vercausa.readings import load_ai4i
from vercausa.reasoning import (
    INTENT_PROMPT, INTENT_SCHEMA, GraphRAG, GraphRetriever, OllamaLLM, rule_intent, understand, verify,
)
from vercausa.simulation import SimulationLayer
from vercausa.twin import DigitalTwin

QUESTIONS = ROOT / "data" / "graphrag_questions.json"
MODEL = ROOT / "models" / "causal_layer.joblib"
FIELDS = ("intent", "row", "mechanism", "signal", "change_type", "change_value")

# What a cited fact must say for the answer to address each kind of question.
EVIDENCE = {
    "explain_event": ("Plain-language root cause", "The causal model's attribution"),
    "what_if": ("Simulated risk", "Simulated change"),
    "recommend": ("Recommended action", "The Action Layer's diagnosis: no failure"),
    "current_state": ("Machine MILL-01 status", "Current sensor values", "Failure risk from the causal model"),
    "list_events": ("failure events are recorded", "Reading "),
}


def matches(pred: dict, exp: dict) -> bool:
    for k in FIELDS:
        if k not in exp:
            continue
        want, got = exp[k], pred.get(k)
        if k == "change_value":
            if got is None or abs(float(got) - float(want)) > 1e-6:
                return False
        elif got != want:
            return False
    return True


def llm_only(question: str, llm) -> dict:
    try:
        return json.loads(llm.chat(INTENT_PROMPT, question, schema=INTENT_SCHEMA))
    except Exception:
        return {"intent": "unsupported"}


def main() -> None:
    llm = OllamaLLM()
    if not llm.available():
        sys.exit(f"Ollama with {llm.model} is not running -- start it first (see README).")
    questions = json.loads(QUESTIONS.read_text())

    readings = list(load_ai4i())
    store = InMemoryGraphStore()
    store.apply_schema()
    twin = DigitalTwin(store)
    twin.register("MILL-01")
    for r in readings:
        twin.ingest(r)
    causal = CausalLayer.load(MODEL)
    sim, typical = SimulationLayer(causal), typical_values(readings)
    agents = ActionAgents(causal, sim, typical, operating_levels(pd.DataFrame([r.values for r in readings])))
    rag = GraphRAG(GraphRetriever(store, causal, sim, typical, "MILL-01", agents), llm)

    rows, started = [], time.time()
    for exp in questions:
        q = exp["q"]
        t0 = time.time()
        rules = dict({"intent": "unsupported"}, **rule_intent(q))
        alone = llm_only(q, llm)
        intent = understand(q, llm)
        rec = {"exp": exp, "rules": matches(rules, exp), "llm": matches(alone, exp), "both": matches(intent, exp)}
        if intent["intent"] == "unsupported":
            rec.update(source="help", answered=exp["intent"] == "unsupported")
        else:
            facts = rag.retrieve(intent)
            ans = rag.generate(q, intent, facts)
            cited = [f for f in facts if f"[{f.id}]" in ans.text]
            rec.update(
                source=ans.source,
                issues=ans.issues,
                grounded=verify(ans.text, facts) == [],
                right_event=("row" not in exp or any(f"reading {exp['row']})" in f.text for f in facts)),
                answered=any(any(k in f.text for k in EVIDENCE.get(exp["intent"], ())) for f in cited),
                text=ans.text,
            )
        rec["seconds"] = time.time() - t0
        rows.append(rec)
        print(f"  {'✓' if rec['both'] and rec['answered'] else '✗'} {q}")

    n = len(rows)
    print(f"\n{n} questions in {time.time() - started:.0f}s "
          f"({sum(r['seconds'] for r in rows) / n:.1f}s per question on this machine)\n")
    print("Understanding (intent and every stated detail correct)")
    for key, label in (("rules", "rules alone"), ("llm", "LLM alone"), ("both", "LLM + rules (used)")):
        print(f"  {label:<22} {sum(r[key] for r in rows)}/{n}")
    print("  by intent (LLM + rules):")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["exp"]["intent"]].append(r["both"])
    for k, v in by.items():
        print(f"    {k:<16} {sum(v)}/{len(v)}")

    answered = [r for r in rows if r["source"] != "help"]
    print(f"\nAnswers ({len(answered)} questions sent to retrieval + generation)")
    named = [r for r in answered if "row" in r["exp"]]
    print(f"  right event retrieved:          {sum(r['right_event'] for r in named)}/{len(named)} (questions naming a reading)")
    print(f"  cites the evidence asked for:   {sum(r['answered'] for r in answered)}/{len(answered)}")
    llm_ok = sum(r["source"] == "llm" for r in answered)
    print(f"  LLM wording passed verification: {llm_ok}/{len(answered)}; "
          f"replaced by fact-built answer: {len(answered) - llm_ok}")
    print(f"  final answers fully grounded:   {sum(r['grounded'] for r in answered)}/{len(answered)}")
    unsupported = [r for r in rows if r["exp"]["intent"] == "unsupported"]
    print(f"\nOff-topic questions refused with the help text: {sum(r['source'] == 'help' for r in unsupported)}/{len(unsupported)}")
    wrongly_refused = [r for r in rows if r["source"] == "help" and r["exp"]["intent"] != "unsupported"]
    print(f"On-topic questions wrongly refused: {len(wrongly_refused)}")

    print("\nWhy LLM wording was rejected:")
    reasons = collections.Counter(i.split(" is not")[0].rsplit(" ", 1)[0] if "number" in i else i
                                  for r in answered for i in r.get("issues", []))
    for reason, k in reasons.most_common() or [("none", 0)]:
        print(f"  {k:>3}  {reason}")
    print("\nMisunderstood questions:")
    for r in rows:
        if not r["both"]:
            print(f"  {r['exp']['q']}")
    if all(r["both"] for r in rows):
        print("  none")


if __name__ == "__main__":
    main()
