"""Reasoning Layer: GraphRAG over the knowledge graph and the causal model.

A question goes through four steps:

1. Understand -- a local LLM maps the question to an intent (explain an
   event, current state, list events, what-if) and its entities. Explicit
   details in the text (reading numbers, percentages, target values) are
   then read by deterministic rules, which override the LLM where they apply.
2. Retrieve -- facts come only from the knowledge graph (failure events,
   their rule, sensor snapshot, permitted cause paths, live twin state) and
   from the causal and simulation layers run on that graph state. Each fact
   is numbered [F1], [F2], ...
3. Generate -- the LLM answers using only those facts and cites them.
4. Verify -- every number in the answer must appear in the facts and every
   citation must exist. Otherwise the LLM text is discarded and the answer
   is built directly from the facts, so an unsupported claim never reaches
   the operator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

from vercausa.causal import CausalLayer
from vercausa.explain import (
    MODE_NAMES, OBSERVABLE, cause_path, derived_values, explain_failure, model_row, whatif,
)
from vercausa.kg import GraphStore
from vercausa.simulation import SimulationLayer

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
INTENTS = ("explain_event", "current_state", "list_events", "what_if", "unsupported")
SIGNALS = ("air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear")
LABEL = {"air_temperature": "air temperature", "process_temperature": "process temperature",
         "rotational_speed": "spindle speed", "torque": "torque", "tool_wear": "tool wear",
         "power": "power", "temperature_difference": "temperature difference",
         "product_variant": "product variant"}
UNIT = {"air_temperature": "K", "process_temperature": "K", "rotational_speed": "rpm", "torque": "Nm",
        "tool_wear": "min", "power": "W", "temperature_difference": "K"}
HELP = ("I can answer questions about this machine's failures (for example \"Why did it fail at reading 51?\"), "
        "its current state, the list of detected failures, and what-if changes such as "
        "\"What if torque was reduced by 20% in the last power failure?\"")


# ---------------------------------------------------------------- LLM

class OllamaLLM:
    """A locally running model served by Ollama. Deterministic (temperature 0)."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_OLLAMA, timeout: float = 120.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def available(self) -> bool:
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=3).json()
        except (httpx.HTTPError, ValueError):
            return False
        return any(m.get("name") == self.model for m in tags.get("models", []))

    def chat(self, system: str, user: str, schema: dict | None = None) -> str:
        body = {"model": self.model, "stream": False,
                "options": {"temperature": 0, "seed": 7},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema is not None:
            body["format"] = schema
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


# ---------------------------------------------------------------- 1. understand

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "row": {"type": ["integer", "null"]},
        "mechanism": {"type": ["string", "null"], "enum": list(OBSERVABLE) + [None]},
        "signal": {"type": ["string", "null"], "enum": list(SIGNALS) + [None]},
        "change_type": {"type": ["string", "null"], "enum": ["percent", "absolute", None]},
        "change_value": {"type": ["number", "null"]},
    },
    "required": ["intent", "row", "mechanism", "signal", "change_type", "change_value"],
}

INTENT_PROMPT = """You convert a maintenance operator's question about a milling machine into JSON.
intent:
- explain_event: why a failure happened, what caused it
- current_state: how the machine is doing right now
- list_events: which failures happened, how many
- what_if: what would happen if a signal were changed
- unsupported: anything else
row: the reading number if one is mentioned, else null.
mechanism: TWF (tool wear failure), HDF (heat dissipation failure), PWF (power failure),
OSF (overstrain failure), or null.
signal (what_if only): air_temperature, process_temperature, rotational_speed (speed, rpm),
torque, tool_wear (tool, wear), or null.
change_type/change_value (what_if only): "percent" with a signed number (reduce 20% -> -20),
or "absolute" with the new value (to 1500 rpm -> 1500). Replacing the tool means
signal tool_wear, absolute 0."""

# A failure type counts only when it is named AS the failure ("overstrain failure",
# "PWF"), not when its keyword merely appears ("... when power was high").
_MECH_PATTERNS = {
    "PWF": r"\bpower failure|\bpwf\b",
    "HDF": r"\bheat(?: dissipation)? failure|\boverheat\w*|\bhdf\b",
    "OSF": r"\boverstrain\w*|\boverload\w*|\bosf\b",
    "TWF": r"\btool[- ]wear failure|\btwf\b",
}
# Whole words only: "cutting force" must not read as "cut".
_DECREASE = r"\b(?:reduc\w*|lower(?:ed|ing)?|decreas\w*|cut|cuts|drop(?:ped|s)?|slow(?:er|ed|ing)?|less)\b"
_SIGNAL_WORDS = [("process_temperature", ("process temp",)), ("air_temperature", ("air temp", "ambient")),
                 ("rotational_speed", ("speed", "rpm")), ("torque", ("torque", "cutting force")),
                 ("tool_wear", ("tool wear", "wear", "tool"))]


def rule_intent(question: str) -> dict:
    """Deterministic reading of explicit details in the question."""
    q = question.lower()
    out: dict = {}
    m = re.search(r"\b(?:reading|row|record)\s*#?\s*(\d+)", q)
    if m:
        out["row"] = int(m.group(1))
    named = [(m.start(), code) for code, pat in _MECH_PATTERNS.items() for m in [re.search(pat, q)] if m]
    if named:
        out["mechanism"] = min(named)[1]
    if re.search(r"\bwhat if\b|\bwould happen\b|\bif we\b|\bsuppose\b", q):
        out["intent"] = "what_if"
        if re.search(r"replac\w* the tool|new tool|change the tool", q):
            out.update(signal="tool_wear", change_type="absolute", change_value=0.0)
        else:
            for sig, words in _SIGNAL_WORDS:
                if any(w in q for w in words):
                    out["signal"] = sig
                    break
            pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", q)
            to = re.search(r"\bto\s+(\d+(?:\.\d+)?)", q)
            if pct:
                sign = -1 if re.search(_DECREASE, q) else 1
                out.update(change_type="percent", change_value=sign * float(pct.group(1)))
            elif to:
                out.update(change_type="absolute", change_value=float(to.group(1)))
    elif re.search(r"\bwhy\b|\bcause|\breason|\bexplain", q):
        out["intent"] = "explain_event"
    elif re.search(r"\b(current|right now|status|state|how is|doing)\b", q):
        out["intent"] = "current_state"
    elif re.search(r"\b(list|which|how many|all)\b.*\bfail", q) or re.search(r"\bfailures\b", q):
        out["intent"] = "list_events"
    return out


def understand(question: str, llm) -> dict:
    parsed = {k: None for k in INTENT_SCHEMA["properties"]}
    parsed["intent"] = "unsupported"
    if llm is not None:
        try:
            raw = json.loads(llm.chat(INTENT_PROMPT, question, schema=INTENT_SCHEMA))
            parsed.update({k: raw.get(k) for k in parsed if k in raw})
        except (httpx.HTTPError, ValueError, KeyError):
            pass
    rules = rule_intent(question)
    if rules.get("intent") and (parsed["intent"] == "unsupported" or llm is None):
        parsed["intent"] = rules["intent"]
    for k in ("row", "mechanism", "signal", "change_type", "change_value"):
        if rules.get(k) is not None:
            parsed[k] = rules[k]
    if parsed["intent"] not in INTENTS:
        parsed["intent"] = "unsupported"
    if parsed["mechanism"] not in OBSERVABLE:
        parsed["mechanism"] = None
    if parsed["signal"] not in SIGNALS:
        parsed["signal"] = None
    if parsed["change_type"] not in ("percent", "absolute"):
        parsed["change_type"] = None
    parsed["row"] = _as_number(parsed["row"], int)
    parsed["change_value"] = _as_number(parsed["change_value"], float)
    return parsed


def _as_number(value, kind):
    """The LLM's JSON may carry numbers as strings; anything unusable becomes None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return kind(float(value))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 2. retrieve

@dataclass
class Fact:
    id: str
    text: str
    source: str


class _Facts:
    def __init__(self):
        self.items: list = []

    def add(self, text: str, source: str) -> str:
        fid = f"F{len(self.items) + 1}"
        self.items.append(Fact(fid, text, source))
        return fid


def _fmt(v: float, digits: int = 1) -> str:
    return f"{v:,.{digits}f}"


def _pct(p: float) -> str:
    return "100%" if p >= 0.995 else "0%" if p < 0.001 else f"{p * 100:.1f}%" if p < 0.01 else f"{round(p * 100)}%"


def _snapshot_text(values: dict) -> str:
    d = derived_values(values)
    parts = [f"{LABEL[s]} {_fmt(values[s], 0 if s in ('rotational_speed', 'tool_wear') else 1)} {UNIT[s]}" for s in SIGNALS]
    return (", ".join(parts) + f", product variant {values['product_variant']}; power {_fmt(d['power'], 0)} W, "
            f"temperature difference {_fmt(d['temperature_difference'], 2)} K")


class GraphRetriever:
    def __init__(self, store: GraphStore, causal: CausalLayer, sim: SimulationLayer, typical: dict, machine_id: str):
        self.store, self.causal, self.sim, self.typical, self.machine_id = store, causal, sim, typical, machine_id

    def find_event(self, row, mechanism):
        events = self.store.failure_events(self.machine_id, mechanism=mechanism)
        if row is not None:
            events = [e for e in events if e["source_row"] == row]
        return events[0] if events else None

    def _event_facts(self, facts: _Facts, event: dict) -> dict:
        values = dict(event["snapshot"])
        mode = event["mechanism"]
        facts.add(f"Failure event {event['event_id']} was recorded on machine {self.machine_id} at "
                  f"{event['timestamp'][:16].replace('T', ' ')} (reading {event['source_row']}). "
                  f"It is an instance of the failure mode {MODE_NAMES[mode]} ({mode}).",
                  f"(:FailureEvent {{event_id: '{event['event_id']}'}})-[:INSTANCE_OF]->(:FailureMode {{code: '{mode}'}})")
        facts.add(f"The detection rule that fired: {event['reasons']}.", "FailureEvent.reasons")
        facts.add(f"Sensor snapshot at that moment: {_snapshot_text(values)}.", "FailureEvent.snapshot")
        path = cause_path(self.causal, mode)
        chains = [f"{LABEL[p['signal']]} → {LABEL[p['via']]}" if p["via"] else LABEL[p["signal"]] for p in path]
        facts.add(f"In the knowledge graph, {MODE_NAMES[mode]} can be caused by: {', '.join(chains)}.",
                  f"(:Signal)-[:FEEDS*0..1]->()-[:CANDIDATE_CAUSE_OF]->(:FailureMode {{code: '{mode}'}})")
        exp = explain_failure(self.causal, mode, values, self.typical)
        shares = ", ".join(f"{LABEL.get(c['signal'], c['signal'].replace('_', ' '))} {round(c['share'] * 100)}%"
                           for c in exp["causes"])
        facts.add(f"The causal model's attribution for this failure: {shares}.", "Causal Layer (Shapley attribution)")
        facts.add(f"Plain-language root cause: {exp['summary']}", "Causal Layer + Knowledge Graph")
        return values

    def explain_event(self, intent: dict) -> _Facts:
        facts = _Facts()
        event = self.find_event(intent.get("row"), intent.get("mechanism"))
        if event is None:
            facts.add("No matching failure event is recorded in the knowledge graph yet.", "(:FailureEvent) query")
            return facts
        self._event_facts(facts, event)
        return facts

    def current_state(self, intent: dict) -> _Facts:
        facts = _Facts()
        st = self.store.machine_state(self.machine_id)
        if st["last_updated"] is None:
            facts.add(f"Machine {self.machine_id} has not reported any readings yet.", "(:Machine)")
            return facts
        alerts = ", ".join(MODE_NAMES[a] for a in st["active_alerts"]) or "none"
        facts.add(f"Machine {self.machine_id} status is {st['status']} as of {st['last_updated'][:16].replace('T', ' ')}; "
                  f"active alerts: {alerts}.", f"(:Machine {{machine_id: '{self.machine_id}'}})")
        values = {s: st["sensors"][s]["value"] for s in SIGNALS}
        values["product_variant"] = st["product_variant"]
        facts.add(f"Current sensor values: {_snapshot_text(values)}.", "(:Sensor)-[:MEASURES]->(:Machine)")
        frame = model_row(values).to_frame().T.astype(float)
        risks = ", ".join(f"{MODE_NAMES[m]} {_pct(self.causal.failure_probability(m, frame)[0])}" for m in OBSERVABLE)
        facts.add(f"Failure risk from the causal model right now: {risks}.", "Causal Layer on live twin state")
        return facts

    def list_events(self, intent: dict) -> _Facts:
        facts = _Facts()
        events = self.store.failure_events(self.machine_id, mechanism=intent.get("mechanism"))
        if not events:
            facts.add("No failure events are recorded in the knowledge graph yet.", "(:FailureEvent) query")
            return facts
        counts = {}
        for e in events:
            counts[e["mechanism"]] = counts.get(e["mechanism"], 0) + 1
        summary = ", ".join(f"{MODE_NAMES[m]}: {n}" for m, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        facts.add(f"{len(events)} failure events are recorded for {self.machine_id} ({summary}).",
                  "(:FailureEvent)-[:OBSERVED_ON]->(:Machine)")
        for e in events[:5]:
            facts.add(f"Reading {e['source_row']} at {e['timestamp'][:16].replace('T', ' ')}: "
                      f"{MODE_NAMES[e['mechanism']]} ({e['reasons']}).", f"FailureEvent {e['event_id']}")
        return facts

    def what_if(self, intent: dict) -> _Facts:
        facts = _Facts()
        event = self.find_event(intent.get("row"), intent.get("mechanism"))
        if event is None:
            facts.add("No matching failure event is recorded in the knowledge graph yet.", "(:FailureEvent) query")
            return facts
        signal, kind, amount = intent.get("signal"), intent.get("change_type"), intent.get("change_value")
        values = self._event_facts(facts, event)
        if signal not in SIGNALS or kind not in ("percent", "absolute") or amount is None:
            facts.add("The question does not say which signal to change and by how much.", "question")
            return facts
        before = values[signal]
        after = before * (1 + amount / 100) if kind == "percent" else float(amount)
        r = whatif(self.sim, values, {signal: after}, event["mechanism"])
        mode = event["mechanism"]
        digits = 0 if signal in ("rotational_speed", "tool_wear") else 1
        change = f" ({amount:+g}%)" if kind == "percent" else ""
        derived = [f"{LABEL[d]} {_fmt(r['derived_before'][d], 0 if d == 'power' else 2)} → "
                   f"{_fmt(r['derived_after'][d], 0 if d == 'power' else 2)} {UNIT[d]}"
                   for d in ("power", "temperature_difference")
                   if abs(r["derived_after"][d] - r["derived_before"][d]) > 1e-6]
        facts.add(f"Simulated change: {LABEL[signal]} {_fmt(before, digits)} → {_fmt(after, digits)} {UNIT[signal]}{change}"
                  + (f"; {', '.join(derived)}" if derived else "") + ".", "Simulation Layer (counterfactual do-intervention)")
        facts.add(f"Simulated risk of this {MODE_NAMES[mode]}: {_pct(r['risk_before'][mode])} before, "
                  f"{_pct(r['risk_after'][mode])} after.", "Simulation Layer")
        still = ", ".join(MODE_NAMES[m] for m in r["still_failing"]) or "none"
        facts.add(f"Simulated risk of any failure: {_pct(r['any_before'])} before, {_pct(r['any_after'])} after; "
                  f"failures still likely after the change: {still}.", "Simulation Layer")
        if not r["trustworthy"]:
            facts.add("Trust flag: this change is outside the data the model learnt from, so the result is low-confidence.",
                      "Simulation Layer (data-support check)")
        return facts

    def retrieve(self, intent: dict) -> _Facts:
        return {"explain_event": self.explain_event, "current_state": self.current_state,
                "list_events": self.list_events, "what_if": self.what_if}[intent["intent"]](intent)


# ---------------------------------------------------------------- 3. generate + 4. verify

ANSWER_PROMPT = """You are Vercausa, a maintenance assistant for a milling machine.
Answer the operator's question using ONLY the numbered facts given.
Rules:
- After every sentence, cite the facts it uses, like [F2] or [F3][F5].
- Do not add numbers, causes, actions or advice that are not stated in the facts.
- Copy numbers exactly as they appear in the facts.
- Do not judge numbers with words like high, low, significant or risky unless a fact says so;
  just state the number.
- If the facts do not answer the question, say so in one sentence.
- Use plain language and at most 4 short sentences."""

# Magnitudes only: a dash is usually punctuation ("2026-01-01", "3,500–9,000"), not a sign.
_NUM = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list:
    out = []
    for tok in _NUM.findall(re.sub(r"\[F\d+\]", " ", text)):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def verify(answer: str, facts: list) -> list:
    """Problems with an answer; an empty list means every claim is grounded."""
    issues = []
    ids = {f.id for f in facts}
    cited = set(re.findall(r"\[(F\d+)\]", answer))
    if not cited:
        issues.append("the answer cites no facts")
    unknown = cited - ids
    if unknown:
        issues.append(f"cites facts that do not exist: {', '.join(sorted(unknown))}")
    known = _numbers(" ".join(f.text for f in facts))
    for n in _numbers(answer):
        if not any(abs(n - k) <= max(0.051, 0.005 * abs(k)) for k in known):
            issues.append(f"the number {n:g} is not in the facts")
    return issues


def template_answer(intent: str, facts: list) -> str:
    """An answer assembled directly from facts, used when the LLM's answer fails verification."""
    by_id = {f.id: f for f in facts}
    if len(facts) == 1:
        return f"{facts[0].text} [{facts[0].id}]"
    if intent == "explain_event":
        plain = next(f for f in facts if f.text.startswith("Plain-language root cause:"))
        return (f"{facts[0].text} [{facts[0].id}] {plain.text.split(': ', 1)[1]} [{plain.id}] "
                f"{facts[4].text} [{facts[4].id}]") if "F5" in by_id else f"{plain.text} [{plain.id}]"
    if intent == "what_if":
        sim = [f for f in facts if f.source.startswith("Simulation Layer")]
        return " ".join(f"{f.text} [{f.id}]" for f in (sim or facts[-1:]))
    return " ".join(f"{f.text} [{f.id}]" for f in facts[:3])


@dataclass
class Answer:
    question: str
    intent: dict
    text: str
    facts: list
    source: str
    issues: list = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return self.source in ("llm", "template")

    def to_dict(self) -> dict:
        return {"question": self.question, "intent": self.intent, "text": self.text, "source": self.source,
                "issues": self.issues, "facts": [f.__dict__ for f in self.facts]}


class GraphRAG:
    def __init__(self, retriever: GraphRetriever, llm=None):
        self.retriever, self.llm = retriever, llm

    def understand(self, question: str) -> dict:
        return understand(question, self.llm)

    def retrieve(self, intent: dict) -> list:
        return self.retriever.retrieve(intent).items

    def generate(self, question: str, intent: dict, facts: list) -> Answer:
        if self.llm is None:
            return Answer(question, intent, template_answer(intent["intent"], facts), facts, "template",
                          ["no language model available"])
        user = f"Question: {question}\n\nFacts:\n" + "\n".join(f"[{f.id}] {f.text}" for f in facts)
        try:
            text = self.llm.chat(ANSWER_PROMPT, user).strip()
        except (httpx.HTTPError, KeyError, ValueError) as e:
            return Answer(question, intent, template_answer(intent["intent"], facts), facts, "template",
                          [f"language model error: {e}"])
        issues = verify(text, facts)
        if issues:
            return Answer(question, intent, template_answer(intent["intent"], facts), facts, "template", issues)
        return Answer(question, intent, text, facts, "llm")

    def answer(self, question: str) -> Answer:
        intent = self.understand(question)
        if intent["intent"] == "unsupported":
            return Answer(question, intent, HELP, [], "help")
        return self.generate(question, intent, self.retrieve(intent))
