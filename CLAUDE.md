# Vercausa — Project Context

**Course:** Advanced Programming with Project-I, 7th semester, Silver Oak University
**Type:** Group project (team of 3)

## What this is

Vercausa is an Enterprise Industrial Causal Intelligence Platform for manufacturing.
It goes beyond standard predictive maintenance (which predicts *that* a failure is
likely) to explain *why* it's happening, simulate what-if interventions, and
recommend the optimal action — using causal reasoning grounded in a knowledge graph,
not just correlation-based ML.

## Final abstract

> Modern manufacturing environments generate vast amounts of data from sensors,
> machines, and production systems. While conventional AI and predictive analytics
> can identify anomalies and forecast failures, they often fail to explain the
> underlying reasons behind these events. This lack of causal understanding leads to
> increased downtime, inefficient maintenance, and suboptimal operational decisions.
>
> Vercausa is an Enterprise Industrial Causal Intelligence Platform designed to move
> beyond prediction by uncovering the true cause-and-effect relationships within
> complex manufacturing systems. The platform integrates Industrial IoT, Knowledge
> Graphs, Structural Causal Models (SCMs), GraphRAG, a lightweight Digital Twin, and
> AI Agents to transform raw industrial data into actionable and explainable
> intelligence. Industrial IoT streams are structured into a Knowledge Graph that
> also serves as a live, continuously updated Digital Twin of current system state;
> Structural Causal Models operate over this graph to identify root causes and
> perform counterfactual "what-if" reasoning, while GraphRAG grounds natural-language
> explanations in the same graph and AI Agents translate the results into ranked,
> actionable recommendations before failures impact production.
>
> Unlike traditional predictive maintenance systems that answer "What is likely to
> happen?", Vercausa answers "Why did it happen?", "What would happen if conditions
> changed?", and "What is the optimal action to take?". This explainable approach
> enhances decision-making, reduces unplanned downtime, optimizes maintenance
> strategies, improves Overall Equipment Effectiveness (OEE), and supports the
> transition toward Industry 5.0.
>
> Vercausa aims to redefine industrial intelligence by empowering organizations with
> transparent, trustworthy, and causality-driven AI, enabling smarter operations,
> higher productivity, and sustainable manufacturing.

**Keywords:** Industrial Causal AI, Industry 5.0, Root Cause Analysis, Explainable AI
(XAI), Industrial IoT, Knowledge Graphs, Structural Causal Models, GraphRAG, Digital
Twin, Predictive Maintenance, Manufacturing Intelligence.

## Architecture decisions (locked in — don't relitigate these without good reason)

### 1. SCM is expert-seeded, not fully auto-discovered
Causal discovery from scratch (PC, NOTEARS, etc.) breaks on industrial data —
feedback control loops violate the acyclicity assumption, and hidden confounders
violate causal sufficiency. Instead:
- The Knowledge Graph's topology (which sensor is upstream of which, physical/process
  connectivity, temporal ordering) defines the **candidate edge set**.
- An edge is only eligible for causal estimation if the KG says the two nodes are
  connected and respects topological order (upstream can cause downstream, never the
  reverse).
- DoWhy / EconML / causal-learn then estimate **effect strength and direction only
  over KG-permitted edges** — they don't search the full unconstrained DAG space.
- This mirrors Wehner et al. 2024 (arXiv:2402.00043), which does the same
  KG-constrained approach for exactly this reason.

### 2. Digital Twin = lightweight state-mirror, NOT a simulator
- The twin is just **live-updated state on the Knowledge Graph nodes** (current
  sensor readings, machine status, timestamps), refreshed off the IoT stream.
- It does NOT run physics simulations and does NOT itself perform "what-if"
  reasoning.
- Counterfactual / what-if simulation is the SCM's job — it runs `do()` interventions
  against the twin's current state, not the other way around.
- This split must stay consistent in all docs/code comments: twin = context,
  SCM = simulation.

### 3. Detection Layer — closes the "what triggers this" gap
Nothing in IoT → KG → SCM → GraphRAG → Twin → Agents actually decides *when* to run
root-cause analysis. Fix: a thin **Detection Layer** sits between ingestion and the
Knowledge Layer, evaluating incoming sensor readings against known failure-condition
rules (see dataset section below — these thresholds come directly from AI4I 2020's
documented failure mechanisms, not a trained model). A rule tripping is what fires
the KG lookup + SCM root-cause step. Deliberately not a black-box anomaly detector —
using an uninterpretable model here would undermine the "explainable" pitch before
the causal layer even runs.

### 4. Pipeline (six layers, in order)
1. **Data / Detection Layer** — ingest real-time sensor data (Industrial IoT), and
   evaluate it against documented failure-condition rules to trigger downstream
   analysis
2. **Knowledge Layer** — Knowledge Graph of machines, processes, sensors, fault
   events; doubles as the live digital twin (state lives on the graph nodes)
3. **Causal Layer** — Structural Causal Models, expert-seeded via KG topology,
   estimate root-cause effects over the permitted edge set
4. **Simulation Layer** — the causal model runs counterfactual `do()` interventions
   against current twin state to evaluate "what if we changed X"
5. **Reasoning Layer** — GraphRAG answers natural-language causal queries ("why did
   line 3 stop"), grounded in the KG + causal model, not free LLM generation
6. **Action Layer** — AI Agents rank causal findings + simulation outcomes into
   explainable, actionable maintenance recommendations

## Dataset

**Primary: AI4I 2020 Predictive Maintenance Dataset**
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
(also: `ucimlrepo` → `fetch_ucirepo(id=601)`)

Why this one specifically: it ships with **documented, deterministic failure
mechanisms**, which is what makes expert-seeding and evaluation possible at all —
most PM datasets don't give you ground truth to check a causal model against.

- **TWF** (Tool Wear Failure): tool replaced/fails in a defined wear-time window
  (200–240 min)
- **HDF** (Heat Dissipation Failure): (process temp − air temp) < 8.6K AND
  rotational speed < 1380 rpm
- **PWF** (Power Failure): torque × rotational speed (power) outside [3500W, 9000W]
- **OSF** (Overstrain Failure): tool wear × torque exceeds a threshold that varies by
  product variant (11,000 / 12,000 / 13,000 minNm for L / M / H)
- **RNF** (Random Failure): 0.1% chance, independent of process state

These five rules are simultaneously: (a) the Detection Layer's trigger logic, (b) the
KG's expert-seeded causal structure, and (c) the ground truth to evaluate whether the
SCM attributes failures to the correct mechanism. Use them for all three.

**For the live-pipeline demo:** the CSV is static, but the abstract promises live IoT
ingestion. Don't demo a spreadsheet — replay the CSV as a simulated stream (a script
publishing rows to MQTT/Kafka on a timer, as if arriving in real time). Static CSV →
SCM training/validation (where ground truth matters). Replayed stream → live
ingestion → KG update → twin refresh → GraphRAG query demo (where "it works
end-to-end" matters). Don't conflate the two roles.

**Secondary / stretch, if time allows:** Tennessee Eastman Process (52 variables, 21
documented faults, real P&ID topology) — heavier engineering lift, use only as a
generalization test after the AI4I2020 pipeline is working, not as the primary
deliverable.

**Rejected and why:** SECOM (no public process topology, can't validate an
expert-seeded graph against it), Microsoft Azure PdM / Scania telemetry (real-looking
but no ground-truth causal mechanism, same problem as SECOM). Don't hand-generate a
synthetic dataset either — if you write the failure rule and then "discover" the same
rule with the SCM, that's circular, not causal discovery.

## Suggested tech stack

- **Industrial IoT:** MQTT / OPC-UA ingestion, Kafka streaming
- **Knowledge Graph:** Neo4j or RDF store
- **Causal Inference:** DoWhy, EconML, or causal-learn
- **GraphRAG:** LLM-based retrieval-augmented generation over the knowledge graph
- **AI Agents:** LangChain / LangGraph
- **Backend / Deployment:** Python, FastAPI, Docker

## Review 1 deliverable

A Review 1 slide deck was already built in Silver Oak University's official PPTX
template — 10 fixed slides (Title, Index, Introduction, Background & Motivation,
Relevance & Importance, Literature Review, Objectives, Tools & Technology,
Method/Approach, Bibliography), fixed 24pt body font with `<a:noAutofit/>` so nothing
auto-shrinks. Literature review cites 6 papers (causal AI vs. correlation ML for
predictive maintenance, KG-based root cause analysis, neuro-symbolic explainable
digital twins, GraphRAG) — see Bibliography slide for full citations. Course
name/code and the enrollment table on the title slide are intentionally left blank —
fill in separately.

## Still open / decide before or early in implementation

- ~~Exact KG schema~~ — decided, see `vercausa/schema.py`. Type level (Signal,
  FailureMode) holds the causal structure; instance level (Machine, Sensor,
  FailureEvent) holds twin state and history. CANDIDATE_CAUSE_OF runs
  Signal | DerivedQuantity → FailureMode. DerivedQuantity nodes (temperature
  difference, power) hold textbook-physics formulas only — nothing
  AI4I-specific (e.g. not wear × torque); thresholds are always learnt.
  Without them the SCM learns correlational shortcuts and gives wrong
  what-if answers (measured in Phase 4).
- Team role split for the three-person group (not yet discussed with Claude)
- ~~Whether GraphRAG runs against a local LLM or an API-based one~~ — decided:
  local, Ollama `qwen2.5:7b-instruct` (no API cost; fine for college demos).
  `vercausa/reasoning.py` verifies every number/citation against graph facts
  and falls back to fact-built answers, so the LLM never adds unsupported claims.
- Evaluation metrics beyond root-cause attribution accuracy (e.g., false-alarm rate,
  simulated OEE improvement) — the causal-vs-correlation benchmark paper cited in the
  literature review (arXiv:2512.01149) is a good template for how to structure this
