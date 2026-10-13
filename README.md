# Vercausa

Industrial causal intelligence for manufacturing. Vercausa detects machine
failures, then explains **why** they happened, simulates **what if**
conditions had been different, and answers questions in plain language. Every
answer is grounded in a knowledge graph, not in correlations or free-form
text generation.

It is built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
which ships with documented failure mechanisms that serve as ground truth.

| Layer | What it does | Code |
|---|---|---|
| Detection | Checks each sensor reading against the documented failure rules | `vercausa/detection.py` |
| Knowledge graph + digital twin | Machines, sensors, signals and failure events; live machine state lives on the graph | `vercausa/schema.py`, `kg.py`, `kg_neo4j.py`, `twin.py` |
| Causal | Structural causal model, with structure taken from the graph and effects learnt from data | `vercausa/causal.py` |
| Simulation | Counterfactual "what if" changes against the twin's recorded state | `vercausa/simulation.py` |
| Reasoning (GraphRAG) | Answers questions from graph facts, using a local LLM, and checks every number and citation | `vercausa/reasoning.py` |
| Streaming | Live sensor readings over MQTT | `vercausa/streaming.py` |

## Setup

You need Python 3.9, the version the saved model was trained with, on macOS
or Linux. On a Mac, the built-in `/usr/bin/python3` is 3.9.

```bash
git clone https://github.com/Zishan710/Vercausa.git
cd Vercausa
python3 -m venv .venv               # check python3 --version shows 3.9
source .venv/bin/activate
pip install -r requirements.txt
```

The library versions in `requirements.txt` are pinned on purpose: the trained
model in `models/causal_layer.joblib` only loads reliably with the same
versions. The dataset is already in `data/ai4i2020.csv`. To download it again,
run `python scripts/fetch_data.py`.

### Local LLM (for the Ask panel)

```bash
brew install ollama
brew services start ollama          # keeps it running in the background
ollama pull qwen2.5:7b-instruct     # about 4.7 GB, one time
```

Without Ollama, everything still works. The Ask panel then answers directly
from the graph facts, without the LLM's wording.

### Optional

- **Mosquitto** (a full MQTT broker): `brew install mosquitto && brew services start mosquitto`.
  The dashboard uses it automatically if it is running. Otherwise it starts
  its own small built-in broker. Mosquitto is needed for `--qos 1`.
- **Neo4j**: set `NEO4J_URI`, plus `NEO4J_PASSWORD` (and optionally
  `NEO4J_USER`), to store the graph in Neo4j instead of in memory.

## Run the dashboard

**Replay mode** (simplest). The server replays the dataset itself:

```bash
python -m app.server
```

**Live mode.** Readings arrive over MQTT from a simulated machine:

```bash
python -m app.server --source mqtt          # terminal 1
python scripts/publish_machine.py --rate 2  # terminal 2: the "machine"
```

Stop the publisher and the dashboard marks the machine offline after
5 seconds. Both open at http://127.0.0.1:8000.

## Scripts

| Command | Purpose |
|---|---|
| `python scripts/train_model.py` | Retrain the causal model and save it to `models/` |
| `python scripts/validate_detection.py` | Check the detection rules against the dataset's labels |
| `python scripts/evaluate_causal.py` | Root-cause accuracy, using 5-fold cross-validation |
| `python scripts/evaluate_simulation.py` | What-if accuracy against the true failure mechanisms |
| `python scripts/build_twin.py --limit 2000` | Fill the knowledge graph / twin from the dataset |
| `python scripts/whatif_demo.py` | Example what-if questions on real failure events |

## Tests

```bash
python -m pytest -q
```

The test that uses the real LLM is skipped automatically when Ollama is not
running.
