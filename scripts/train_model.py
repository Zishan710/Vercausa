"""Train the Causal Layer on all of AI4I 2020 and save it.

Writes:
  models/causal_layer.joblib      the fitted model, loadable with CausalLayer.load()
  models/training_report.json     what was trained, on what data, with which settings

Accuracy is measured separately, on held-out data, by scripts/evaluate_causal.py
and scripts/evaluate_simulation.py -- scoring the model on the rows it was
trained on would overstate it.

    python scripts/train_model.py
"""

import datetime
import hashlib
import json
import pathlib
import platform
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import dowhy
import sklearn

from vercausa.causal import CausalLayer, training_frame
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    data = training_frame(DEFAULT_CSV)
    store = open_store()
    store.apply_schema()

    print(f"Training on {DEFAULT_CSV.name}: {len(data)} rows, "
          f"{int(data['machine_failure'].sum())} machine failures")
    start = time.perf_counter()
    layer = CausalLayer.fit_from_store(store, data)
    seconds = time.perf_counter() - start

    model_path = MODELS / "causal_layer.joblib"
    layer.save(model_path)

    mechanisms = []
    for node in layer.dag.nodes:
        parents = layer.parents(node)
        kind = layer.dag.nodes[node]["kind"]
        if not parents:
            continue
        entry = {"node": node, "kind": kind, "inputs": parents}
        if kind == "failure_mode":
            entry["model"] = "HistGradientBoostingClassifier (DoWhy-GCM ClassifierFCM)"
            entry["min_samples_leaf"] = layer.min_samples_leaf[node]
            entry["training_positives"] = int(data[node].sum())
            entry["explained_in_terms_of"] = layer.explanatory_signals(node)
        elif kind == "derived":
            entry["model"] = "physics formula (not learnt)"
        else:
            entry["model"] = "linear additive-noise model (DoWhy-GCM)"
        mechanisms.append(entry)

    report = {
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "training_seconds": round(seconds, 1),
        "data_file": str(DEFAULT_CSV.relative_to(DEFAULT_CSV.parent.parent)),
        "data_sha256": hashlib.sha256(DEFAULT_CSV.read_bytes()).hexdigest(),
        "rows": len(data),
        "machine_failures": int(data["machine_failure"].sum()),
        "structure_source": "Knowledge Graph candidate edges (vercausa/schema.py)",
        "mechanisms": mechanisms,
        "edge_strengths": {f"{a} -> {b}": round(v, 3) for (a, b), v in layer.edge_strengths(data).items()},
        "validation": "5-fold cross-validation on held-out rows: scripts/evaluate_causal.py, "
                      "scripts/evaluate_simulation.py",
        "versions": {"python": platform.python_version(), "dowhy": dowhy.__version__,
                     "scikit-learn": sklearn.__version__},
        "model_file": str(model_path.relative_to(MODELS.parent)),
    }
    (MODELS / "training_report.json").write_text(json.dumps(report, indent=2))

    print(f"Trained in {seconds:.1f}s\n")
    print("Learnt mechanisms (each sees only the causes the Knowledge Graph allows):")
    for m in mechanisms:
        extra = f"  leaf={m['min_samples_leaf']}, {m['training_positives']} failures" if "min_samples_leaf" in m else ""
        print(f"  {m['node']:<20} <- {', '.join(m['inputs']):<45} {m['model'].split(' (')[0]}{extra}")
    print(f"\nSaved model:  {model_path.relative_to(MODELS.parent)} "
          f"({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"Saved report: models/training_report.json")
    store.close()


if __name__ == "__main__":
    main()
