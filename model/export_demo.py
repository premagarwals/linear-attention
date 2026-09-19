"""Create deterministic browser assets for the trace engine.

This bootstrap exporter deliberately keeps the architecture transparent. Replace it
with train.py output once PyTorch is available; the JSON schema remains unchanged.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "model"
VOCAB = ["<unk>", ".", "the", "red", "robot", "picked", "up", "blue", "key", "it", "walked", "toward", "door", "dog", "dropped", "ball", "rolled", "under", "table", "chased", "barked", "loudly", "alice", "handed", "map", "to", "bob", "he", "thanked", "maya", "placed", "book", "on", "desk", "she", "opened", "a", "green", "cat", "and"]
D_MODEL, D_FEATURE = 32, 16

def matrix(rng: random.Random, rows: int, cols: int, scale: float = .08):
    return [[round(rng.gauss(0, scale), 7) for _ in range(cols)] for _ in range(rows)]

def main():
    rng = random.Random(314159)
    OUT.mkdir(parents=True, exist_ok=True)
    layers = [{"wq": matrix(rng, D_MODEL, D_FEATURE), "wk": matrix(rng, D_MODEL, D_FEATURE), "wv": matrix(rng, D_MODEL, D_MODEL), "wo": matrix(rng, D_MODEL, D_MODEL), "bq": [0] * D_FEATURE, "bk": [0] * D_FEATURE, "bv": [0] * D_MODEL} for _ in range(2)]
    weights = {"embeddings": matrix(rng, len(VOCAB), D_MODEL, .12), "layers": layers, "head": matrix(rng, D_MODEL, len(VOCAB), .06)}
    digest = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:12]
    config = {"version": "bootstrap-0.1", "dModel": D_MODEL, "dFeature": D_FEATURE, "maxLength": 48, "epsilon": 1e-6, "layers": 2, "featureMap": "ELU(x) + 1 + eps", "tokenizer": "lowercase-word-punctuation-v1", "weightsChecksum": digest, "training": {"status": "bootstrap initialization; run model/train.py before claiming trained metrics"}}
    (OUT / "model-config.json").write_text(json.dumps(config, indent=2))
    (OUT / "vocab.json").write_text(json.dumps(VOCAB, indent=2))
    (OUT / "weights.json").write_text(json.dumps(weights, separators=(",", ":")))
    (OUT / "metrics.json").write_text(json.dumps({"status": "not trained", "note": "Model assets are valid for exact browser trace mechanics; task predictions are not yet trained."}, indent=2))
    (OUT / "examples.json").write_text(json.dumps(["The red robot picked up the blue key. It walked toward the door.", "The dog dropped the ball. It rolled under the table."], indent=2))

if __name__ == "__main__": main()
