# Linear Attention Trace Lab

A browser-only visual debugger for a causal kernelized linear-attention recurrence.
It computes the complete trace first, then lets the UI replay token embeddings,
Q/K/V projections, `phi`, pre-write reads, outer-product contributions, and `S`/`Z`
state updates. Values displayed in the workspace come from `Float32Array` inference.

## Run locally

```bash
npm install
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python model/train.py
npm run dev
```

For a production build and unit tests:

```bash
npm run build
npm test
```

## Model truthfulness status

The committed assets are trained on generated, unambiguous simple-English
coreference templates. They are a real two-layer causal linear-attention model, not
a Transformer. Results outside that deliberately narrow domain are unreliable.
Metrics and scope are stored in `public/model/metrics.json`; a regression test
verifies the browser engine predicts `robot` for the canonical example using the
exported trained weights.
