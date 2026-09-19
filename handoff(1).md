# Linear Attention Visualizer — Full Implementation Handoff

> **Purpose of this file:** This is the implementation contract for Codex. Work through the checkboxes until the repository contains a complete, locally runnable, production-quality educational website that visualizes a **real tiny trained causal linear-attention model** frame by frame.
>
> The final site must prioritize **visual explanation over text**. Users should be able to type a short/simple English sentence, run it through the model, and inspect how every token becomes an embedding, how Q/K/V are derived, how the linear-attention state changes, how a later query reads the state, and how the model arrives at a simple NLP prediction such as a pronoun referring to a prior entity.

---

# 0. Non-negotiable outcome

- [ ] Deliver a **fully working website**, not a mockup.
- [ ] Work inside the existing git repository.
- [ ] You may install/download any required open-source packages or model/training dependencies.
- [ ] Do not stop at scaffolding. Run the application, fix errors, run tests, build production assets, and leave the repository in a working state.
- [ ] The final website must run locally with a small number of commands documented in `README.md`.
- [ ] Prefer a static/client-side deployment. No backend should be required for normal use.
- [ ] All model inference required for the visualization must run **inside the browser**.
- [ ] Training may happen offline in Python during development, but final trained weights must be committed/exported so end users do not need Python.
- [ ] The browser must use the **actual trained model weights** for token embeddings and all displayed neural-network operations.
- [ ] The main educational sequence must use **real intermediate tensor values**, not invented numbers.
- [ ] Do not use a normal Transformer internally and label it “linear attention.”
- [ ] Do not use a large opaque pretrained model to secretly perform the NLP task while a fake linear-attention animation runs beside it.
- [ ] The core demonstration must use a real causal kernelized linear-attention layer whose state is updated token by token.
- [ ] The model may intentionally support only **short, simple English**. Clearly communicate this limitation in the UI without cluttering the experience.
- [ ] If free-form input is outside the model’s vocabulary/domain, the UI must show uncertainty/OOV information rather than pretending the result is reliable.
- [ ] The UI must contain substantially more **visual motion, matrices, vectors, paths, highlights, and transitions** than prose.
- [ ] Detailed explanations should appear on hover, click, focus, or an inspector panel rather than as permanent paragraphs.
- [ ] Every important value should have provenance: users should be able to answer **“where did this number come from?”**

---

# 1. Product goal

- [ ] Build an interactive “neural network debugger” that teaches linear attention by exposing the complete data flow.
- [ ] Primary mental model: tensors behave like physical data packets that are created, transformed, moved, merged into state, and later read by a query.
- [ ] Primary learning objective: after using the site, a user should understand:
  - [ ] what tokenization does;
  - [ ] what a learned embedding is;
  - [ ] what Q, K, and V represent computationally;
  - [ ] how Q/K/V are actually calculated from learned matrices;
  - [ ] what the feature map `phi()` does;
  - [ ] what the running linear-attention states `S` and `Z` are;
  - [ ] how each token changes `S` and `Z`;
  - [ ] how a later query reads the accumulated state;
  - [ ] why the state can be updated incrementally;
  - [ ] how per-token influence can be reconstructed for explanation;
  - [ ] why sequence-length scaling differs from standard full pairwise attention.
- [ ] Secondary learning objective: demonstrate a tiny NLP task such as pronoun/coreference prediction, e.g.:
  - [ ] `The red robot picked up the blue key. It walked toward the door.`
  - [ ] model prediction should ideally associate `It` with `robot` for a supported/trained example.

---

# 2. Truthfulness / “real vs simulated” contract

Create a small legend in the UI with these categories and use it consistently.

- [ ] **Learned** — parameter came from training and was loaded from model weights.
- [ ] **Computed** — value was calculated by the browser during the current inference trace.
- [ ] **Reconstructed** — mathematically derived after inference for explanation, but equivalent to a decomposition of the real computation.
- [ ] **Presentation only** — animation timing, truncation, layout, or visual metaphor; not part of model computation.

The following must be **real**:

- [ ] token IDs;
- [ ] embedding lookup values;
- [ ] positional embeddings if used;
- [ ] learned `Wq`, `Wk`, `Wv`, output, normalization, FFN, and task-head weights;
- [ ] Q/K/V vectors;
- [ ] feature-map outputs;
- [ ] outer products used to update state;
- [ ] `S` state values;
- [ ] `Z` state values;
- [ ] numerator and denominator of the linear-attention read;
- [ ] contextual output vectors;
- [ ] final prediction scores;
- [ ] all displayed scalar arithmetic when a user drills down.

The following may be **presentation-only or reconstructed**:

- [ ] slowing inference down for replay;
- [ ] showing only 4–8 dimensions of a 16/32/64-D vector by default;
- [ ] “ghost” overlays showing old vs new state;
- [ ] contribution-layer stacks separating terms that are summed in real inference;
- [ ] reconstructed per-token kernel similarity/influence bars;
- [ ] arrows and moving tensor cards;
- [ ] semantic labels such as “current token,” “query,” “memory read,” etc.

Never claim:

- [ ] that one arbitrary hidden-state dimension literally means `robot`, `red`, `object`, etc.;
- [ ] that a high attention/influence score mathematically proves linguistic coreference;
- [ ] that the tiny model is a general-purpose English language model;
- [ ] that the visualizer is showing a large modern LLM’s internal chain of thought.

Preferred wording in the UI:

- [ ] “Model prediction: `It` → `robot`” when the task head actually predicts that token/entity.
- [ ] “`robot` contributes strongly to this query” for reconstructed kernel contribution.
- [ ] “Tiny model — optimized for short/simple English” as a compact badge.

---

# 3. Recommended implementation strategy

## 3.1 Frontend stack

Use a modern, simple client-only stack.

- [ ] React + TypeScript.
- [ ] Vite for development/build.
- [ ] `motion` / Framer Motion for coordinated UI transitions.
- [ ] KaTeX (`katex` + React wrapper or custom component) for formulas.
- [ ] Zustand or a small custom reducer/store for playback/debugger state.
- [ ] SVG overlays for arrows, data-flow paths, matrix highlights, and moving packets.
- [ ] Prefer native CSS/CSS modules/Tailwind only if it materially speeds implementation; avoid creating a huge styling dependency tree.
- [ ] Use `Float32Array` for model tensors in the browser.
- [ ] Do **not** depend on ONNX for the main traceable inference path unless absolutely necessary. A hand-written tiny inference engine is preferred because every intermediate operation must be inspectable.

## 3.2 Training stack

- [ ] Python 3.11+.
- [ ] PyTorch for training/reference inference.
- [ ] NumPy for export and parity fixtures.
- [ ] Keep model architecture intentionally small and fully mirrored in TypeScript.
- [ ] Export final weights into a browser-loadable format under `public/model/`.
- [ ] Commit exported weights with the repo.

## 3.3 Why this architecture

- [ ] The model is small enough that browser CPU execution is sufficient.
- [ ] Pure TypeScript inference allows exact trace events for every operation.
- [ ] Training in PyTorch is easier and more reliable than training in the browser.
- [ ] A parity test can ensure browser inference is numerically consistent with Python reference inference.

---

# 4. Tiny model specification

The exact dimensions may be tuned if needed, but keep the model small and interpretable.

## 4.1 Baseline architecture

Start with:

- [ ] vocabulary size: about 1,000–3,000 word/punctuation tokens;
- [ ] maximum sequence length: 48 or 64 tokens;
- [ ] model dimension `d_model = 32`;
- [ ] linear-attention feature dimension `d_feature = 16`;
- [ ] value dimension `d_value = 32`;
- [ ] 2 causal linear-attention layers;
- [ ] 1 attention head per layer for visual simplicity;
- [ ] FFN hidden dimension: 64;
- [ ] learned token embeddings;
- [ ] learned positional embeddings or a very simple deterministic position encoding;
- [ ] RMSNorm or LayerNorm; RMSNorm is preferable because the formula is compact;
- [ ] activation in FFN: GELU;
- [ ] linear-attention positive feature map: `phi(x) = ELU(x) + 1 + eps`;
- [ ] small epsilon such as `1e-6` in denominators.

Target parameter count should remain tiny (roughly well under a few hundred thousand parameters).

## 4.2 Causal linear-attention equations

For a token representation `x_t` in a layer:

- [ ] `q_t = x_t W_q + b_q`
- [ ] `k_t = x_t W_k + b_k`
- [ ] `v_t = x_t W_v + b_v`
- [ ] `qphi_t = phi(q_t)`
- [ ] `kphi_t = phi(k_t)`

Maintain running states:

- [ ] `S_(t-1) ∈ R^(d_feature × d_value)`
- [ ] `Z_(t-1) ∈ R^(d_feature)`

Read **before writing the current token** so the current token only reads previous context:

- [ ] `numerator_t = qphi_t^T S_(t-1)`
- [ ] `denominator_t = qphi_t^T Z_(t-1) + eps`
- [ ] `context_t = numerator_t / denominator_t`

Then perform any output projection/residual/norm/FFN required by the architecture.

Then update state:

- [ ] `C_t = kphi_t ⊗ v_t`  (outer product)
- [ ] `S_t = S_(t-1) + C_t`
- [ ] `Z_t = Z_(t-1) + kphi_t`

For the first token, previous state is zero, so context is zero or handled safely.

## 4.3 Two-layer rationale

- [ ] Do not use only one raw-embedding attention layer if the pronoun query must depend on prior sentence context.
- [ ] Layer 1 produces a contextual hidden representation for each token.
- [ ] Layer 2 computes Q/K/V from the Layer-1 contextual representation.
- [ ] This allows a token such as `it` to produce different final queries depending on the preceding sentence.
- [ ] The UI must allow switching between Layer 1 and Layer 2 traces.
- [ ] Default “story mode” may focus on Layer 2 for the final pronoun read while still indicating that Layer 1 already contextualized the token.

## 4.4 NLP task head

The site needs a genuine small NLP task so the final outcome is not merely a pretty state update.

Preferred design:

- [ ] Train a small pronoun/coreference prediction head on the final contextual hidden state.
- [ ] At pronoun positions (`he`, `she`, `it`, `him`, `her`, `they`, `them`, etc.), predict the antecedent’s token identity or entity ID.
- [ ] If predicting token identity, map that predicted token back to matching previous token positions in the sentence.
- [ ] Avoid examples with duplicate candidate tokens during initial training/evaluation, or provide a separate position head if duplicate disambiguation is needed.
- [ ] The final site should show top-k prediction probabilities.

Recommended auxiliary alignment objective:

- [ ] Reconstruct kernel scores over previous token positions in the final linear-attention layer:
  - `score_j = qphi_t · kphi_j`
  - `alpha_j = score_j / sum_{i<t}(score_i + eps)`
- [ ] Add an auxiliary loss encouraging the target antecedent position to receive a strong kernel score.
- [ ] This makes the later “which earlier token contributed?” visualization much easier to interpret while remaining mathematically tied to the real model.

Recommended total training loss:

- [ ] `L = L_coref + λ * L_alignment`
- [ ] Start with `λ` around `0.25–0.5` and tune if needed.

Optional:

- [ ] Add a small language-model/next-token auxiliary loss if training becomes unstable, but do not add complexity unless needed.

---

# 5. Simple-English training domain

The goal is not general English. The goal is a tiny model that behaves reliably on a controlled but interactive language domain.

## 5.1 Tokenizer

Implement a transparent tokenizer in both Python and TypeScript.

- [ ] lowercase for model lookup but preserve original text for display;
- [ ] split punctuation into separate tokens;
- [ ] support basic contractions only if desired;
- [ ] include `<pad>`, `<unk>`, `<bos>`, `<eos>` if needed;
- [ ] tokenizer rules must be identical in Python and browser;
- [ ] add tokenizer parity tests.

Avoid a complex BPE implementation unless needed. A simple word/punctuation tokenizer is more educational.

## 5.2 Vocabulary

Include enough words to support varied examples:

- [ ] common determiners/prepositions/conjunctions;
- [ ] pronouns;
- [ ] 20–50 person names split across gendered/unisex groups where relevant;
- [ ] animals;
- [ ] simple animate objects such as robot/android;
- [ ] household objects;
- [ ] colors;
- [ ] locations;
- [ ] action verbs;
- [ ] semantic clue verbs such as `walked`, `barked`, `rolled`, `fell`, `thanked`, etc.;
- [ ] simple adjectives;
- [ ] punctuation.

Aim for a vocabulary large enough to feel flexible but small enough to ship and train quickly.

## 5.3 Synthetic data generator

Create `model/data.py` or equivalent.

Generate many non-ambiguous training examples with known antecedents.

Template families should include at least:

- [ ] **subject continuation**
  - `the red robot picked up the blue key . it walked toward the door .`
  - antecedent: `robot`
- [ ] **object continuation**
  - `the dog dropped the ball . it rolled under the table .`
  - antecedent: `ball`
- [ ] **animate subject clue**
  - `the dog chased the ball . it barked loudly .`
  - antecedent: `dog`
- [ ] **person-gender clue**
  - `alice handed the map to bob . he thanked alice .`
  - antecedent: `bob`
- [ ] **object pronoun**
  - `maya placed the book on the desk . she opened it .`
  - `she` → `maya`, `it` → `book` (can produce separate supervised positions)
- [ ] **location/object variants**
- [ ] **adjective variants**
- [ ] **word-order variants**
- [ ] **distractor nouns** so the model cannot solve every example by “nearest noun.”
- [ ] **distractor subjects** so it cannot solve every example by “first noun.”
- [ ] **randomized colors/names/objects** to prevent memorizing fixed sentences.

Important:

- [ ] Make training/validation/test splits by generated sentence instances and, where practical, hold out some combinations of names/objects/colors to test compositional behavior.
- [ ] Do not include intentionally ambiguous sentences in supervised evaluation.
- [ ] Add a small set of hand-written human-readable test sentences used by the website’s example picker.

## 5.4 Required model quality

- [ ] Train until the small model reliably solves the supported coreference task on held-out synthetic test data.
- [ ] Target at least ~90% accuracy on the controlled held-out task if practical.
- [ ] If accuracy is poor, tune model dimension, epochs, training data variety, or objective rather than hiding failures.
- [ ] Record final evaluation metrics in `model/metrics.json` and expose a short “model scope” tooltip in the UI.
- [ ] The repo must include already-trained exported weights.

---

# 6. Weight export format

Create a stable browser format under `public/model/`.

Suggested files:

- [ ] `model-config.json`
- [ ] `vocab.json`
- [ ] `weights.json` or compact binary weight files plus an index JSON
- [ ] `metrics.json`
- [ ] `examples.json`

`model-config.json` should include:

- [ ] architecture version;
- [ ] vocab size;
- [ ] max sequence length;
- [ ] `d_model`;
- [ ] `d_feature`;
- [ ] `d_value`;
- [ ] layer count;
- [ ] feature-map name;
- [ ] epsilon;
- [ ] pronoun token IDs;
- [ ] tokenizer version;
- [ ] training metadata;
- [ ] checksum/version for weights.

Implement loader validation:

- [ ] fail with a clear UI error if shapes do not match config;
- [ ] validate numeric arrays;
- [ ] report model loading state visibly but compactly.

---

# 7. Browser tensor/inference engine

Create a small numerical engine rather than relying on hidden kernels.

## 7.1 Tensor primitives

Implement and test:

- [ ] vector add;
- [ ] vector multiply by scalar;
- [ ] dot product;
- [ ] matrix-vector multiplication;
- [ ] vector-matrix multiplication if needed;
- [ ] outer product;
- [ ] matrix add;
- [ ] elementwise ELU;
- [ ] feature map `phi`;
- [ ] RMSNorm/LayerNorm;
- [ ] GELU;
- [ ] softmax;
- [ ] argmax/top-k;
- [ ] cosine similarity only if used for a clearly labeled supplemental view.

Use deterministic `Float32Array` operations.

## 7.2 Trace-first inference design

Do **not** animate while computing.

When the user clicks Run:

1. [ ] tokenize the sentence;
2. [ ] run the entire model forward pass immediately;
3. [ ] generate an immutable trace describing every important operation;
4. [ ] store snapshots needed for rewind/jump;
5. [ ] let the UI replay this trace at human speed.

This separation is critical.

## 7.3 Core trace types

Define strict TypeScript types similar to:

```ts
interface TensorArtifact {
  id: string;
  label: string;
  kind: 'vector' | 'matrix' | 'scalar';
  shape: number[];
  values: Float32Array | number;
  source: 'learned' | 'computed' | 'reconstructed';
  tokenIndex?: number;
  layerIndex?: number;
  opId?: string;
  parentIds?: string[];
  fullPrecision?: boolean;
}

interface TraceEvent {
  id: string;
  index: number;
  tokenIndex?: number;
  layerIndex?: number;
  phase: string;
  kind:
    | 'tokenize'
    | 'embedding_lookup'
    | 'position_add'
    | 'norm'
    | 'q_projection'
    | 'k_projection'
    | 'v_projection'
    | 'feature_map_q'
    | 'feature_map_k'
    | 'state_read_numerator'
    | 'state_read_denominator'
    | 'context_divide'
    | 'outer_product'
    | 'state_s_update'
    | 'state_z_update'
    | 'output_projection'
    | 'residual'
    | 'ffn'
    | 'prediction'
    | 'reconstructed_influence';
  inputIds: string[];
  outputIds: string[];
  formulaKey: string;
  shortLabel: string;
  renderHints?: Record<string, unknown>;
}
```

Adapt as needed, but preserve the concept.

## 7.4 Provenance graph

- [ ] Every computed tensor must reference its parent artifacts/operation.
- [ ] Every learned tensor must point to a model parameter name.
- [ ] Clicking a value must allow the inspector to walk backward through provenance.
- [ ] For huge derivations, calculate the scalar derivation lazily on click rather than storing every multiplication in the trace.

---

# 8. Numerical parity / correctness

This is essential because the browser is displaying “real” calculations.

- [ ] Implement a Python reference forward pass using the same exported weights.
- [ ] Export one or more golden fixtures containing:
  - [ ] token IDs;
  - [ ] selected embeddings;
  - [ ] selected Q/K/V vectors;
  - [ ] selected `phi(K)` values;
  - [ ] selected state matrices;
  - [ ] final context output;
  - [ ] final prediction probabilities.
- [ ] Vitest/Jest must compare TypeScript inference with the Python fixture within reasonable float tolerance (e.g. `1e-4` or tighter).
- [ ] Include at least one full supported sentence fixture such as the robot/key example.
- [ ] If parity fails, fix it before considering the site complete.

---

# 9. Main UI information architecture

The experience should be a **visual computation workspace**, not a long article.

Recommended desktop structure:

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Input / examples / run                                               │
├──────────────────────────────────────────────────────────────────────┤
│ Token timeline                                                       │
├──────────────────────┬─────────────────────┬─────────────────────────┤
│ Token + Embedding    │ Q/K/V Laboratory   │ Contribution Builder    │
│                      │                     │                         │
├──────────────────────┴─────────────────────┼─────────────────────────┤
│ Query / Readout                            │ State Memory S / Z      │
├────────────────────────────────────────────┴─────────────────────────┤
│ Prediction + reconstructed token influence                           │
├──────────────────────────────────────────────────────────────────────┤
│ Playback / stepper / event timeline / current formula                │
└──────────────────────────────────────────────────────────────────────┘
```

Do not force this exact grid if a better responsive layout emerges, but preserve the logical stages.

---

# 10. Input panel

- [ ] Large single-line or compact multi-line sentence input.
- [ ] Placeholder with a supported example.
- [ ] “Run” button.
- [ ] Example sentence dropdown/cards.
- [ ] “Random supported example” action.
- [ ] Character/token limit indicator.
- [ ] Compact model-scope badge: `Tiny model · simple English · max 64 tokens`.
- [ ] OOV count after tokenization.
- [ ] If the user enters unsupported/OOV-heavy text, show a compact warning and still allow inspection.
- [ ] Keep instructional prose minimal.

Suggested built-in examples:

- [ ] `The red robot picked up the blue key. It walked toward the door.`
- [ ] `The dog dropped the ball. It rolled under the table.`
- [ ] `The dog chased the ball. It barked loudly.`
- [ ] `Alice handed the map to Bob. He thanked Alice.`
- [ ] `Maya placed the book on the desk. She opened it.`

---

# 11. Token timeline panel

Permanent horizontal token strip near the top.

For each token show:

- [ ] exact display token;
- [ ] token index;
- [ ] state: future / current / processed;
- [ ] pronoun indicator if relevant;
- [ ] OOV indicator if token mapped to `<unk>`.

Interactions:

- [ ] click token → jump to the first event for that token;
- [ ] shift-click or dedicated action → jump to the completed state after that token;
- [ ] hover → tooltip with token ID, normalized form, position, and event range;
- [ ] processed tokens visually dim/check;
- [ ] current token visually emphasized;
- [ ] future tokens remain muted.

When reconstructed influence is shown:

- [ ] earlier influential tokens should light up in the token strip;
- [ ] draw a curved connection from current query token to selected/highest-influence token(s).

---

# 12. Token + embedding panel

This panel answers “where did the token’s vector come from?”

Animation sequence:

1. [ ] current token card appears;
2. [ ] token passes through tokenizer badge;
3. [ ] token ID appears;
4. [ ] visual embedding table appears or a small slice of it is shown;
5. [ ] corresponding embedding row highlights;
6. [ ] selected row slides out as an embedding vector card;
7. [ ] positional vector appears if used;
8. [ ] vectors add together with animated element updates;
9. [ ] final layer input vector moves to Q/K/V panel.

Default display:

- [ ] show first 6–8 dimensions only;
- [ ] label `showing 8 / 32 dimensions`;
- [ ] “expand full vector” control;
- [ ] tooltips show full precision;
- [ ] visible values may be rounded to 3–4 decimals.

Every scalar should be inspectable:

- [ ] click embedding value → identify token row/dimension and label it **Learned**;
- [ ] click positional value → show where it came from;
- [ ] click combined value → show `embedding[i] + position[i]`.

---

# 13. Layer selector

- [ ] Show Layer 1 and Layer 2 tabs/chips.
- [ ] Display current layer clearly.
- [ ] Story mode may auto-follow the most relevant layer.
- [ ] Advanced users can lock the view to a layer.
- [ ] Switching layers at the same token should update Q/K/V/state panels consistently.
- [ ] Include a small diagram showing `Embedding → Layer 1 → Layer 2 → Task head`.

---

# 14. Q/K/V laboratory panel

This is one of the primary visual panels.

Layout concept:

```text
                   x_t
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       Wq          Wk          Wv
        │           │           │
        ▼           ▼           ▼
        Q           K           V
```

Requirements:

- [ ] Q, K, and V are three distinct visual lanes.
- [ ] Use consistent semantic styling for Q, K, V everywhere in the site.
- [ ] Do not rely on color alone—include `Q`, `K`, `V` labels/icons.
- [ ] Animate projection one lane at a time in slow/step mode.
- [ ] Show vector/matrix shapes.
- [ ] Show formula above/below the active lane.
- [ ] When a projection finishes, the resulting tensor card physically moves toward its next consumer.

Default formula:

- [ ] `Q = x Wq + bq`
- [ ] `K = x Wk + bk`
- [ ] `V = x Wv + bv`

## 14.1 Matrix multiplication microscope

On `Inspect calculation` or click of an output scalar:

- [ ] zoom/open an inspector for one output dimension;
- [ ] show the dot-product equation;
- [ ] animate multiply-add terms;
- [ ] show running sum;
- [ ] allow stepping through individual terms;
- [ ] show matrix row/column cells that participate;
- [ ] trace values back to embedding and learned weight cells.

Example:

`K[0] = x[0]*Wk[0,0] + x[1]*Wk[1,0] + ... + bk[0]`

This mode should exist for every matrix multiplication, not just K.

---

# 15. Feature-map panel

For Q and K, expose `phi()` as its own transformation.

- [ ] Display input vector.
- [ ] Display formula `phi(x) = ELU(x) + 1 + eps`.
- [ ] Animate each visible element changing.
- [ ] For negative values, hover/click should show the ELU branch used.
- [ ] For positive values, show the direct branch.
- [ ] Output `qphi` or `kphi` becomes a new tensor card.
- [ ] Tensor card continues to read/update state panels.

Scalar drilldown example:

- [ ] input `-0.42`
- [ ] `ELU(-0.42) = exp(-0.42) - 1`
- [ ] `+ 1 + eps`
- [ ] final value.

---

# 16. Contribution builder panel

For the write path, visualize:

`C_t = kphi_t ⊗ v_t`

Requirements:

- [ ] show `kphi_t` vertically and `V_t` horizontally;
- [ ] animate the outer-product matrix being filled;
- [ ] show one row/cell at a time in microscope mode;
- [ ] label resulting matrix `Contribution from <token>`;
- [ ] show shape `d_feature × d_value`;
- [ ] contribution matrix becomes a movable tensor card;
- [ ] animate this card flowing into the state panel.

Clicking a contribution cell must show:

- [ ] `C[i,j] = kphi[i] * V[j]`;
- [ ] exact source values;
- [ ] resulting scalar.

---

# 17. Linear-attention state panel — centerpiece

This is the most important panel.

Show both:

- [ ] main state `S` — matrix;
- [ ] normalization state `Z` — vector.

Always make clear which snapshot is being shown:

- [ ] `S_(t-1)` before current write;
- [ ] current contribution `C_t`;
- [ ] `S_t` after write;
- [ ] same for `Z`.

## 17.1 State update animation

For `S_t = S_(t-1) + C_t`:

- [ ] do not instantly replace the old matrix;
- [ ] keep an `OLD` ghost copy visible;
- [ ] move the incoming contribution over/next to it;
- [ ] highlight corresponding cells;
- [ ] animate values transitioning from old → new;
- [ ] in normal mode update row-by-row or region-by-region;
- [ ] in microscope mode update one scalar at a time;
- [ ] after completion, move/fade old state into history and relabel new state as current.

For `Z_t = Z_(t-1) + kphi_t`:

- [ ] animate vector addition in parallel or immediately after S update;
- [ ] show the branch from `kphi_t`: one branch contributes to `C_t`, one branch adds directly to `Z`.

## 17.2 State history

Under the state panel show a compact history scrubber:

`S0 → S1 → S2 → S3 → ...`

- [ ] each state marker maps to a token;
- [ ] click marker → inspect that snapshot;
- [ ] hover marker → token + layer + step summary;
- [ ] compare two adjacent snapshots;
- [ ] toggle `Difference view` showing `ΔS = S_t - S_(t-1) = C_t`.

## 17.3 Contribution decomposition view

Provide two tabs:

- [ ] `Combined state`
- [ ] `Contribution layers`

Contribution layers may visually stack:

`C_The + C_red + C_robot + ...`

This view is **reconstructed/presentation**, but mathematically reflects the sum.

- [ ] label it clearly as “Contribution decomposition.”
- [ ] allow clicking a token layer to isolate its matrix contribution.
- [ ] allow opacity control or solo/mute toggles for selected token contributions.
- [ ] provide “recombine” animation where layers collapse back into `S`.

---

# 18. Query/read path panel

When a token arrives, especially a pronoun such as `It`, show the read operation separately from the write operation.

Sequence:

1. [ ] `Q_t` produced;
2. [ ] `qphi_t` produced;
3. [ ] qphi tensor card moves toward the current pre-write state `S_(t-1)` and `Z_(t-1)`;
4. [ ] calculate numerator;
5. [ ] calculate denominator;
6. [ ] divide to obtain context;
7. [ ] context flows through output projection/residual/norm/FFN as applicable.

Explicit formulas:

- [ ] `N_t = qphi_t^T S_(t-1)`
- [ ] `D_t = qphi_t^T Z_(t-1) + eps`
- [ ] `context_t = N_t / D_t`

Show shapes throughout.

## 18.1 Numerator visualization

- [ ] qphi vector displayed vertically/horizontally as appropriate;
- [ ] state matrix displayed;
- [ ] highlight one output dimension at a time in microscope mode;
- [ ] show dot-product terms and running sum;
- [ ] resulting numerator vector appears.

## 18.2 Denominator visualization

- [ ] show qphi dot Z;
- [ ] resulting scalar appears;
- [ ] epsilon addition appears visibly but compactly.

## 18.3 Context division

- [ ] numerator vector positioned above denominator scalar;
- [ ] animate each visible numerator element dividing by denominator;
- [ ] contextual vector card appears and moves onward.

---

# 19. Residual / normalization / FFN panel

The model may require these blocks for useful behavior. Do not hide them entirely.

Default story mode:

- [ ] collapsed compact pipeline chips: `Residual → Norm → FFN → Residual → Norm`.
- [ ] animate data passing through them.
- [ ] show output vector.

On expansion:

- [ ] show exact formulas;
- [ ] show RMS calculation for norm;
- [ ] allow scalar drilldown;
- [ ] show FFN matrices and GELU;
- [ ] support matrix multiplication microscope exactly like Q/K/V.

This keeps default UI readable while maintaining full inspectability.

---

# 20. Prediction panel

When the current token is a pronoun or a task-relevant token:

- [ ] show task-head output probabilities.
- [ ] top prediction should be visually prominent.
- [ ] show top 3–5 alternatives.
- [ ] if predicted antecedent token exists earlier in sentence, highlight that token.
- [ ] draw a relationship arc from pronoun to predicted antecedent.
- [ ] label as `Model prediction`, not “ground truth.”
- [ ] if confidence is low, visibly show uncertainty.

Example:

```text
Model prediction for “It”
robot   0.82
key     0.11
door    0.04
other   0.03
```

Do not hard-code these numbers; use the actual model output.

---

# 21. Reconstructed token-influence panel

This is the “where did the query relate most strongly?” educational view.

For final-layer current query `qphi_t`, reconstruct previous kernel similarities:

- [ ] `score_j = qphi_t · kphi_j`
- [ ] normalized `alpha_j = score_j / sum(score)`
- [ ] keep previous `kphi_j` in the explanation trace even though efficient recurrent inference only needs the aggregate state.

Display as animated bars or vertical token columns.

- [ ] bars grow as reconstruction is calculated;
- [ ] matching token in token timeline pulses/highlights;
- [ ] hover each bar → exact dot-product formula and values;
- [ ] click a bar → open full Q-vs-K dot-product microscope;
- [ ] include a label such as `Reconstructed kernel influence`.

Critical explanatory note in a tooltip:

- [ ] Efficient linear inference did not need to revisit all previous tokens.
- [ ] This per-token view is reconstructed for teaching from stored trace data.

This panel is where a result like `robot` receiving the strongest influence can be visualized.

---

# 22. Data lineage / “trace this value” system

Every important scalar/vector/matrix should support provenance inspection.

Interaction:

- [ ] hover → compact tooltip;
- [ ] click → pin/open inspector;
- [ ] inspector contains `Trace this value` action;
- [ ] draw animated lines back to parent tensors/operations.

Examples:

For a final context cell:

`context[3]`

trace backward:

- [ ] context division;
- [ ] numerator[3];
- [ ] denominator;
- [ ] Q feature map;
- [ ] S and Z states;
- [ ] state contributions;
- [ ] original K/V values;
- [ ] embeddings and learned matrices.

For a state cell:

`S_t[i,j]`

show:

- [ ] `S_t[i,j] = S_(t-1)[i,j] + C_t[i,j]`
- [ ] then `C_t[i,j] = kphi_t[i] * V_t[j]`
- [ ] then trace `kphi_t[i]` and `V_t[j]` backward.

For a Q/K/V scalar:

- [ ] show every multiply-add term on demand.

Implement lineage as a graph using artifact IDs/operation IDs rather than hard-coded UI logic.

---

# 23. Floating/current formula dock

Keep a compact formula area always visible near the playback controls.

It should update with the active trace event.

Examples:

- [ ] `K = x Wk + bk`
- [ ] `phi(K) = ELU(K) + 1 + eps`
- [ ] `C_t = phi(K_t) ⊗ V_t`
- [ ] `S_t = S_(t-1) + C_t`
- [ ] `N_t = phi(Q_t)^T S_(t-1)`
- [ ] `context_t = N_t / D_t`

Progressively substitute real values where useful.

Example progression:

1. symbolic;
2. current token substituted;
3. visible numeric slice;
4. result.

Render formulas with KaTeX.

Keep long explanations in hover/click tooltips rather than next to formulas.

---

# 24. Tensor cards / data packets

Create a reusable visual component representing tensors.

Each card should show:

- [ ] tensor name (`K_robot`, `S_8`, etc.);
- [ ] shape;
- [ ] small preview of values;
- [ ] source badge (`Learned`, `Computed`, `Reconstructed`);
- [ ] token/layer identity if applicable.

Animation behavior:

- [ ] cards physically travel between panels;
- [ ] cards can split/branch when one tensor feeds multiple operations;
- [ ] cards merge visually during addition/multiplication;
- [ ] completed cards may shrink into history.

Prefer a coherent motion language:

- [ ] movement = data dependency;
- [ ] pulse = value changed;
- [ ] glow/emphasis = currently active operand;
- [ ] fade = inactive historical context;
- [ ] dashed path = reconstructed/explanation-only path;
- [ ] solid path = actual forward-pass path.

---

# 25. State transition visual language

The user specifically wants to see “how one state is imposed over another state” and transitions.

Implement:

- [ ] old state remains visible temporarily;
- [ ] incoming contribution floats over or beside it;
- [ ] corresponding cells line up;
- [ ] values interpolate from old to new;
- [ ] changed cells flash/pulse;
- [ ] unchanged/near-unchanged cells remain quiet;
- [ ] after update, old state becomes a ghost and moves into history;
- [ ] new state takes the primary state position.

Optional but recommended:

- [ ] heatmap mode showing magnitude of state change;
- [ ] `Δ` overlay showing only the current contribution;
- [ ] slider morphing between `S_(t-1)` and `S_t`.

---

# 26. Playback / debugger controls

Persistent control bar:

- [ ] `Previous operation`
- [ ] `Next operation`
- [ ] `Previous token`
- [ ] `Next token`
- [ ] `Play / Pause`
- [ ] speed selector/slider (e.g. 0.25×, 0.5×, 1×, 2×, 4×)
- [ ] `Restart`
- [ ] `Jump to pronoun`
- [ ] `Jump to final prediction`
- [ ] `Microscope mode`
- [ ] `Visual / Math / Balanced` density mode

Keyboard shortcuts:

- [ ] left/right arrows for operation stepping;
- [ ] shift+left/right for token stepping;
- [ ] space for play/pause;
- [ ] `R` restart;
- [ ] escape closes inspectors.

Always display:

- [ ] current step index / total steps;
- [ ] current token;
- [ ] current layer;
- [ ] current operation short label.

Example:

`Step 147 / 382 · token “robot” · Layer 2 · State update`

---

# 27. Animation scheduler

Animations must be deterministic and tied to trace events.

- [ ] Each trace event has enter/active/exit timing.
- [ ] Playback speed scales those durations.
- [ ] User stepping must complete/cancel the current animation cleanly before moving.
- [ ] Rewind must restore exact previous snapshots, not reverse floating-point computation.
- [ ] Jumping to a step should directly restore the target UI state, then optionally animate the final event.
- [ ] Use `requestAnimationFrame`/Motion, not `setInterval` chains.
- [ ] Respect browser `prefers-reduced-motion`; when enabled, use fades/highlights rather than long spatial movement.

---

# 28. Three explanation depths

Every major panel should support:

## 28.1 Visual mode

- [ ] minimal text;
- [ ] vector/matrix cards;
- [ ] moving connections;
- [ ] current formula only;
- [ ] simple values;
- [ ] best default for first-time users.

## 28.2 Numbers mode

- [ ] more tensor dimensions;
- [ ] shapes;
- [ ] before/after values;
- [ ] state deltas;
- [ ] top influence scores.

## 28.3 Microscope/deep math mode

- [ ] scalar-level derivations;
- [ ] full matrices where feasible;
- [ ] learned weight inspection;
- [ ] multiply-add stepping;
- [ ] provenance graph.

Do not duplicate three entirely different pages; these should be views of the same underlying trace.

---

# 29. Full-dimension inspection

Although default views show only a subset of dimensions:

- [ ] allow opening a modal/drawer/table containing the full vector/matrix;
- [ ] searchable by index;
- [ ] hover reveals full-precision value;
- [ ] click any cell opens provenance;
- [ ] matrix supports zoom/pan if needed;
- [ ] include shape and memory size.

Do not claim the visible 6 dimensions are the whole vector.

---

# 30. Standard attention comparison mode

After users understand linear attention, provide a final comparison.

Split-screen concept:

```text
STANDARD SELF-ATTENTION           LINEAR ATTENTION
query → every previous token     token → running S/Z state
pairwise map grows               state shape stays fixed
```

Requirements:

- [ ] use the same token count slider for both sides;
- [ ] visualize pairwise query-key connections/grid on the standard side;
- [ ] visualize fixed-shape recurrent state on linear side;
- [ ] show theoretical sequence-length scaling with fixed feature sizes:
  - [ ] standard attention pairwise score storage/work grows ~quadratically in sequence length;
  - [ ] linear-attention recurrence grows ~linearly in sequence length.
- [ ] Avoid oversimplifying total model complexity; explicitly say “with model/feature dimensions fixed.”
- [ ] Add a `+ tokens` interaction where the standard grid visibly expands while the linear state dimensions remain constant.
- [ ] This mode may use simulated token counts after the real sentence; label it as a scaling visualization.

---

# 31. UI styling direction

Desired feel: scientific debugger / interactive lab, not marketing landing page.

- [ ] dark or neutral high-contrast base theme;
- [ ] optional light theme only if easy;
- [ ] monospaced font for tensors/numbers;
- [ ] readable sans-serif for labels;
- [ ] restrained semantic accents for Q/K/V/state/prediction;
- [ ] stable semantic styling throughout the app;
- [ ] rounded panels acceptable, but avoid generic card-heavy SaaS feel;
- [ ] use thin connector lines, node labels, matrix grids, and motion;
- [ ] avoid decorative stock imagery;
- [ ] avoid walls of explanatory text;
- [ ] tooltips and inspectors carry detailed explanations;
- [ ] permanent labels should be short: `Token`, `Embed`, `Q`, `K`, `V`, `phi(K)`, `S`, `Z`, `Read`, `Predict`.

---

# 32. Hover / tooltip system

All important interactive elements should reveal extra information without clutter.

Tooltips may include:

- [ ] definition;
- [ ] tensor shape;
- [ ] source category;
- [ ] current token/layer;
- [ ] exact value/full precision;
- [ ] formula;
- [ ] “click to inspect calculation.”

Examples:

Hover `Q`:

- [ ] `Query vector`
- [ ] `Computed from current token representation using Wq.`

Hover `S`:

- [ ] `Running linear-attention memory`
- [ ] `Sum of previous kphi ⊗ V contributions in this layer.`

Hover reconstructed influence:

- [ ] `Explanation reconstruction`
- [ ] `Efficient forward inference did not need to revisit each previous token.`

---

# 33. Inspector drawer

Clicking a tensor or scalar opens a persistent inspector drawer.

Sections:

- [ ] Identity
- [ ] Shape
- [ ] Source category
- [ ] Token/layer
- [ ] Formula
- [ ] Inputs
- [ ] Output value(s)
- [ ] Parent operations
- [ ] Child/consumer operations
- [ ] `Trace backward`
- [ ] `Trace forward`
- [ ] `Open full tensor`
- [ ] `Pin while playback continues`

Inspector should not interrupt playback unless the user pauses.

---

# 34. Relationship between actual efficient state and explanation history

The implementation should preserve two concepts:

## 34.1 Actual model state

- [ ] only `S` and `Z` are required for efficient recurrent linear-attention reads (plus layer hidden values as needed for current token processing).

## 34.2 Visualization trace history

- [ ] keep per-token K, V, qphi/kphi, contributions, and snapshots so the user can rewind and reconstruct influence.
- [ ] explicitly mark that this extra history exists for visualization/debugging.
- [ ] do not imply the efficient algorithm requires keeping all prior token tensors in memory.

This distinction is a core learning outcome.

---

# 35. Supported sentence handling

- [ ] Detect token count > max sequence length and prevent run with compact explanation.
- [ ] Detect OOV tokens.
- [ ] Show OOV tokens distinctly in timeline.
- [ ] If OOV ratio is high, show `Model may be unreliable for this sentence`.
- [ ] Still allow the user to inspect the computation if technically valid.
- [ ] Add example picker to guarantee good demonstrations.
- [ ] If sentence contains no supported pronoun, still visualize all linear-attention mechanics and show a generic final representation/task-head result.
- [ ] If there are multiple pronouns, allow user to select which pronoun to inspect.

---

# 36. Example “robot” walkthrough that must work

The repository should contain and validate this canonical example:

`The red robot picked up the blue key. It walked toward the door.`

Expected visual story:

- [ ] tokenize the sentence;
- [ ] process each token sequentially;
- [ ] show state evolving;
- [ ] emphasize the `robot` token contribution when processed;
- [ ] later process `It`;
- [ ] show Layer-2 `Q_it` and feature map;
- [ ] show query reading the pre-write state;
- [ ] show numerator, denominator, context;
- [ ] show final task prediction;
- [ ] model should ideally predict/highlight `robot`;
- [ ] reconstructed kernel-influence view should preferably also assign a meaningful/high contribution to `robot`;
- [ ] draw a clear relationship arc `It → robot` only if the actual model output supports it.

Do not hard-code `robot` as the answer. If model training does not produce the desired behavior, improve the model/training data.

---

# 37. Training workflow requirements

Create repeatable scripts.

Suggested commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r model/requirements.txt
python model/train.py
python model/export.py
python model/validate.py
```

Codex should adapt commands for the repo environment.

Requirements:

- [ ] fixed random seed by default;
- [ ] configurable seed;
- [ ] save best validation checkpoint;
- [ ] early stopping optional;
- [ ] progress logging;
- [ ] final train/val/test metrics;
- [ ] export browser weights automatically after successful training;
- [ ] generate golden parity fixture automatically;
- [ ] store model version/hash;
- [ ] do not require GPU;
- [ ] GPU may be used if available but CPU must remain supported.

Add quick/dev mode:

- [ ] fewer generated samples/epochs for fast iteration;
- [ ] clearly separate quick mode from final training.

Final committed weights must come from the full/final training configuration.

---

# 38. Suggested repository structure

Adapt to current repo if necessary.

```text
/
├─ handoff.md
├─ README.md
├─ package.json
├─ vite.config.ts
├─ tsconfig.json
├─ src/
│  ├─ app/
│  │  ├─ App.tsx
│  │  └─ store.ts
│  ├─ model/
│  │  ├─ types.ts
│  │  ├─ tokenizer.ts
│  │  ├─ tensor.ts
│  │  ├─ featureMap.ts
│  │  ├─ linearAttention.ts
│  │  ├─ layer.ts
│  │  ├─ inference.ts
│  │  ├─ trace.ts
│  │  ├─ lineage.ts
│  │  └─ loadModel.ts
│  ├─ playback/
│  │  ├─ scheduler.ts
│  │  ├─ selectors.ts
│  │  └─ snapshots.ts
│  ├─ components/
│  │  ├─ InputPanel.tsx
│  │  ├─ TokenTimeline.tsx
│  │  ├─ TensorCard.tsx
│  │  ├─ VectorView.tsx
│  │  ├─ MatrixView.tsx
│  │  ├─ EmbeddingPanel.tsx
│  │  ├─ QKVPanel.tsx
│  │  ├─ FeatureMapPanel.tsx
│  │  ├─ ContributionPanel.tsx
│  │  ├─ StatePanel.tsx
│  │  ├─ ReadoutPanel.tsx
│  │  ├─ PredictionPanel.tsx
│  │  ├─ InfluencePanel.tsx
│  │  ├─ FormulaDock.tsx
│  │  ├─ PlaybackControls.tsx
│  │  ├─ InspectorDrawer.tsx
│  │  └─ DataFlowOverlay.tsx
│  ├─ formulas/
│  │  └─ registry.ts
│  ├─ styles/
│  └─ tests/
├─ public/
│  └─ model/
│     ├─ model-config.json
│     ├─ vocab.json
│     ├─ weights.json
│     ├─ metrics.json
│     └─ examples.json
├─ model/
│  ├─ architecture.py
│  ├─ tokenizer.py
│  ├─ data.py
│  ├─ train.py
│  ├─ export.py
│  ├─ validate.py
│  ├─ requirements.txt
│  └─ fixtures/
└─ e2e/
```

---

# 39. Formula registry

Do not hard-code formulas independently in many components.

Create a central registry where each operation defines:

- [ ] symbolic formula;
- [ ] short explanation;
- [ ] expected tensor shapes;
- [ ] source category;
- [ ] optional scalar expansion builder;
- [ ] KaTeX string.

Example keys:

- [ ] `embedding_lookup`
- [ ] `position_add`
- [ ] `q_projection`
- [ ] `k_projection`
- [ ] `v_projection`
- [ ] `elu_feature_map`
- [ ] `outer_product`
- [ ] `state_s_update`
- [ ] `state_z_update`
- [ ] `state_read_numerator`
- [ ] `state_read_denominator`
- [ ] `context_divide`
- [ ] `rms_norm`
- [ ] `ffn`
- [ ] `coref_head`
- [ ] `kernel_reconstruction`

---

# 40. Responsive behavior

Desktop is primary, but the site should remain usable on smaller screens.

- [ ] Desktop: multi-panel grid with visible data-flow paths.
- [ ] Tablet: stack some panels vertically, keep timeline sticky.
- [ ] Mobile: sequential panel carousel/accordion with a sticky mini timeline and playback controls.
- [ ] Do not attempt to show six tiny matrices side by side on mobile.
- [ ] Preserve inspectability even if layout changes.

---

# 41. Accessibility

- [ ] Keyboard navigation for all controls.
- [ ] Visible focus states.
- [ ] Tooltips accessible via focus as well as hover.
- [ ] `prefers-reduced-motion` support.
- [ ] Do not encode Q/K/V/state meaning by color alone.
- [ ] Sufficient contrast for matrix text and highlights.
- [ ] ARIA labels for playback buttons and interactive cells where practical.

---

# 42. Performance requirements

The model is tiny, so prioritize trace clarity over micro-optimization.

- [ ] Avoid re-running inference on every animation step.
- [ ] Generate trace once per submitted sentence.
- [ ] Memoize tensor previews and matrix cell formatting.
- [ ] Virtualize full matrices only if necessary.
- [ ] Avoid React re-rendering every cell at 60fps; use memoized components/canvas/SVG intelligently if required.
- [ ] Keep normal interaction smooth on a typical laptop browser.
- [ ] No network request should be required after static assets/model weights load.

---

# 43. Testing requirements

## 43.1 Unit tests

- [ ] tokenizer tests;
- [ ] tensor primitive tests;
- [ ] feature map tests;
- [ ] outer product tests;
- [ ] state update tests;
- [ ] readout formula tests;
- [ ] normalization tests;
- [ ] softmax/top-k tests;
- [ ] trace provenance tests;
- [ ] snapshot/rewind tests.

## 43.2 Python/TypeScript parity tests

- [ ] golden sentence fixture;
- [ ] compare key intermediate vectors/matrices;
- [ ] compare final predictions.

## 43.3 UI/component tests

- [ ] token click jumps correctly;
- [ ] operation stepper moves one event;
- [ ] token stepper moves to proper boundary;
- [ ] rewind restores exact previous values;
- [ ] inspector shows correct parents;
- [ ] formula dock matches active event;
- [ ] OOV warning appears.

## 43.4 End-to-end tests

Use Playwright or equivalent.

At minimum:

- [ ] load app;
- [ ] run canonical robot sentence;
- [ ] verify token timeline appears;
- [ ] step through several operations;
- [ ] jump to `It`;
- [ ] open Q/K/V inspector;
- [ ] open state difference view;
- [ ] reach final prediction;
- [ ] ensure prediction panel contains actual model output;
- [ ] verify no console errors;
- [ ] verify production build serves correctly.

---

# 44. Development/CI scripts

Add useful npm scripts, adapting names as needed:

- [ ] `npm run dev`
- [ ] `npm run build`
- [ ] `npm run preview`
- [ ] `npm run lint`
- [ ] `npm run typecheck`
- [ ] `npm run test`
- [ ] `npm run test:e2e`
- [ ] `npm run model:verify`

Optional:

- [ ] `npm run model:train`
- [ ] `npm run model:export`

If cross-language scripts are awkward, add `scripts/` shell/Python helpers.

---

# 45. README requirements

Update `README.md` with:

- [ ] what the project demonstrates;
- [ ] screenshot/GIF placeholder or generated preview if easy;
- [ ] prerequisites;
- [ ] install command;
- [ ] dev command;
- [ ] test commands;
- [ ] production build command;
- [ ] explanation of model scope;
- [ ] statement that model inference is client-side;
- [ ] statement that the displayed tensors are actual model calculations;
- [ ] training/retraining instructions;
- [ ] how to add vocabulary/templates;
- [ ] how parity is validated;
- [ ] limitations.

---

# 46. Minimal permanent text in the actual website

The site should not feel like reading documentation.

Permanent text should mainly be:

- [ ] panel names;
- [ ] active formulas;
- [ ] tensor labels;
- [ ] token labels;
- [ ] short operation labels;
- [ ] prediction results;
- [ ] tiny model-scope badge;
- [ ] buttons/controls.

Move these into hover/click explanations:

- [ ] definitions of Q/K/V;
- [ ] why linear attention uses `S` and `Z`;
- [ ] explanation of reconstruction;
- [ ] why the tiny model is limited;
- [ ] derivation of formulas;
- [ ] learned-vs-computed distinction;
- [ ] state-history explanation.

---

# 47. Animation storyboard for one token

For every normal token, the replay should be capable of showing this sequence:

1. [ ] timeline marks token current;
2. [ ] token moves to tokenizer;
3. [ ] token ID appears;
4. [ ] embedding row selected;
5. [ ] embedding vector extracted;
6. [ ] positional information added;
7. [ ] layer input enters Layer 1;
8. [ ] Q projected;
9. [ ] K projected;
10. [ ] V projected;
11. [ ] Q feature map;
12. [ ] K feature map;
13. [ ] current Q reads old `S/Z`;
14. [ ] numerator calculated;
15. [ ] denominator calculated;
16. [ ] context calculated;
17. [ ] residual/norm/FFN path;
18. [ ] contribution outer product calculated;
19. [ ] contribution flows into S;
20. [ ] old S + contribution → new S;
21. [ ] K feature vector flows into Z;
22. [ ] old Z + kphi → new Z;
23. [ ] Layer 1 output enters Layer 2;
24. [ ] repeat layer operations;
25. [ ] final token hidden state stored;
26. [ ] if pronoun, task head prediction displayed;
27. [ ] timeline marks token processed;
28. [ ] next token becomes current.

Story mode may collapse several events together. Microscope mode must expose them individually.

---

# 48. Special storyboard for pronoun token

When current token is `It`/`he`/`she` etc.:

- [ ] visually pause/emphasize the read path;
- [ ] show the current query as a distinct moving packet;
- [ ] visibly query `S_(t-1)` and `Z_(t-1)` before current token is written;
- [ ] derive context;
- [ ] show final-layer output;
- [ ] run real coreference head;
- [ ] highlight predicted antecedent in token timeline;
- [ ] then open reconstructed kernel influence bars;
- [ ] connect strongest influence tokens to pronoun with arcs;
- [ ] only after the read/prediction sequence, animate current token’s own state write.

This creates the strongest “aha” moment.

---

# 49. Operation counters / scaling visualization

Optional but strongly recommended:

- [ ] count scalar multiply/add operations for selected linear-attention operations;
- [ ] display cumulative operation count as tokens progress;
- [ ] standard-attention comparison can display theoretical pairwise score count `n²`;
- [ ] linear side can display number of fixed-shape state updates;
- [ ] make clear this is a simplified educational operation count, not a benchmark of complete model runtime.

---

# 50. Avoid these implementation traps

- [ ] Do not make a static animation with hard-coded vectors.
- [ ] Do not hard-code the robot example’s prediction.
- [ ] Do not secretly call an external LLM/API.
- [ ] Do not make every panel permanently show huge matrices; default should be readable.
- [ ] Do not call raw embedding cosine similarity “coreference.”
- [ ] Do not say the model “remembers robot in cell X.”
- [ ] Do not treat reconstructed per-token influence as proof of linguistic reasoning.
- [ ] Do not omit `Z` and then introduce it only in the final equation.
- [ ] Do not update the state instantly without showing the old state and contribution.
- [ ] Do not animate computation directly without a replayable trace.
- [ ] Do not make rewind recompute floats in reverse.
- [ ] Do not use a giant model that makes the browser experience heavy or opaque.
- [ ] Do not rely on server inference.
- [ ] Do not leave placeholder TODOs in the final delivered experience.

---

# 51. Suggested implementation phases

Codex should execute these, not merely describe them.

## Phase A — inspect and scaffold

- [ ] inspect existing repo;
- [ ] preserve useful existing structure;
- [ ] create/update frontend stack;
- [ ] establish lint/typecheck/test commands;
- [ ] commit/record model folder structure.

## Phase B — train tiny model

- [ ] implement tokenizer;
- [ ] implement synthetic data generator;
- [ ] implement two-layer causal linear-attention model;
- [ ] implement coreference/task head;
- [ ] train and evaluate;
- [ ] tune until examples and held-out metrics are acceptable;
- [ ] export final weights/config/vocab/metrics/examples;
- [ ] generate parity fixtures.

## Phase C — browser inference engine

- [ ] implement typed-array math;
- [ ] implement model loader;
- [ ] reproduce Python forward pass;
- [ ] implement trace generation;
- [ ] pass parity tests.

## Phase D — core debugger UI

- [ ] sentence input;
- [ ] token timeline;
- [ ] playback controls;
- [ ] tensor cards;
- [ ] formula dock;
- [ ] Q/K/V panel;
- [ ] state panel;
- [ ] readout panel;
- [ ] prediction panel.

## Phase E — deep inspection

- [ ] matrix microscope;
- [ ] full tensor viewer;
- [ ] state history;
- [ ] difference view;
- [ ] provenance/lineage inspector;
- [ ] reconstructed influence.

## Phase F — animation polish

- [ ] data-flow SVG paths;
- [ ] moving tensor cards;
- [ ] state overlay transitions;
- [ ] query read animation;
- [ ] pronoun relationship arc;
- [ ] reduced-motion support.

## Phase G — comparison mode

- [ ] standard vs linear attention visualization;
- [ ] token-count scaling interaction;
- [ ] theoretical operation/storage indicators.

## Phase H — validation and delivery

- [ ] run all tests;
- [ ] run typecheck/lint;
- [ ] run production build;
- [ ] run end-to-end tests against production build;
- [ ] test canonical examples manually;
- [ ] fix console warnings/errors;
- [ ] update README;
- [ ] ensure exported model assets are committed;
- [ ] leave clean git status except intentional changes.

---

# 52. Acceptance checklist — functionality

The project is not complete until all applicable items pass.

- [ ] User can enter a sentence.
- [ ] Sentence is tokenized visibly.
- [ ] Token IDs are real.
- [ ] OOV tokens are marked.
- [ ] Learned embedding values load from model assets.
- [ ] Browser derives Q/K/V with actual trained matrices.
- [ ] User can inspect one Q/K/V scalar down to multiply-add terms.
- [ ] Browser applies real feature map.
- [ ] Browser constructs real contribution outer product.
- [ ] Browser updates real `S` state.
- [ ] Browser updates real `Z` state.
- [ ] State history can be scrubbed.
- [ ] Old/new state can be compared.
- [ ] Query reads pre-write state.
- [ ] Numerator/denominator/context are inspectable.
- [ ] Two linear-attention layers run correctly.
- [ ] Coreference/task head uses actual model output.
- [ ] Top-k predictions are displayed.
- [ ] Predicted antecedent is highlighted if present.
- [ ] Reconstructed token influence uses actual qphi/kphi values.
- [ ] Playback can step operation-by-operation.
- [ ] Playback can step token-by-token.
- [ ] Playback can run automatically.
- [ ] Playback can rewind reliably.
- [ ] User can jump to any token.
- [ ] User can jump to pronoun/final prediction.
- [ ] Formula dock follows current operation.
- [ ] Full tensor view exists.
- [ ] Inspector/provenance exists.
- [ ] Visual mode has low text density.
- [ ] Hover/click provides deeper explanations.
- [ ] Canonical example behaves sensibly.
- [ ] At least several built-in examples behave sensibly.

---

# 53. Acceptance checklist — realism

- [ ] No hard-coded intermediate values.
- [ ] No hard-coded final answers.
- [ ] Learned values originate in committed exported weights.
- [ ] Browser model math matches Python reference.
- [ ] `S` and `Z` shown are the real model states.
- [ ] “Reconstructed” views are explicitly labeled.
- [ ] Truncated vector displays say how many dimensions are hidden.
- [ ] OOV/uncertain free-form inputs are not presented as guaranteed correct.
- [ ] Site identifies itself as a tiny educational model, not a full LLM.

---

# 54. Acceptance checklist — visual quality

- [ ] Main screen is understandable without reading long paragraphs.
- [ ] Active token is obvious.
- [ ] Active operation is obvious.
- [ ] Data visually travels between stages.
- [ ] Q/K/V lanes are distinct and consistent.
- [ ] State updates visibly transition old → new.
- [ ] Query read path is visually distinct from state write path.
- [ ] Matrices remain legible.
- [ ] Values do not visually jump without transition in normal playback.
- [ ] Hover tooltips contain useful detail.
- [ ] Inspector can stay open while navigating.
- [ ] UI remains responsive during playback.
- [ ] Reduced-motion mode still communicates state changes.

---

# 55. Acceptance checklist — engineering quality

- [ ] `npm install` succeeds.
- [ ] `npm run dev` succeeds.
- [ ] `npm run build` succeeds.
- [ ] `npm run typecheck` succeeds.
- [ ] `npm run lint` succeeds, or there is an intentionally documented equivalent.
- [ ] unit tests pass.
- [ ] parity tests pass.
- [ ] e2e tests pass.
- [ ] no runtime console errors on canonical flow.
- [ ] no missing model assets.
- [ ] repo contains training code and exported final weights.
- [ ] README is complete.

---

# 56. Final delivery behavior for Codex

When implementing this handoff:

- [ ] Make reasonable technical decisions without repeatedly asking the user for permission.
- [ ] Prefer working software over over-engineering.
- [ ] If a proposed package causes friction, replace it with a simpler implementation.
- [ ] If training accuracy is inadequate, iterate on the synthetic data/model until the supported examples work credibly.
- [ ] If one visualization becomes too heavy, preserve the underlying feature but simplify its rendering.
- [ ] Do not remove inspectability to save time.
- [ ] Do not fake any model output to satisfy the demo.
- [ ] Keep the architecture small enough that future contributors can understand it.
- [ ] Before finishing, run the full acceptance checklist and fix failures.

---

# 57. Concise conceptual model the UI should convey

The entire experience should visually communicate this sequence:

```text
Sentence
  ↓
Tokens
  ↓
Learned embeddings
  ↓
Q / K / V
  ↓
phi(Q), phi(K)
  ↓
            WRITE PATH
K,V ──→ token contribution ──→ S,Z updated

            READ PATH
Q ───────────────────────────→ previous S,Z
                                ↓
                              context
                                ↓
                         contextual token state
                                ↓
                           NLP prediction
```

And the key state identity:

```text
S_t = S_(t-1) + phi(K_t) ⊗ V_t
Z_t = Z_(t-1) + phi(K_t)
```

And the key read identity:

```text
context_t = [phi(Q_t)^T S_(t-1)] / [phi(Q_t)^T Z_(t-1) + eps]
```

The user should be able to pause at **any arrow** in this diagram and inspect the exact numeric data flowing through it.

---

# 58. Definition of done

This project is done only when a fresh user can:

1. [ ] open the site;
2. [ ] choose or type a short English sentence;
3. [ ] press Run;
4. [ ] watch tokens move through the model;
5. [ ] see real embeddings and Q/K/V values;
6. [ ] see a token contribution added to `S` and `Z`;
7. [ ] rewind and inspect the state before/after that token;
8. [ ] arrive at a pronoun;
9. [ ] watch its query read the accumulated state;
10. [ ] inspect numerator, denominator, and contextual output;
11. [ ] see a real model prediction for the antecedent;
12. [ ] inspect reconstructed influence from earlier tokens;
13. [ ] click any important number and trace where it came from;
14. [ ] compare linear attention with standard pairwise attention;
15. [ ] understand the mechanism primarily from visuals and motion rather than long text.

If any of these are only mocked, hard-coded, or disconnected from the actual browser model computation, the implementation is incomplete.
