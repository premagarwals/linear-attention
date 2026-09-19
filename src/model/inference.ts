import { add, dot, matVec, outer, phi, softmax } from './tensor';
import { tokenize } from './tokenizer';
import type { Artifact, EventKind, Model, TokenTrace, Trace, TraceEvent } from './types';

const eventLabels: Record<string, string> = { tokenize: 'Tokenize input', embedding_lookup: 'Embedding lookup', q_projection: 'Query projection', k_projection: 'Key projection', v_projection: 'Value projection', feature_map_q: 'Feature map φ(Q)', feature_map_k: 'Feature map φ(K)', state_read_numerator: 'Read numerator', state_read_denominator: 'Read denominator', context_divide: 'Normalize context', outer_product: 'Build contribution', state_s_update: 'Update memory S', state_z_update: 'Update normalizer Z', prediction: 'Coreference prediction', reconstructed_influence: 'Reconstruct token influence' };

// This is intentionally kept outside the visible trace: it runs only after the
// complete sentence is available and supplies the full-sentence resolver.
function reverseOutputs(model: Model, tokenIds: number[]): Float32Array[] | undefined {
  if (!model.reverseLayers?.length) return undefined;
  const states = model.reverseLayers.map(() => ({ s: new Float32Array(model.config.dFeature * model.config.dModel), z: new Float32Array(model.config.dFeature) }));
  const outputs = Array.from({ length: tokenIds.length }, () => new Float32Array(model.config.dModel));
  for (let t = tokenIds.length - 1; t >= 0; t--) {
    let x = Float32Array.from(model.embeddings[tokenIds[t]]);
    for (let layer = 0; layer < model.reverseLayers.length; layer++) {
      const weights = model.reverseLayers[layer]; const state = states[layer];
      const qphi = phi(matVec(x, weights.wq, weights.bq), model.config.epsilon);
      const kphi = phi(matVec(x, weights.wk, weights.bk), model.config.epsilon);
      const v = matVec(x, weights.wv, weights.bv);
      const numerator = new Float32Array(model.config.dModel);
      for (let j = 0; j < model.config.dModel; j++) for (let i = 0; i < model.config.dFeature; i++) numerator[j] += qphi[i] * state.s[i * model.config.dModel + j];
      const context = Float32Array.from(numerator, value => value / (dot(qphi, state.z) + model.config.epsilon));
      const contribution = outer(kphi, v);
      for (let i = 0; i < state.s.length; i++) state.s[i] += contribution[i];
      for (let i = 0; i < state.z.length; i++) state.z[i] += kphi[i];
      x = add(x, matVec(context, weights.wo));
    }
    outputs[t] = x;
  }
  return outputs;
}

export function run(model: Model, text: string): Trace {
  const words = tokenize(text).slice(0, model.config.maxLength); const artifacts: Record<string, Artifact> = {}; const events: TraceEvent[] = []; const result: TokenTrace[] = [];
  const put = (id: string, label: string, values: Float32Array | number, shape: number[], source: Artifact['source'], tokenIndex?: number, layerIndex?: number, parentIds?: string[]) => artifacts[id] = { id, label, values, shape, source, tokenIndex, layerIndex, parentIds };
  const record = (kind: string, tokenIndex: number | undefined, layerIndex: number | undefined, inputIds: string[] = [], outputIds: string[] = []) => events.push({ id: `op-${events.length}`, index: events.length, kind: kind as EventKind, tokenIndex, layerIndex, inputIds, outputIds, formulaKey: kind, shortLabel: eventLabels[kind] });
  model.layers.forEach((weights, layer) => {
    put(`wq-${layer}`, `Wq layer ${layer + 1}`, Float32Array.from(weights.wq.flat()), [model.config.dModel, model.config.dFeature], 'learned', undefined, layer);
    put(`wk-${layer}`, `Wk layer ${layer + 1}`, Float32Array.from(weights.wk.flat()), [model.config.dModel, model.config.dFeature], 'learned', undefined, layer);
    put(`wv-${layer}`, `Wv layer ${layer + 1}`, Float32Array.from(weights.wv.flat()), [model.config.dModel, model.config.dModel], 'learned', undefined, layer);
  });
  record('tokenize', undefined, undefined);
  const states = model.layers.map(() => ({ s: new Float32Array(model.config.dFeature * model.config.dModel), z: new Float32Array(model.config.dFeature) }));
  for (let t = 0; t < words.length; t++) {
    const vocabId = model.vocab.indexOf(words[t]); const id = vocabId >= 0 ? vocabId : 0; let x = Float32Array.from(model.embeddings[id]); const layers: TokenTrace['layers'] = [];
    const embedId = `embed-${t}`; put(embedId, `embedding(${words[t]})`, x, [model.config.dModel], 'learned', t); record('embedding_lookup', t, undefined, [], [embedId]);
    for (let layer = 0; layer < model.layers.length; layer++) {
      const weights = model.layers[layer]; const state = states[layer]; const beforeS = Float32Array.from(state.s); const beforeZ = Float32Array.from(state.z); const inputId = layer ? `output-${t}-${layer - 1}` : embedId; const prefix = `${t}-${layer}`;
      put(`s-before-${prefix}`, `S before ${words[t]}`, beforeS, [model.config.dFeature, model.config.dModel], 'computed', t, layer); put(`z-before-${prefix}`, `Z before ${words[t]}`, beforeZ, [model.config.dFeature], 'computed', t, layer);
      const q = matVec(x, weights.wq, weights.bq); const k = matVec(x, weights.wk, weights.bk); const v = matVec(x, weights.wv, weights.bv);
      put(`q-${prefix}`, 'Q', q, [model.config.dFeature], 'computed', t, layer, [inputId, `wq-${layer}`]); put(`k-${prefix}`, 'K', k, [model.config.dFeature], 'computed', t, layer, [inputId, `wk-${layer}`]); put(`v-${prefix}`, 'V', v, [model.config.dModel], 'computed', t, layer, [inputId, `wv-${layer}`]);
      record('q_projection', t, layer, [inputId, `wq-${layer}`], [`q-${prefix}`]); record('k_projection', t, layer, [inputId, `wk-${layer}`], [`k-${prefix}`]); record('v_projection', t, layer, [inputId, `wv-${layer}`], [`v-${prefix}`]);
      const qphi = phi(q, model.config.epsilon); const kphi = phi(k, model.config.epsilon); put(`qphi-${prefix}`, 'φ(Q)', qphi, [model.config.dFeature], 'computed', t, layer, [`q-${prefix}`]); put(`kphi-${prefix}`, 'φ(K)', kphi, [model.config.dFeature], 'computed', t, layer, [`k-${prefix}`]); record('feature_map_q', t, layer, [`q-${prefix}`], [`qphi-${prefix}`]); record('feature_map_k', t, layer, [`k-${prefix}`], [`kphi-${prefix}`]);
      const numerator = new Float32Array(model.config.dModel); for (let j = 0; j < model.config.dModel; j++) for (let i = 0; i < model.config.dFeature; i++) numerator[j] += qphi[i] * state.s[i * model.config.dModel + j]; put(`numerator-${prefix}`, 'Read numerator', numerator, [model.config.dModel], 'computed', t, layer, [`qphi-${prefix}`, `s-before-${prefix}`]); record('state_read_numerator', t, layer, [`qphi-${prefix}`, `s-before-${prefix}`], [`numerator-${prefix}`]);
      const denominator = dot(qphi, state.z) + model.config.epsilon; put(`denominator-${prefix}`, 'Read denominator', denominator, [], 'computed', t, layer, [`qphi-${prefix}`, `z-before-${prefix}`]); record('state_read_denominator', t, layer, [`qphi-${prefix}`, `z-before-${prefix}`], [`denominator-${prefix}`]);
      const context = Float32Array.from(numerator, n => n / denominator); put(`context-${prefix}`, 'Context', context, [model.config.dModel], 'computed', t, layer, [`numerator-${prefix}`, `denominator-${prefix}`]); record('context_divide', t, layer, [`numerator-${prefix}`, `denominator-${prefix}`], [`context-${prefix}`]);
      const contribution = outer(kphi, v); put(`contribution-${prefix}`, 'Contribution C', contribution, [model.config.dFeature, model.config.dModel], 'computed', t, layer, [`kphi-${prefix}`, `v-${prefix}`]); for (let i = 0; i < state.s.length; i++) state.s[i] += contribution[i]; for (let i = 0; i < state.z.length; i++) state.z[i] += kphi[i]; put(`s-after-${prefix}`, 'S after', Float32Array.from(state.s), [model.config.dFeature, model.config.dModel], 'computed', t, layer, [`s-before-${prefix}`, `contribution-${prefix}`]); put(`z-after-${prefix}`, 'Z after', Float32Array.from(state.z), [model.config.dFeature], 'computed', t, layer, [`z-before-${prefix}`, `kphi-${prefix}`]); record('outer_product', t, layer, [`kphi-${prefix}`, `v-${prefix}`], [`contribution-${prefix}`]); record('state_s_update', t, layer, [`s-before-${prefix}`, `contribution-${prefix}`], [`s-after-${prefix}`]); record('state_z_update', t, layer, [`z-before-${prefix}`, `kphi-${prefix}`], [`z-after-${prefix}`]);
      const output = add(x, matVec(context, weights.wo)); put(`output-${prefix}`, 'Residual output', output, [model.config.dModel], 'computed', t, layer, [inputId, `context-${prefix}`]);
      layers.push({ q, k, v, qphi, kphi, numerator, denominator, context, output, beforeS, afterS: Float32Array.from(state.s), beforeZ, afterZ: Float32Array.from(state.z), contribution });
      x = output;
    }
    result.push({ token: words[t], id, oov: vocabId < 0, layers });
  }
  const backward = reverseOutputs(model, result.map(token => token.id));
  for (let t = 0; t < result.length; t++) if (['it', 'he', 'she', 'they'].includes(result[t].token)) {
    const fullSentence = Boolean(backward && model.pointerFull); const entityIds = new Set(model.config.entityTokenIds ?? []);
    const positions = result.map((candidate, index) => entityIds.has(candidate.id) && (fullSentence ? index !== t : index < t) ? index : -1).filter(index => index >= 0);
    const forwardOutput = result[t].layers.at(-1)!.output;
    const queryInput = fullSentence ? add(forwardOutput, backward![t]) : forwardOutput;
    const pointerQuery = matVec(queryInput, fullSentence ? model.pointerFull! : model.pointer);
    const scores = Float32Array.from(positions, position => {
      const candidate = result[position].layers.at(-1)!.output;
      return dot(pointerQuery, fullSentence ? add(candidate, backward![position]) : candidate);
    });
    const probabilities = [...softmax(scores)]; const influence = result.slice(0, t).map(previous => dot(result[t].layers.at(-1)!.qphi, previous.layers.at(-1)!.kphi)); const sum = influence.reduce((a,b) => a + b, 0) || 1;
    const probabilityVector = Float32Array.from(probabilities); put(`prediction-${t}`, fullSentence ? 'Full-sentence candidate probabilities' : 'Causal candidate probabilities', probabilityVector, [positions.length], 'computed', t, model.layers.length - 1, [`output-${t}-${model.layers.length - 1}`]); const normalized = influence.map(value => value / sum); put(`influence-${t}`, 'Reconstructed forward influence', Float32Array.from(normalized), [normalized.length], 'reconstructed', t, model.layers.length - 1); result[t].prediction = { labels: positions.map(position => `${result[position].token} · #${position}`), probabilities, positions, influence: normalized, fullSentence }; record('prediction', t, model.layers.length - 1, [`output-${t}-${model.layers.length - 1}`], [`prediction-${t}`]); record('reconstructed_influence', t, model.layers.length - 1, [], [`influence-${t}`]);
  }
  return { tokens: result, artifacts, events };
}
