import type { Model } from './types';
export async function loadModel(): Promise<Model> {
  const [config, vocab, weights] = await Promise.all(['model-config.json', 'vocab.json', 'weights.json'].map(file => fetch(`/model/${file}`).then(r => { if (!r.ok) throw new Error(`Could not load ${file}`); return r.json(); })));
  if (weights.embeddings.length !== vocab.length || weights.embeddings[0].length !== config.dModel) throw new Error('Model asset shapes do not match configuration.');
  return { config, vocab, ...weights };
}
