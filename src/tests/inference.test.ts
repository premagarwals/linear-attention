import { describe, expect, it } from 'vitest';
import { run } from '../model/inference';
import type { Model } from '../model/types';

const model: Model = { config: { dModel: 2, dFeature: 2, maxLength: 10, epsilon: 1e-6, version: 'test', entityTokenIds: [1] }, vocab: ['<unk>', 'it'], embeddings: [[0, 0], [1, 2]], layers: Array.from({ length: 2 }, () => ({ wq: [[1, 0], [0, 1]], wk: [[1, 0], [0, 1]], wv: [[1, 0], [0, 1]], wo: [[0, 0], [0, 0]], bq: [0, 0], bk: [0, 0], bv: [0, 0] })), head: [[0, 0], [0, 0]], pointer: [[1, 0], [0, 1]] };
describe('causal linear attention trace', () => {
  it('reads state before the token contribution is written', () => {
    const trace = run(model, 'it it');
    expect([...trace.tokens[0].layers[0].context]).toEqual([0, 0]);
    expect(trace.tokens[1].layers[0].denominator).toBeGreaterThan(0);
    expect(trace.tokens[1].layers[0].beforeS.some(x => x !== 0)).toBe(true);
    expect(trace.events.some(event => event.kind === 'state_s_update')).toBe(true);
    const stateUpdate = trace.events.find(event => event.kind === 'state_s_update')!;
    expect(stateUpdate.inputIds).toHaveLength(2);
    expect(trace.artifacts[stateUpdate.outputIds[0]].parentIds).toEqual(stateUpdate.inputIds);
  });
});
