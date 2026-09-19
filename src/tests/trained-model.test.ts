import { describe, expect, it } from 'vitest';
import config from '../../public/model/model-config.json';
import vocab from '../../public/model/vocab.json';
import weights from '../../public/model/weights.json';
import { run } from '../model/inference';
import type { Model } from '../model/types';

const model: Model = { config, vocab, ...weights };
describe('trained browser export', () => {
  it('predicts robot for the canonical supported sentence using exported weights', () => {
    const trace = run(model, 'The red robot picked up the blue key. It walked toward the door.');
    const pronoun = trace.tokens.find(token => token.token === 'it')!;
    const top = pronoun.prediction!.probabilities.reduce((best, value, index, values) => value > values[best] ? index : best, 0);
    expect(pronoun.prediction!.labels[top]).toContain('robot');
    expect(pronoun.layers[1].beforeS.some(value => value !== 0)).toBe(true);
  });
  it('never proposes an entity absent from the current sentence', () => {
    const trace = run(model, 'Maya handed Alice a key, but she thanked Maya.');
    const she = trace.tokens.find(token => token.token === 'she')!;
    expect(she.prediction!.labels.join(' ')).not.toContain('bob');
    expect(she.prediction!.labels.every(label => /maya|alice|key/.test(label))).toBe(true);
  });
  it('uses the completed sentence to resolve a later antecedent', () => {
    const trace = run(model, 'Before she walked, Maya thanked Bob.');
    const she = trace.tokens.find(token => token.token === 'she')!;
    const top = she.prediction!.probabilities.reduce((best, value, index, values) => value > values[best] ? index : best, 0);
    expect(she.prediction!.fullSentence).toBe(true);
    expect(she.prediction!.labels[top]).toContain('maya');
  });
  it('handles recipient-role pronouns without offering unrelated vocabulary', () => {
    const trace = run(model, 'Alice handed a box to Emma and she thanked Alice.');
    const she = trace.tokens.find(token => token.token === 'she')!;
    const top = she.prediction!.probabilities.reduce((best, value, index, values) => value > values[best] ? index : best, 0);
    expect(she.prediction!.labels[top]).toContain('emma');
    expect(she.prediction!.labels.join(' ')).not.toContain('bob');
  });
});
