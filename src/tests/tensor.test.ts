import { describe, expect, it } from 'vitest';
import { dot, outer, phi } from '../model/tensor';
import { tokenize } from '../model/tokenizer';
describe('tensor primitives', () => {
  it('computes a dot product and outer product', () => { expect(dot(new Float32Array([2, 3]), new Float32Array([4, 5]))).toBe(23); expect([...outer(new Float32Array([2, 3]), new Float32Array([4, 5]))]).toEqual([8, 10, 12, 15]); });
  it('keeps the feature map positive', () => expect([...phi(new Float32Array([-2, 0, 3]), 1e-6)].every(x => x > 0)).toBe(true));
  it('tokenizes lowercase words and punctuation transparently', () => expect(tokenize('The Robot, walked.')).toEqual(['the', 'robot', ',', 'walked', '.']));
});
