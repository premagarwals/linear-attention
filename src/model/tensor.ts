export const dot = (a: Float32Array, b: Float32Array) => { let sum = 0; for (let i = 0; i < a.length; i++) sum += a[i] * b[i]; return sum; };
export const add = (a: Float32Array, b: Float32Array) => Float32Array.from(a, (v, i) => v + b[i]);
export const matVec = (x: Float32Array, matrix: number[][], bias?: number[]) => Float32Array.from({ length: matrix[0].length }, (_, column) => (bias?.[column] ?? 0) + x.reduce((sum, value, row) => sum + value * matrix[row][column], 0));
export const outer = (a: Float32Array, b: Float32Array) => Float32Array.from({ length: a.length * b.length }, (_, i) => a[Math.floor(i / b.length)] * b[i % b.length]);
export const phi = (x: Float32Array, epsilon: number) => Float32Array.from(x, value => (value >= 0 ? value : Math.expm1(value)) + 1 + epsilon);
export const softmax = (x: Float32Array) => { const max = Math.max(...x); const values = Float32Array.from(x, value => Math.exp(value - max)); const total = values.reduce((a, b) => a + b, 0); return Float32Array.from(values, value => value / total); };
