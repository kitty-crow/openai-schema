declare module "node:assert/strict" {
  interface Assert {
    equal(actual: unknown, expected: unknown): void;
    deepEqual(actual: unknown, expected: unknown): void;
  }
  const assert: Assert;
  export default assert;
}

declare module "node:test" {
  type Test = (name: string, fn: () => void | Promise<void>) => void;
  const test: Test;
  export default test;
}
