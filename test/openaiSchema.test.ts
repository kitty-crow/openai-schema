import assert from "node:assert/strict";
import test from "node:test";
import {
  OpenAISchema,
  array,
  object,
  shape,
  string,
  type Dict,
  type Fetch,
} from "../src/openaiSchema.js";

type Out = { text: string };
type Next = { items: string[] };

function response(value: unknown, status = 200): Response {
  return new Response(typeof value === "string" ? value : JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const out = shape<Out>("out", object({ text: string() }));
const next = shape<Next>("next", object({ items: array(string(), 2, 2) }));

test("creates and reuses a managed conversation", async () => {
  const calls: Array<{ url: string; body: Dict }> = [];
  const fetcher: Fetch = async (input, init) => {
    const url = String(input);
    const body = JSON.parse(String(init?.body ?? "{}")) as Dict;
    calls.push({ url, body });
    if (url.endsWith("/conversations")) return response({ id: "conv_1" });
    return response({ output_text: JSON.stringify({ text: "hello" }) });
  };

  const client = new OpenAISchema("secret", out, undefined, { fetch: fetcher });
  const value = await client.send([{ role: "user", content: "hi" }], {
    body: { model: "model-a", store: false },
  });

  assert.deepEqual(value, { text: "hello" });
  assert.equal(client.id, "conv_1");
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[1]?.body.conversation, { id: "conv_1" });
  assert.equal((calls[1]?.body.text as Dict).format instanceof Object, true);
});

test("can remain stateless and leaves model settings to the caller", async () => {
  const calls: Dict[] = [];
  const fetcher: Fetch = async (_input, init) => {
    calls.push(JSON.parse(String(init?.body)) as Dict);
    return response({ output: [{ content: [{ text: JSON.stringify({ text: "ok" }) }] }] });
  };

  const client = new OpenAISchema("secret", out, undefined, {
    fetch: fetcher,
    conversation: false,
  });
  const value = await client.send("prompt", {
    body: {
      model: "model-b",
      reasoning: { effort: "low" },
      max_output_tokens: 40,
      store: false,
    },
  });

  assert.equal(value.text, "ok");
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.model, "model-b");
  assert.equal(calls[0]?.conversation, undefined);
});

test("mutates T with updateSchema", async () => {
  const fetcher: Fetch = async () => response({ output_text: JSON.stringify({ items: ["a", "b"] }) });
  const client = new OpenAISchema("secret", out, undefined, { fetch: fetcher, conversation: false });
  const changed = await client.updateSchema(next);
  const value = await changed.send("prompt", { body: { model: "model-c" } });
  assert.deepEqual(value, { items: ["a", "b"] });
});

test("run changes schema and request atomically", async () => {
  const names: string[] = [];
  const fetcher: Fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body)) as Dict;
    const format = (body.text as Dict).format as Dict;
    const name = String(format.name);
    names.push(name);
    await new Promise(resolve => setTimeout(resolve, name === "first" ? 20 : 0));
    return response({
      output_text: name === "first"
        ? JSON.stringify({ text: "one" })
        : JSON.stringify({ items: ["two", "three"] }),
    });
  };

  const client = new OpenAISchema("secret", out, undefined, { fetch: fetcher, conversation: false });
  const first = shape<Out>("first", object({ text: string() }));
  const second = shape<Next>("second", object({ items: array(string(), 2, 2) }));

  const [a, b] = await Promise.all([
    client.run(first, "a", { body: { model: "model-d" } }),
    client.run(second, "b", { body: { model: "model-d" } }),
  ]);

  assert.deepEqual(a, { text: "one" });
  assert.deepEqual(b, { items: ["two", "three"] });
  assert.deepEqual(names, ["first", "second"]);
});

test("retries parsing and permits generic input replacement", async () => {
  let count = 0;
  const inputs: unknown[] = [];
  const fetcher: Fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body)) as Dict;
    inputs.push(body.input);
    count += 1;
    return count === 1
      ? response({ output_text: "not json" })
      : response({ output_text: JSON.stringify({ text: "fixed" }) });
  };

  const client = new OpenAISchema("secret", out, undefined, { fetch: fetcher, conversation: false });
  const value = await client.send("first", {
    body: { model: "model-e" },
    retries: 1,
    retryDelayMs: 0,
    onRetry: () => "second",
  });

  assert.equal(value.text, "fixed");
  assert.deepEqual(inputs, ["first", "second"]);
});
