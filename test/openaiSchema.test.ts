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
  type Usage,
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

test("compaction is request-level opt-in and false preserves legacy requests", async () => {
  const calls: Dict[] = [];
  const fetcher: Fetch = async (_input, init) => {
    calls.push(JSON.parse(String(init?.body)) as Dict);
    return response({ output_text: JSON.stringify({ text: "ok" }) });
  };

  const client = new OpenAISchema("secret", out, undefined, {
    fetch: fetcher,
    conversation: false,
  });

  await client.send("legacy", { body: { model: "model-b" } });
  await client.send("explicit-off", { body: { model: "model-b" }, compaction: false });
  await client.send("on", { body: { model: "model-b" }, compaction: true });
  await client.send("caller-configured", {
    body: {
      model: "model-b",
      context_management: [{ type: "compaction", compact_threshold: 1234 }],
    },
    compaction: true,
  });

  assert.equal(calls[0]?.context_management, undefined);
  assert.equal(calls[1]?.context_management, undefined);
  assert.deepEqual(calls[2]?.context_management, [{ type: "compaction" }]);
  assert.deepEqual(calls[3]?.context_management, [{ type: "compaction", compact_threshold: 1234 }]);
});

test("exposes usage from every Responses API result", async () => {
  const seen: Usage[] = [];
  let count = 0;
  const fetcher: Fetch = async () => {
    count += 1;
    return count === 1
      ? response({
          output_text: "not json",
          usage: {
            input_tokens: 100,
            output_tokens: 5,
            total_tokens: 105,
            input_tokens_details: { cached_tokens: 80 },
            output_tokens_details: { reasoning_tokens: 2 },
          },
        })
      : response({
          output_text: JSON.stringify({ text: "fixed" }),
          usage: {
            input_tokens: 120,
            output_tokens: 8,
            total_tokens: 128,
          },
        });
  };

  const client = new OpenAISchema("secret", out, undefined, {
    fetch: fetcher,
    conversation: false,
  });
  const value = await client.send("prompt", {
    body: { model: "model-b" },
    retries: 1,
    retryDelayMs: 0,
    onUsage: stats => seen.push(stats),
  });

  assert.equal(value.text, "fixed");
  assert.deepEqual(seen, [
    {
      inputTokens: 100,
      outputTokens: 5,
      totalTokens: 105,
      cachedTokens: 80,
      reasoningTokens: 2,
    },
    {
      inputTokens: 120,
      outputTokens: 8,
      totalTokens: 128,
      cachedTokens: 0,
      reasoningTokens: 0,
    },
  ]);
  assert.deepEqual(client.lastUsage, seen[1]);
});

test("rolls over a managed conversation after a genuine context-window failure", async () => {
  const calls: Array<{ url: string; method: string; body?: Dict }> = [];
  let responseCalls = 0;
  const fetcher: Fetch = async (input, init) => {
    const url = String(input);
    const method = String(init?.method ?? "GET");
    const body = init?.body === undefined ? undefined : JSON.parse(String(init.body)) as Dict;
    calls.push({ url, method, ...(body === undefined ? {} : { body }) });

    if (url.includes("/conversations/conv_old/items?")) {
      return response({
        object: "list",
        data: [
          { type: "message", role: "user", content: "old question" },
          { type: "message", role: "assistant", content: [{ type: "output_text", text: "old answer" }] },
        ],
        first_id: "item_1",
        last_id: "item_2",
        has_more: false,
      });
    }
    if (url.endsWith("/responses/compact")) {
      return response({
        id: "cmp_1",
        object: "response.compaction",
        output: [
          { type: "message", role: "user", content: "old question" },
          { type: "compaction", id: "cmp_item", encrypted_content: "opaque" },
        ],
        usage: { input_tokens: 400, output_tokens: 20, total_tokens: 420 },
      });
    }
    if (url.endsWith("/conversations")) return response({ id: "conv_new" });
    if (url.endsWith("/responses")) {
      responseCalls += 1;
      if (responseCalls === 1) {
        return response({
          error: {
            code: "context_length_exceeded",
            message: "Maximum context length exceeded",
          },
        }, 400);
      }
      return response({
        id: "resp_new",
        output_text: JSON.stringify({ text: "recovered" }),
        usage: { input_tokens: 90, output_tokens: 5, total_tokens: 95 },
      });
    }
    return response({ error: { message: "unexpected" } }, 500);
  };

  const seen: Usage[] = [];
  const client = new OpenAISchema("secret", out, "conv_old", { fetch: fetcher });
  const value = await client.send("new turn", {
    body: { model: "model-a", instructions: "Keep continuity." },
    compaction: true,
    onUsage: stats => seen.push(stats),
  });

  assert.equal(value.text, "recovered");
  assert.equal(client.id, "conv_new");
  assert.deepEqual(seen, [
    { inputTokens: 400, outputTokens: 20, totalTokens: 420, cachedTokens: 0, reasoningTokens: 0 },
    { inputTokens: 90, outputTokens: 5, totalTokens: 95, cachedTokens: 0, reasoningTokens: 0 },
  ]);

  const compact = calls.find(call => call.url.endsWith("/responses/compact"));
  assert.equal(compact?.body?.model, "model-a");
  assert.equal(compact?.body?.instructions, "Keep continuity.");

  const retried = calls.filter(call => call.url.endsWith("/responses"))[1];
  assert.deepEqual(retried?.body?.conversation, { id: "conv_new" });
  assert.deepEqual(retried?.body?.input, [
    { type: "compaction", id: "cmp_item", encrypted_content: "opaque" },
    { role: "user", content: "new turn" },
  ]);
});

test("does not rollover on unrelated HTTP 400 errors", async () => {
  const urls: string[] = [];
  const fetcher: Fetch = async (input) => {
    urls.push(String(input));
    return response({ error: { code: "invalid_request_error", message: "Schema is invalid" } }, 400);
  };

  const client = new OpenAISchema("secret", out, "conv_old", { fetch: fetcher });
  let caught: unknown;
  try {
    await client.send("prompt", { body: { model: "model-a" }, compaction: true });
  } catch (error: unknown) {
    caught = error;
  }

  assert.equal(caught instanceof Error, true);
  assert.equal(urls.length, 1);
  assert.equal(urls[0]?.endsWith("/responses"), true);
  assert.equal(client.id, "conv_old");
});

test("normalises generic input for the Responses API", async () => {
  const inputs: unknown[] = [];
  const fetcher: Fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body)) as Dict;
    inputs.push(body.input);
    return response({ output_text: JSON.stringify({ text: "ok" }) });
  };

  const client = new OpenAISchema("secret", out, undefined, {
    fetch: fetcher,
    conversation: false,
  });

  await client.send({ topic: "astrology", count: 2 }, { body: { model: "model-b" } });
  await client.send([{ role: "user", content: "hello" }], { body: { model: "model-b" } });
  await client.send(42, { body: { model: "model-b" } });

  assert.deepEqual(inputs, [
    JSON.stringify({ topic: "astrology", count: 2 }),
    [{ role: "user", content: "hello" }],
    "42",
  ]);
});

test("rejects input that cannot be represented by the Responses API", async () => {
  let calls = 0;
  const fetcher: Fetch = async () => {
    calls += 1;
    return response({ output_text: JSON.stringify({ text: "unreachable" }) });
  };

  const client = new OpenAISchema("secret", out, undefined, {
    fetch: fetcher,
    conversation: false,
  });

  const circular: { self?: unknown } = {};
  circular.self = circular;

  let caught: unknown;
  try {
    await client.send(circular, { body: { model: "model-b" } });
  } catch (error: unknown) {
    caught = error;
  }

  assert.equal(caught instanceof TypeError, true);
  assert.equal(
    caught instanceof Error ? caught.message : "",
    "OpenAI input must be a string, an array of input items, or JSON-serialisable",
  );
  assert.equal(calls, 0);
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

test("retries parsing and normalises generic input replacement", async () => {
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
  const value = await client.send({ attempt: "first" }, {
    body: { model: "model-e" },
    retries: 1,
    retryDelayMs: 0,
    onRetry: () => ({ attempt: "second" }),
  });

  assert.equal(value.text, "fixed");
  assert.deepEqual(inputs, [
    JSON.stringify({ attempt: "first" }),
    JSON.stringify({ attempt: "second" }),
  ]);
});
