# openai-schema

Small, strongly typed structured-output wrapper for OpenAI's Responses API.

It owns schema attachment, JSON extraction, optional conversation IDs, retries and per-instance serialisation. The application still chooses the model, prompt, reasoning, storage, token limits and every other request option. Managed-conversation compaction is available only when a caller explicitly opts in.

## Use

```ts
import { OpenAISchema, array, object, shape, string } from "openai-schema";

interface Reply {
  text: string;
}

const reply = shape<Reply>(
  "reply",
  object({ text: string() }),
);

const ai = new OpenAISchema(process.env.OPENAI_API_KEY!, reply);

const out = await ai.send(
  [{ role: "user", content: "Write one sentence." }],
  {
    body: {
      model: "gpt-5.4-nano",
      store: false,
      max_output_tokens: 80,
    },
  },
);
```

`OpenAISchema` creates an OpenAI conversation lazily and exposes its ID through `id`. Pass `{ conversation: false }` for stateless calls, or pass an existing conversation ID as the third constructor argument.

Responses API input may be supplied as a string or an array of input items. Other JSON-serialisable values are encoded as JSON strings before transmission, including values returned by `onRetry`.

## Managed conversation compaction

Compaction is deliberately backwards-compatible and request-scoped. Existing callers do not change behaviour unless they set the optional `compaction` Boolean:

```ts
const out = await ai.send(input, {
  compaction: true,
  body: { model },
});
```

When `compaction: true` is present and the caller has not already supplied `body.context_management`, the wrapper adds:

```json
{
  "context_management": [
    { "type": "compaction" }
  ]
}
```

OpenAI then manages server-side compaction for the Responses call. Omitting `compaction`, or setting it to `false`, adds nothing and preserves the legacy request shape. A caller that needs lower-level control may still provide its own `body.context_management`, including a `compact_threshold`; the wrapper leaves that value untouched.

For managed conversations, opt-in compaction also provides a last-resort rollover path. If the Responses API still returns a recognised context-window HTTP 400, the wrapper retrieves the existing conversation items, sends the relevant state to `/responses/compact`, creates a fresh managed conversation, seeds the retry with the returned opaque compaction item plus the current input, and changes `id` only after that retry succeeds. Unrelated HTTP 400 errors do not trigger rollover. If compaction or rollover itself fails, the original context-window error is preserved.

Response token usage is normalised into `Usage`. `lastUsage` exposes the most recent Responses or compaction result, and optional `onUsage` receives usage from each successful API result, including retry and fallback-compaction calls:

```ts
await ai.send(input, {
  body: { model },
  onUsage: usage => {
    console.log(usage.inputTokens, usage.totalTokens);
  },
});
```

The normalised fields are `inputTokens`, `outputTokens`, `totalTokens`, `cachedTokens` and `reasoningTokens`.

## Mutable output types

```ts
interface ListReply {
  items: string[];
}

const listReply = shape<ListReply>(
  "list_reply",
  object({ items: array(string(), 3, 3) }),
);

const changed = await ai.updateSchema(listReply);
const list = await changed.send(input, { body: { model } });
```

For concurrent callers, prefer `run()`. It changes the active shape and performs the request in one queued operation:

```ts
const list = await ai.run(listReply, input, {
  body: { model, store: false },
});
```

The wrapper never needs to know the application's interface in advance. A `Shape<T>` supplies the name, JSON Schema and optional runtime parser for one call.

## API

- `send(input, options)` uses the current `Shape<T>`.
- `run(shape, input, options)` atomically changes shape and sends.
- `updateSchema(shape)` mutates the current generic type.
- `id` returns the managed conversation ID.
- `lastUsage` returns the most recent normalised Responses token usage when supplied by OpenAI.
- `isBusy` and `queued` expose queue state.
- `registerTool()` retains generic tool metadata for hosts that use it.
- `compaction: true` opts one `send()`/`run()` call into server-side Responses compaction and managed-conversation rollover recovery.
- `onUsage` observes usage from each successful Responses or compaction result without changing the returned structured output.

The package has no OpenAI SDK dependency. It uses `fetch`, so it works in modern Node.js, Bun, workers and other Web API runtimes.

## Development

```bash
npm install
npm run ci
```

All authored source and tests are strict TypeScript. Generated JavaScript and declarations belong in ignored build directories.
