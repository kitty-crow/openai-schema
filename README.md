# openai-schema

Small, strongly typed structured-output wrapper for OpenAI's Responses API.

It owns schema attachment, JSON extraction, optional conversation IDs, retries and per-instance serialisation. The application still chooses the model, prompt, reasoning, storage, token limits and every other request option.

## Use

```ts
import { OpenAISchema, object, shape, string } from "openai-schema";

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
- `isBusy` and `queued` expose queue state.
- `registerTool()` retains generic tool metadata for hosts that use it.

The package has no OpenAI SDK dependency. It uses `fetch`, so it works in modern Node.js, Bun, workers and other Web API runtimes.

## Development

```bash
npm install
npm run ci
```

All authored source and tests are strict TypeScript. Generated JavaScript and declarations belong in ignored build directories.
