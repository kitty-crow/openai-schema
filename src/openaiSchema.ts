export type Dict = Record<string, unknown>;
export type Schema = Dict;
export type Fetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
export type Parse<T> = (value: unknown) => T;

export interface Shape<T extends object> {
  readonly name: string;
  readonly schema: Schema;
  readonly parse?: Parse<T>;
  readonly description?: string;
  readonly strict?: boolean;
}

export type ShapeDef<T extends object> = Shape<T> | Schema;

export interface Retry {
  readonly attempt: number;
  readonly maxAttempts: number;
  readonly rawText: string;
  readonly error: string;
  readonly input: unknown;
}

export interface Send {
  readonly body: Dict & { model: string };
  readonly signal?: AbortSignal;
  readonly retries?: number;
  readonly retryDelayMs?: number;
  readonly onRetry?: (info: Retry) => unknown | void | Promise<unknown | void>;
}

export type ToolFn = (params: Dict) => unknown | Promise<unknown>;

export interface Tool {
  readonly name: string;
  readonly description: string;
  readonly parameters: Schema;
  readonly handler: string | ToolFn;
}

export interface Init {
  readonly fetch?: Fetch;
  readonly base?: string;
  readonly headers?: Readonly<Record<string, string>>;
  readonly conversation?: boolean;
  readonly name?: string;
}

export class OpenAIError extends Error {
  public readonly status: number;
  public readonly body: string;

  public constructor(status: number, body: string) {
    super(`OpenAI request failed with status ${status}`);
    this.name = "OpenAIError";
    this.status = status;
    this.body = body;
  }
}

export class OutputError extends Error {
  public readonly rawText: string;
  public readonly attempts: number;

  public constructor(message: string, rawText: string, attempts: number) {
    super(message);
    this.name = "OutputError";
    this.rawText = rawText;
    this.attempts = attempts;
  }
}

function rec(value: unknown): value is Dict {
  return typeof value === "object" && value !== null;
}

function text(value: unknown): string | null {
  if (!rec(value)) return null;
  if (typeof value.output_text === "string") return value.output_text;
  if (!Array.isArray(value.output)) return null;

  for (const item of value.output) {
    if (!rec(item) || !Array.isArray(item.content)) continue;
    for (const part of item.content) {
      if (rec(part) && typeof part.text === "string") return part.text;
    }
  }
  return null;
}

function pause(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function safeName(name: string): string {
  const clean = name.replace(/[^a-zA-Z0-9_-]/gu, "_").slice(0, 64);
  return clean || "DynamicSchema";
}

function asShape<T extends object>(def: ShapeDef<T>, name = "DynamicSchema"): Shape<T> {
  if (rec(def) && typeof def.name === "string" && rec(def.schema)) {
    return def as unknown as Shape<T>;
  }
  return { name: safeName(name), schema: def as Schema };
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function responseInput(value: unknown): string | unknown[] {
  if (typeof value === "string" || Array.isArray(value)) return value;

  try {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) throw new TypeError();
    return encoded;
  } catch {
    throw new TypeError(
      "OpenAI input must be a string, an array of input items, or JSON-serialisable",
    );
  }
}

export class OpenAISchema<T extends object> {
  private readonly apiKey: string;
  private readonly fetcher: Fetch;
  private readonly base: string;
  private readonly headers: Readonly<Record<string, string>>;
  private readonly managedConversation: boolean;
  private current: Shape<object>;
  private conversationId: string | undefined;
  private tools: Record<string, Tool> = {};
  private tail: Promise<void> = Promise.resolve();
  private busyCount = 0;
  private pendingCount = 0;

  public constructor(
    apiKey: string,
    schema: ShapeDef<T>,
    conversationId?: string,
    init: Init = {},
  ) {
    this.apiKey = apiKey;
    this.fetcher = init.fetch ?? globalThis.fetch.bind(globalThis);
    this.base = (init.base ?? "https://api.openai.com/v1").replace(/\/+$/u, "");
    this.headers = init.headers ?? {};
    this.managedConversation = init.conversation ?? true;
    this.conversationId = conversationId;
    this.current = asShape(schema, init.name) as Shape<object>;
  }

  public get key(): string {
    return this.apiKey;
  }

  public get id(): string | undefined {
    return this.conversationId;
  }

  public get isBusy(): boolean {
    return this.busyCount > 0;
  }

  public get queued(): number {
    return this.pendingCount;
  }

  public get registeredTools(): Readonly<Record<string, Tool>> {
    return this.tools;
  }

  public async updateSchema<U extends object>(
    schema: ShapeDef<U>,
    name = "DynamicSchema",
  ): Promise<OpenAISchema<U>> {
    return this.push(async () => {
      this.current = asShape(schema, name) as Shape<object>;
      return this as unknown as OpenAISchema<U>;
    });
  }

  public async registerTool(tool: Tool): Promise<void> {
    await this.push(async () => {
      this.tools = { ...this.tools, [tool.name]: tool };
    });
  }

  public async send(input: unknown, opts: Send): Promise<T> {
    return this.push(() => this.call(this.current as Shape<T>, input, opts));
  }

  public async run<U extends object>(
    schema: ShapeDef<U>,
    input: unknown,
    opts: Send,
    name = "DynamicSchema",
  ): Promise<U> {
    return this.push(async () => {
      const next = asShape(schema, name);
      this.current = next as Shape<object>;
      return this.call(next, input, opts);
    });
  }

  private push<R>(fn: () => Promise<R>): Promise<R> {
    this.pendingCount += 1;

    const run = async (): Promise<R> => {
      this.busyCount += 1;
      try {
        return await fn();
      } finally {
        this.busyCount -= 1;
        this.pendingCount -= 1;
      }
    };

    const start = this.tail.catch(() => undefined);
    const out = start.then(run);
    this.tail = out.then(() => undefined, () => undefined);
    return out;
  }

  private headersFor(): HeadersInit {
    return {
      authorization: `Bearer ${this.apiKey}`,
      "content-type": "application/json",
      ...this.headers,
    };
  }

  private async initConversation(): Promise<void> {
    if (!this.managedConversation || this.conversationId) return;

    const response = await this.fetcher(`${this.base}/conversations`, {
      method: "POST",
      headers: this.headersFor(),
      body: "{}",
    });

    if (!response.ok) throw new OpenAIError(response.status, await response.text());
    const value: unknown = await response.json();
    if (!rec(value) || typeof value.id !== "string" || !value.id) {
      throw new OutputError("OpenAI did not return a conversation id", "", 1);
    }
    this.conversationId = value.id;
  }

  private async call<U extends object>(shape: Shape<U>, input: unknown, opts: Send): Promise<U> {
    await this.initConversation();

    const retries = Math.max(0, Math.floor(opts.retries ?? 0));
    const attempts = retries + 1;
    const delay = Math.max(0, Math.floor(opts.retryDelayMs ?? 450));
    let currentInput = input;
    let raw = "";
    let last = "Structured output was unavailable";

    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      const existingText = rec(opts.body.text) ? opts.body.text : {};
      const body: Dict = {
        ...opts.body,
        input: responseInput(currentInput),
        text: {
          ...existingText,
          format: {
            type: "json_schema",
            name: safeName(shape.name),
            ...(shape.description === undefined ? {} : { description: shape.description }),
            strict: shape.strict ?? true,
            schema: shape.schema,
          },
        },
      };

      if (this.managedConversation && this.conversationId) {
        body.conversation = { id: this.conversationId };
      }

      const response = await this.fetcher(`${this.base}/responses`, {
        method: "POST",
        headers: this.headersFor(),
        body: JSON.stringify(body),
        ...(opts.signal === undefined ? {} : { signal: opts.signal }),
      });

      if (!response.ok) throw new OpenAIError(response.status, await response.text());
      const value: unknown = await response.json();
      raw = text(value) ?? "";

      try {
        if (!raw) throw new Error("OpenAI returned no output text");
        const parsed: unknown = JSON.parse(raw);
        return shape.parse ? shape.parse(parsed) : parsed as U;
      } catch (error: unknown) {
        last = errorText(error);
        if (attempt >= attempts) break;

        if (opts.onRetry) {
          const next = await opts.onRetry({
            attempt,
            maxAttempts: attempts,
            rawText: raw,
            error: last,
            input: currentInput,
          });
          if (next !== undefined) currentInput = next;
        }
        await pause(delay + (attempt - 1) * 250);
      }
    }

    throw new OutputError(
      `Could not parse the expected output after ${attempts} attempt${attempts === 1 ? "" : "s"}: ${last}`,
      raw,
      attempts,
    );
  }
}

export function openaiSchema<T extends object>(
  apiKey: string,
  schema: ShapeDef<T>,
  conversationId?: string,
  init?: Init,
): OpenAISchema<T> {
  return new OpenAISchema(apiKey, schema, conversationId, init);
}

export function shape<T extends object>(
  name: string,
  schema: Schema,
  parse?: Parse<T>,
): Shape<T> {
  return {
    name: safeName(name),
    schema,
    ...(parse === undefined ? {} : { parse }),
  };
}

export function object(
  properties: Record<string, Schema>,
  required: readonly string[] = Object.keys(properties),
): Schema {
  return {
    type: "object",
    additionalProperties: false,
    properties,
    required: [...required],
  };
}

export function string(values?: readonly string[]): Schema {
  return values ? { type: "string", enum: [...values] } : { type: "string" };
}

export function integer(): Schema {
  return { type: "integer" };
}

export function number(): Schema {
  return { type: "number" };
}

export function boolean(): Schema {
  return { type: "boolean" };
}

export function array(items: Schema, minItems = 0, maxItems?: number): Schema {
  return {
    type: "array",
    items,
    minItems,
    ...(maxItems === undefined ? {} : { maxItems }),
  };
}

export function nullable(item: Schema): Schema {
  return { anyOf: [item, { type: "null" }] };
}
