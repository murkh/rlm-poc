# rlm-poc

A small, readable implementation of **Recursive Language Models (RLMs)**, built to learn how they work.

The core is about 200 lines in [rlm.py](rlm.py). It uses the `openai` package and no agent framework, so you can follow the whole loop in one sitting.

References: [paper (Zhang & Khattab, arXiv 2512.24601)](https://arxiv.org/abs/2512.24601) · [blog post](https://alexzhang13.github.io/blog/2025/rlm/) · [official repo](https://github.com/alexzhang13/rlm)

## The idea

A normal LLM call puts the whole input into the prompt. Very long inputs then exceed the context window, or the model's quality drops as the context grows ("context rot").

An RLM keeps the input out of the prompt:

1. The long input is stored as a variable, `context`, inside a Python REPL.
2. The **root LM** gets only the query, the size of `context`, and its first 500 characters.
3. The root LM writes code to explore `context`: it can check its length, slice it, search it with regex, or split it into chunks.
4. From that code it can call **sub-LMs** on pieces of the input: `llm_query`, `llm_query_batched`, or a recursive `rlm_query`.
5. It collects results in REPL variables and ends with `FINAL(answer)` or `FINAL_VAR(var_name)`.

The model decides how to break the problem down, so the input can be much larger than any single model's context window.

## How the loop works

```
completion(query, context)
│
├─ REPL namespace = { context, llm_query, llm_query_batched, rlm_query }
├─ messages = [system prompt, "Query: ... context is N chars, first 500: ..."]
│
└─ repeat up to max_iterations:
     reply = root_model(messages)
     run every ```repl block in reply  → capture stdout (truncated)
     FINAL_VAR(x) outside code?  → return ns[x]
     FINAL(...)   outside code?  → return text
     else append "REPL output: ..." to messages and continue
   on limit: ask for one last FINAL(...)
```

Where to find each part in [rlm.py](rlm.py):

| Concept | Code |
|---|---|
| Instructions given to the root LM | `SYSTEM_PROMPT` |
| Root loop | `RLM.completion` |
| Running model-written code, capturing and truncating its output | `RLM._exec` |
| Single sub-LM call (`sub_model`, no REPL) | `RLM.llm_query` |
| Parallel sub-LM calls (8 threads) | `RLM.llm_query_batched` |
| Recursion: a child `RLM` with its own REPL at `depth + 1` | `RLM.rlm_query` |
| Depth limit: at `max_depth`, `rlm_query` falls back to `llm_query` | `RLM.rlm_query` |
| Parsing `FINAL` and `FINAL_VAR` (ignored inside code blocks) | `FINAL_RE`, `FINAL_VAR_RE` |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python test_rlm.py      # offline check with a scripted fake client, no API calls
```

Configure an OpenAI-compatible endpoint. A `.env` file works because `python-dotenv` is loaded:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...            # optional: vLLM, a gateway, etc.
```

The default models (`zai.glm-5` for the root, `nvidia.nemotron-nano-3-30b` for sub-calls) assume a gateway that serves those names. Pass `--model` and `--sub-model` to use others.

## Run it

```bash
# Needle in a haystack: about 1M chars of random words with one hidden "magic number"
uv run python main.py

# Different models
uv run python main.py --model gpt-5-mini --sub-model gpt-5-nano

# Allow deeper recursion (rlm_query can spawn children of children)
uv run python main.py --max-depth 2

# Your own document
uv run python main.py --context-file doc.txt --query "Summarize the key risks"

# Show the full root LM replies, the code it ran, and the REPL output
uv run python main.py --log-level DEBUG
```

From Python:

```python
from rlm import RLM

rlm = RLM(model="gpt-5", sub_model="gpt-5-mini")
print(rlm.completion("What changed between v1 and v2?", huge_text))
print(rlm.sub_calls)
```

## What to watch

Run with `--log-level DEBUG` and follow a single trace. Things worth noticing:

- **The first iteration**: the root LM usually checks the structure first (`len(context)`, `context[:1000]`, `context.split("\n")`) before calling any sub-LM.
- **Chunking strategy**: what chunk size does it pick, and does it use `llm_query_batched` to run chunks in parallel?
- **Aggregation**: sub-LM answers become Python variables, and the root combines them with code instead of rereading the text.
- **Errors**: exceptions are returned as tracebacks, and the model usually fixes its own code on the next turn.
- **Cost**: compare `sub-calls` and the token totals in the logs with what a single long-context call would cost.
- **Indentation**: log lines are indented by recursion depth (`[d=0]`, `  [d=1]`), so you can see when `rlm_query` starts a child.

Experiments to try:

- Set `--max-depth 0` (no recursion) or `--max-depth 2` and compare the answers and the number of calls.
- Use a weaker `--sub-model` and see whether the root LM compensates.
- Change the `haystack()` size in [main.py](main.py) beyond the root model's context window. The RLM should still work, while a direct call would not.
- Edit `SYSTEM_PROMPT` and see how the model's strategy changes.

## Files

| File | Purpose |
|---|---|
| [rlm.py](rlm.py) | The RLM: REPL, sub-calls, recursion, and the root loop |
| [main.py](main.py) | CLI: needle-in-a-haystack demo or your own `--context-file` |
| [test_rlm.py](test_rlm.py) | Offline test with a fake client: the loop order, feeding output back to the root LM, the root never seeing the raw context, and `FINAL` being ignored inside code |

## Limitations

This is a proof of concept for learning, not a production system.

- ⚠️ **No sandbox.** Model-written code runs in-process with `exec`. Use it only with trusted models and inputs. A safe version would run the REPL in a subprocess or a container.
- Calls are synchronous apart from the batched helper, and there is no retry or backoff.
- There is no token or cost budget, only `max_iterations` and `max_depth`.
- REPL output is cut at `max_output_chars` (20k by default), so the model has to print summaries rather than raw text.
