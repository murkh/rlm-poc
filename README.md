# rlm-poc

Minimal POC of **Recursive Language Models** ([paper](https://arxiv.org/abs/2512.24601), [blog](https://alexzhang13.github.io/blog/2025/rlm/), [official repo](https://github.com/alexzhang13/rlm)) on the `openai` package.

The root LM never sees the full context. The context is stored as the `context` variable in a Python REPL. The LM writes ```` ```repl ```` code to inspect and split it, and calls `llm_query` / `llm_query_batched` / `rlm_query` to run sub-LMs on pieces. It ends with `FINAL(answer)` or `FINAL_VAR(var)`.

```bash
uv sync
uv run python test_rlm.py                      # offline check, no API calls
export OPENAI_API_KEY=...                      # OPENAI_BASE_URL also works (vLLM etc.)
uv run python main.py                          # needle-in-haystack demo (~1M chars)
uv run python main.py --model gpt-5-mini --sub-model gpt-5-nano
uv run python main.py --context-file doc.txt --query "Summarize the key risks"
```

```python
from rlm import RLM
print(RLM("gpt-5", "gpt-5-mini").completion(query, huge_text))
```

⚠️ Model-written code runs in-process via `exec` with no sandbox. Use it only with trusted models and inputs.
