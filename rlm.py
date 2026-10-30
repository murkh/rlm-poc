"""Recursive Language Models (Zhang & Khattab, arXiv 2512.24601) — minimal POC.

The root LM never sees the context directly. It lives as `context` in a Python REPL;
the LM writes ```repl``` code blocks to inspect/partition it and calls sub-LMs on pieces.
"""

import contextlib
import io
import re
import traceback
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable holding the context for the query. It may be huge: do NOT print it whole; inspect it in slices (len, head, split, regex, etc.).
2. `llm_query(prompt)`: a single sub-LLM completion call (no REPL). Use for extraction, summarization or Q&A over a chunk. The sub-LLM can handle around 500K chars.
3. `llm_query_batched(prompts)`: runs several `llm_query` calls concurrently, returns list[str] in input order. Prefer it for independent chunks.
4. `rlm_query(prompt, context="")`: spawns a recursive RLM sub-call with its own REPL, for subtasks needing multi-step reasoning. Falls back to `llm_query` at max depth.
5. `print()` to see results. Output is truncated, so print summaries, not raw dumps.

To run code, write it in a block fenced with ```repl and ```. Variables persist across blocks. You will see the output in the next turn.

Strategy: first peek at the context (size, structure), then chunk it, query sub-LLMs over chunks, store results in variables, and aggregate.

When done, give the final answer OUTSIDE any code block, on its own line, as either:
- FINAL(your answer here)
- FINAL_VAR(variable_name)   # returns a REPL variable as the answer
Do not give a final answer until you have actually examined the context."""

CODE_RE = re.compile(r"```repl\s*\n(.*?)```", re.S)
FINAL_VAR_RE = re.compile(r"^\s*FINAL_VAR\(\s*['\"]?(\w+)['\"]?\s*\)", re.M)
FINAL_RE = re.compile(r"^\s*FINAL\((.*)\)\s*$", re.M | re.S)


class RLM:
    def __init__(self, model="gpt-5", sub_model="gpt-5-mini", max_depth=1, depth=0,
                 max_iterations=20, max_output_chars=20_000, client=None, verbose=True):
        self.model, self.sub_model = model, sub_model
        self.max_depth, self.depth = max_depth, depth
        self.max_iterations, self.max_output_chars = max_iterations, max_output_chars
        self.client = client or OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL
        self.verbose = verbose
        self.sub_calls = 0

    def _chat(self, model, messages):
        resp = self.client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    def llm_query(self, prompt):
        self.sub_calls += 1
        return self._chat(self.sub_model, [{"role": "user", "content": prompt}])

    def llm_query_batched(self, prompts):
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(self.llm_query, prompts))

    def rlm_query(self, prompt, context=""):
        if self.depth >= self.max_depth:
            return self.llm_query(prompt + ("\n\n" + str(context) if context else ""))
        self.sub_calls += 1
        child = RLM(self.model, self.sub_model, self.max_depth, self.depth + 1,
                    self.max_iterations, self.max_output_chars, self.client, self.verbose)
        return child.completion(prompt, context)

    def _exec(self, code, ns):
        # ponytail: in-process exec, no sandbox — trusted models/inputs only; upgrade = subprocess/Docker REPL.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                exec(code, ns)
            except Exception:
                print(traceback.format_exc(limit=3))
        out = buf.getvalue()
        if len(out) > self.max_output_chars:
            out = out[:self.max_output_chars] + f"\n...[truncated {len(out) - self.max_output_chars} chars]"
        return out or "(no output)"

    def _log(self, msg):
        if self.verbose:
            print(f"{'  ' * self.depth}[rlm d={self.depth}] {msg}")

    def completion(self, query, context):
        ns = {"context": context, "llm_query": self.llm_query,
              "llm_query_batched": self.llm_query_batched, "rlm_query": self.rlm_query}
        preview = str(context)[:500]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\n`context` is a {type(context).__name__} "
                                        f"of {len(context)} chars. First 500 chars:\n{preview}"},
        ]
        for i in range(self.max_iterations):
            reply = self._chat(self.model, messages)
            messages.append({"role": "assistant", "content": reply})
            blocks = CODE_RE.findall(reply)
            self._log(f"iter {i + 1}: {len(blocks)} code block(s), sub-calls so far {self.sub_calls}")
            outputs = [self._exec(code, ns) for code in blocks]

            text = CODE_RE.sub("", reply)  # FINAL is only valid outside code blocks
            if m := FINAL_VAR_RE.search(text):
                if m.group(1) in ns:
                    return str(ns[m.group(1)])
                outputs.append(f"Error: variable {m.group(1)!r} not found in REPL.")
            elif m := FINAL_RE.search(text):
                return m.group(1).strip()

            feedback = "\n\n".join(f"REPL output:\n{o}" for o in outputs) if outputs else \
                "No code was run. Write ```repl``` code or give FINAL(...)."
            messages.append({"role": "user", "content": feedback})

        messages.append({"role": "user", "content": "Iteration limit reached. Give your best final answer now as FINAL(...)."})
        reply = self._chat(self.model, messages)
        m = FINAL_RE.search(reply)
        return m.group(1).strip() if m else reply.strip()
