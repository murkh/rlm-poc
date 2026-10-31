"""Recursive Language Models (Zhang & Khattab, arXiv 2512.24601) — minimal POC.

The root LM never sees the context directly. It lives as `context` in a Python REPL;
the LM writes ```repl``` code blocks to inspect/partition it and calls sub-LMs on pieces.
"""

import contextlib
import io
import logging
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

log = logging.getLogger("rlm")


def _preview(s, n=200):
    s = str(s)
    return repr(s[:n] + (f"...[+{len(s) - n} chars]" if len(s) > n else ""))

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
    def __init__(
        self,
        model="zai.glm-5",
        sub_model="nvidia.nemotron-nano-3-30b",
        max_depth=1,
        depth=0,
        max_iterations=20,
        max_output_chars=20_000,
        client=None,
        verbose=True,
    ):
        self.model, self.sub_model = model, sub_model
        self.max_depth, self.depth = max_depth, depth
        self.max_iterations, self.max_output_chars = max_iterations, max_output_chars
        self.client = client or OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL
        self.verbose = verbose
        self.sub_calls = 0

    def _chat(self, model, messages):
        self._log(f"-> {model}: {len(messages)} msg(s), {sum(len(m['content']) for m in messages):,} chars")
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(model=model, messages=messages)
        content = resp.choices[0].message.content or ""
        u = resp.usage
        tokens = f", tokens in/out {u.prompt_tokens}/{u.completion_tokens}" if u else ""
        self._log(f"<- {model}: {len(content):,} chars in {time.perf_counter() - t0:.1f}s{tokens}")
        return content

    def llm_query(self, prompt):
        self.sub_calls += 1
        n = self.sub_calls
        self._log(f"llm_query #{n}: prompt {len(prompt):,} chars")
        self._log(f"llm_query #{n} prompt: {_preview(prompt, 500)}", logging.DEBUG)
        answer = self._chat(self.sub_model, [{"role": "user", "content": prompt}])
        self._log(f"llm_query #{n} answer: {_preview(answer)}")
        return answer

    def llm_query_batched(self, prompts):
        self._log(f"llm_query_batched: {len(prompts)} prompt(s) in parallel")
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(self.llm_query, prompts))

    def rlm_query(self, prompt, context=""):
        if self.depth >= self.max_depth:
            self._log(f"rlm_query at max depth {self.max_depth}: falling back to llm_query")
            return self.llm_query(prompt + ("\n\n" + str(context) if context else ""))
        self.sub_calls += 1
        self._log(f"rlm_query: spawning child RLM at depth {self.depth + 1}, prompt {_preview(prompt)}")
        child = RLM(
            self.model,
            self.sub_model,
            self.max_depth,
            self.depth + 1,
            self.max_iterations,
            self.max_output_chars,
            self.client,
            self.verbose,
        )
        answer = child.completion(prompt, context)
        self._log(f"rlm_query: child returned {_preview(answer)}")
        return answer

    def _exec(self, code, ns):
        # ponytail: in-process exec, no sandbox — trusted models/inputs only; upgrade = subprocess/Docker REPL.
        self._log(f"exec code:\n{code}", logging.DEBUG)
        t0 = time.perf_counter()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                exec(code, ns)
            except Exception as e:
                self._log(f"exec raised {type(e).__name__}: {e}", logging.WARNING)
                print(traceback.format_exc(limit=3))
        out = buf.getvalue()
        self._log(f"exec done in {time.perf_counter() - t0:.1f}s, {len(out):,} chars output")
        self._log(f"exec output:\n{out}", logging.DEBUG)
        if len(out) > self.max_output_chars:
            out = (
                out[: self.max_output_chars]
                + f"\n...[truncated {len(out) - self.max_output_chars} chars]"
            )
        return out or "(no output)"

    def _log(self, msg, level=logging.INFO):
        # stdout is redirected during exec, so logging (stderr) keeps logs out of REPL output.
        if self.verbose:
            log.log(level, f"{'  ' * self.depth}[d={self.depth}] {msg}")

    def completion(self, query, context):
        self._log(f"completion start: query {_preview(query)}, context {type(context).__name__} of {len(context):,} chars")
        ns = {
            "context": context,
            "llm_query": self.llm_query,
            "llm_query_batched": self.llm_query_batched,
            "rlm_query": self.rlm_query,
        }
        preview = str(context)[:500]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\n\n`context` is a {type(context).__name__} "
                f"of {len(context)} chars. First 500 chars:\n{preview}",
            },
        ]
        for i in range(self.max_iterations):
            self._log(f"=== iter {i + 1}/{self.max_iterations} ===")
            reply = self._chat(self.model, messages)
            messages.append({"role": "assistant", "content": reply})
            self._log(f"root LM reply:\n{reply}", logging.DEBUG)
            blocks = CODE_RE.findall(reply)
            self._log(
                f"iter {i + 1}: {len(blocks)} code block(s), sub-calls so far {self.sub_calls}"
            )
            outputs = [self._exec(code, ns) for code in blocks]

            text = CODE_RE.sub("", reply)  # FINAL is only valid outside code blocks
            if m := FINAL_VAR_RE.search(text):
                if m.group(1) in ns:
                    self._log(f"FINAL_VAR({m.group(1)}) -> {_preview(ns[m.group(1)])}")
                    return str(ns[m.group(1)])
                self._log(f"FINAL_VAR({m.group(1)}) but variable missing; continuing", logging.WARNING)
                outputs.append(f"Error: variable {m.group(1)!r} not found in REPL.")
            elif m := FINAL_RE.search(text):
                self._log(f"FINAL -> {_preview(m.group(1).strip())}")
                return m.group(1).strip()

            feedback = (
                "\n\n".join(f"REPL output:\n{o}" for o in outputs)
                if outputs
                else "No code was run. Write ```repl``` code or give FINAL(...)."
            )
            messages.append({"role": "user", "content": feedback})

        self._log(f"iteration limit {self.max_iterations} reached; forcing final answer", logging.WARNING)
        messages.append(
            {
                "role": "user",
                "content": "Iteration limit reached. Give your best final answer now as FINAL(...).",
            }
        )
        reply = self._chat(self.model, messages)
        m = FINAL_RE.search(reply)
        return m.group(1).strip() if m else reply.strip()
