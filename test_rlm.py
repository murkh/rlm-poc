"""Offline check of the RLM loop with a scripted fake OpenAI client. Run: uv run python test_rlm.py"""
from types import SimpleNamespace

from rlm import RLM


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages):
        self.calls.append((model, list(messages)))
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


client = FakeClient([
    "Peek first.\n```repl\nprint(len(context))\nx = llm_query('find number in: ' + context)\n```",
    "sub-LM reply: 42",                       # consumed by llm_query (sub model)
    "Done.\n```repl\nans = x.split(': ')[1]\n```\nFINAL_VAR(ans)",
])
rlm = RLM("root", "sub", client=client, verbose=False)
assert rlm.completion("What number?", "abc 42 xyz") == "42"
assert rlm.sub_calls == 1
assert [c[0] for c in client.calls] == ["root", "sub", "root"]
assert "REPL output:\n10\n" in client.calls[2][1][-1]["content"]  # exec stdout fed back to root
assert "abc 42 xyz" not in client.calls[0][1][1]["content"].split("First 500 chars:")[0]  # root gets metadata, not context

# FINAL(...) inside a code block must not end the loop; errors feed back as tracebacks.
client = FakeClient(["```repl\n1/0\n# FINAL(wrong)\n```", "FINAL(ok)"])
assert RLM("root", "sub", client=client, verbose=False).completion("q", "c") == "ok"
assert "ZeroDivisionError" in client.calls[1][1][-1]["content"]
print("ok")
