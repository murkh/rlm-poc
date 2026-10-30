import argparse
import random

from rlm import RLM


def haystack(n_lines=20_000, seed=0):
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "cloud", "paper", "lamp", "orbit", "quiet", "green"]
    lines = [" ".join(rng.choices(words, k=8)) for _ in range(n_lines)]
    magic = str(rng.randint(1_000_000, 9_999_999))
    lines.insert(rng.randrange(n_lines), f"The magic number is {magic}.")
    return "\n".join(lines), magic


def main():
    p = argparse.ArgumentParser(description="Recursive Language Model POC")
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--sub-model", default="gpt-5-mini")
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--context-file", help="run on your own context instead of the needle demo")
    p.add_argument("--query")
    args = p.parse_args()

    rlm = RLM(args.model, args.sub_model, max_depth=args.max_depth)
    if args.context_file:
        if not args.query:
            p.error("--query is required with --context-file")
        with open(args.context_file, encoding="utf-8") as f:
            print(rlm.completion(args.query, f.read()))
        return

    context, magic = haystack()
    print(f"Context: {len(context):,} chars, expected magic number {magic}")
    answer = rlm.completion("What is the magic number hidden in the context? Answer with just the number.", context)
    print(f"Answer: {answer} | sub-calls: {rlm.sub_calls} | correct: {magic in answer}")


if __name__ == "__main__":
    main()
