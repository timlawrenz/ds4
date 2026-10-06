#!/usr/bin/env python3
"""kv_reuse_test.py -- measure disk-KV prefix reuse under eviction pressure.

WHY THIS EXISTS
    ds4's disk KV cache only earns its footprint if a checkpoint survives until the
    request that would reuse it comes back. Waiting for real cron jobs to recur takes
    hours, so this test makes the mechanism observable in minutes: it sends an
    identical long prompt twice and reports exactly what the server reused.

    For PR #1170 ("keep the disk KV checkpoint the next request loads when storing an
    evict snapshot"), the failure is that a checkpoint gets evicted while making room
    for a store -- i.e. the server deletes the very state it is about to need. Nothing
    about that requires a full 64 GiB cache, so run the server with a deliberately
    SMALL --kv-disk-space-mb and eviction fires on almost every store.

HOW TO READ THE OUTPUT
    cached_tokens on pass 3 is the answer. Cold == 0. A working cache reuses a large
    leading prefix. cache_write_tokens shows what it had to re-write to disk instead.

USAGE
    python3 kv_reuse_test.py --base-url http://127.0.0.1:8001/v1 \
        --prompt-tokens 8192 --churn 8 --json /tmp/kv-reuse.json --label control

    stdlib only, on purpose -- the same constraint the repo's own benches keep.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "promessi_sposi.txt")


def load_corpus(path, target_tokens):
    """Approximate a prompt of ~target_tokens by repeating the corpus.
    English runs ~4 chars/token; the response's prompt_tokens is the real number."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = "The quick brown fox jumps over the lazy dog. " * 200
    if not text.strip():
        text = "placeholder corpus text. " * 100
    want = max(400, target_tokens * 4)
    if len(text) >= want:
        return text[:want]
    return (text * (want // len(text) + 1))[:want]


def post(base_url, payload, timeout=3600):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body, time.time() - started


def usage_field(usage, *paths):
    """Pull a number out of the usage block, tolerating several spellings."""
    for path in paths:
        node = usage
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, (int, float)):
            return int(node)
    return None


def one_pass(base_url, model, system_text, user_text, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    body, wall = post(base_url, payload)
    usage = body.get("usage") or {}
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": usage_field(usage, "prompt_tokens_details.cached_tokens", "cached_tokens"),
        "cache_write_tokens": usage_field(usage, "prompt_tokens_details.cache_write_tokens", "cache_write_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "raw_usage": usage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--churn", type=int, default=8, help="distinct prompts sent between the two identical passes")
    ap.add_argument("--churn-tokens", type=int, default=8192)
    ap.add_argument("--gen", type=int, default=1, help="tokens to generate (1 keeps prefill the dominant cost)")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--label", default="run")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    target_text = load_corpus(args.corpus, args.prompt_tokens)
    churn_text = (target_text[::-1])[: max(400, args.churn_tokens * 4)]

    print(f"[{args.label}] target ~{args.prompt_tokens} tok, churn {args.churn} x ~{args.churn_tokens} tok, gen {args.gen}")
    sys.stdout.flush()

    passes = []

    # Pass 1: cold -- establishes the checkpoint we will try to reuse.
    p1 = one_pass(args.base_url, args.model, target_text, "Reply with the single word: READY", args.gen)
    passes.append(("pass1_cold", p1))
    print(f"  pass1 cold   : {p1['wall_s']:>7.2f}s  prompt={p1['prompt_tokens']}  cached={p1['cached_tokens']}  write={p1['cache_write_tokens']}")
    sys.stdout.flush()

    # Churn -- distinct prompts push the cache; with a small budget this evicts pass1's state.
    for i in range(args.churn):
        cp = one_pass(args.base_url, args.model, churn_text + f"\n[churn {i}]",
                      "Reply with the single word: OK", args.gen)
        passes.append((f"churn{i}", cp))
        print(f"  churn {i:<3}     : {cp['wall_s']:>7.2f}s  prompt={cp['prompt_tokens']}  cached={cp['cached_tokens']}  write={cp['cache_write_tokens']}")
        sys.stdout.flush()

    # Pass 3: identical to pass 1 -- did the checkpoint survive?
    p3 = one_pass(args.base_url, args.model, target_text, "Reply with the single word: READY", args.gen)
    passes.append(("pass3_reuse", p3))
    print(f"  pass3 reuse  : {p3['wall_s']:>7.2f}s  prompt={p3['prompt_tokens']}  cached={p3['cached_tokens']}  write={p3['cache_write_tokens']}")
    sys.stdout.flush()

    pt = p1.get("prompt_tokens") or 0
    reuse = p3.get("cached_tokens") or 0
    ratio = (reuse / pt) if pt else 0.0
    speedup = (p1["wall_s"] / p3["wall_s"]) if p3["wall_s"] else 0.0

    verdict = "REUSED" if reuse > 0 else "NOT REUSED (cold)"
    print()
    print(f"  [{args.label}] pass3 re-prefill verdict: {verdict}")
    print(f"  [{args.label}] cached {reuse}/{pt} tokens ({ratio:.1%})  wall {p1['wall_s']}s -> {p3['wall_s']}s ({speedup:.2f}x)")

    result = {
        "label": args.label,
        "base_url": args.base_url,
        "config": vars(args),
        "passes": {name: data for name, data in passes},
        "verdict": verdict,
        "pass1_wall_s": p1["wall_s"],
        "pass3_wall_s": p3["wall_s"],
        "pass3_cached_tokens": reuse,
        "pass3_prompt_tokens": pt,
        "reuse_ratio": round(ratio, 4),
        "wall_speedup": round(speedup, 3),
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"  -> {args.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read()[:400]!r}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
