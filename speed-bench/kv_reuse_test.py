#!/usr/bin/env python3
"""kv_reuse_test.py -- measure disk-KV prefix reuse.

TWO MODES, and the difference matters:

  conversation (default) -- models what a dev-worker tick actually does: a stable long
      system prefix, then turns that GROW. Turn 2's prompt contains turn 1's prompt plus
      the assistant's reply, so turn 1's stored state is a genuine prefix of turn 2's
      request. This is the shape in which ds4's disk cache can pay off.

  replay -- send an identical prompt twice. Kept because it is the tempting test to
      write, and it produces a FALSE NEGATIVE: ds4 stores the outgoing session's state
      (prompt + generated tokens), which can never be a prefix of an identical
      re-send. Do not draw conclusions about cache health from replay mode.

HOW TO READ IT
    cached_tokens on turn >= 2 is the answer. A working disk cache reuses the whole
    leading prefix; cache_write_tokens shows what it had to re-write instead.

USAGE
    python3 kv_reuse_test.py --base-url http://127.0.0.1:8001/v1 --prompt-tokens 8192
    python3 kv_reuse_test.py --mode replay --churn 2 ...

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


def load_corpus(path, target_tokens, offset=0):
    """Approximate a prompt of ~target_tokens by repeating the corpus, starting at
    `offset` chars in. The offset lets a caller create a DIFFERENT prefix (churn) or
    return to the SAME prefix later, which is what the eviction-pressure test needs."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = "The quick brown fox jumps over the lazy dog. " * 200
    if not text.strip():
        text = "placeholder corpus text. " * 100
    if offset:
        text = (text[offset:] + text[:offset])
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


def send(base_url, model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    body, wall = post(base_url, payload)
    usage = body.get("usage") or {}
    try:
        reply = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        reply = ""
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": usage_field(usage, "prompt_tokens_details.cached_tokens", "cached_tokens"),
        "cache_write_tokens": usage_field(usage, "prompt_tokens_details.cache_write_tokens", "cache_write_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reply": reply,
    }


def report_turn(label, r):
    print(f"  {label:<14}: {r['wall_s']:>7.2f}s  prompt={r['prompt_tokens']}  "
          f"cached={r['cached_tokens']}  write={r['cache_write_tokens']}  gen={r['completion_tokens']}")
    sys.stdout.flush()


def mode_conversation(args):
    """Stable system prefix; turns grow. The shape a dev-worker tick actually has."""
    system_text = load_corpus(args.corpus, args.prompt_tokens, args.prefix_offset)
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "Reply with the single word: READY"},
    ]
    passes = []

    t1 = send(args.base_url, args.model, msgs, max(1, args.gen))
    passes.append(("turn1_cold", t1))
    report_turn("turn1 cold", t1)

    # Grow the conversation: turn 2's prompt contains turn 1's prompt + the reply.
    msgs = msgs + [
        {"role": "assistant", "content": t1["reply"]},
        {"role": "user", "content": "Now reply with the single word: DONE"},
    ]
    t2 = send(args.base_url, args.model, msgs, max(1, args.gen))
    passes.append(("turn2_grown", t2))
    report_turn("turn2 grown", t2)

    last = t2
    for extra in range(max(0, args.turns - 2)):
        msgs = msgs + [
            {"role": "assistant", "content": last["reply"]},
            {"role": "user", "content": f"Reply with the single word: STEP{extra}"},
        ]
        tn = send(args.base_url, args.model, msgs, max(1, args.gen))
        passes.append((f"turn{extra + 3}_grown", tn))
        report_turn(f"turn{extra + 3} grown", tn)
        last = tn

    t2_pt = t2.get("prompt_tokens") or 0
    t2_cached = t2.get("cached_tokens") or 0
    ratio = (t2_cached / t2_pt) if t2_pt else 0.0
    verdict = "REUSED" if t2_cached > 0 else "NOT REUSED"
    print()
    print(f"  [{args.label}] positive-control verdict: {verdict}")
    print(f"  [{args.label}] turn2 reused {t2_cached}/{t2_pt} tokens ({ratio:.1%}); "
          f"wall {t1['wall_s']}s -> {t2['wall_s']}s")
    return {
        "label": args.label, "mode": "conversation", "base_url": args.base_url,
        "config": vars(args), "passes": {n: d for n, d in passes},
        "verdict": verdict, "turn2_cached_tokens": t2_cached, "turn2_prompt_tokens": t2_pt,
        "reuse_ratio": round(ratio, 4),
        "turn1_wall_s": t1["wall_s"], "turn2_wall_s": t2["wall_s"],
    }


def mode_replay(args):
    """Identical re-send. Known to produce false negatives -- see the module docstring."""
    target_text = load_corpus(args.corpus, args.prompt_tokens)
    churn_text = (target_text[::-1])[: max(400, args.churn_tokens * 4)]
    passes = []

    p1 = send(args.base_url, args.model, [
        {"role": "system", "content": target_text},
        {"role": "user", "content": "Reply with the single word: READY"}], max(1, args.gen))
    passes.append(("pass1_cold", p1))
    report_turn("pass1 cold", p1)

    for i in range(args.churn):
        cp = send(args.base_url, args.model, [
            {"role": "system", "content": churn_text + f"\n[churn {i}]"},
            {"role": "user", "content": "Reply with the single word: OK"}], max(1, args.gen))
        passes.append((f"churn{i}", cp))
        report_turn(f"churn {i}", cp)

    p3 = send(args.base_url, args.model, [
        {"role": "system", "content": target_text},
        {"role": "user", "content": "Reply with the single word: READY"}], max(1, args.gen))
    passes.append(("pass3_reuse", p3))
    report_turn("pass3 replay", p3)

    pt = p1.get("prompt_tokens") or 0
    reuse = p3.get("cached_tokens") or 0
    ratio = (reuse / pt) if pt else 0.0
    verdict = "REUSED" if reuse > 0 else "NOT REUSED"
    print()
    print(f"  [{args.label}] replay verdict: {verdict} ({reuse}/{pt} = {ratio:.1%})")
    return {
        "label": args.label, "mode": "replay", "base_url": args.base_url,
        "config": vars(args), "passes": {n: d for n, d in passes},
        "verdict": verdict, "pass3_cached_tokens": reuse, "pass3_prompt_tokens": pt,
        "reuse_ratio": round(ratio, 4),
        "pass1_wall_s": p1["wall_s"], "pass3_wall_s": p3["wall_s"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("conversation", "replay"), default="conversation")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--prefix-offset", type=int, default=0,
                    help="rotate the corpus by N chars to get a DIFFERENT system prefix (or 0 to return to the same one)")
    ap.add_argument("--turns", type=int, default=2, help="conversation mode: total turns")
    ap.add_argument("--churn", type=int, default=2, help="replay mode: prompts between the two identical passes")
    ap.add_argument("--churn-tokens", type=int, default=8192)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--label", default="run")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print(f"[{args.label}] mode={args.mode} target ~{args.prompt_tokens} tok, gen {args.gen}")
    sys.stdout.flush()

    result = mode_conversation(args) if args.mode == "conversation" else mode_replay(args)

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
