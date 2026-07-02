#!/usr/bin/env python3
"""Check whether a stage output JSON is complete relative to its input.

Exit codes:
  0 -> complete, safe to skip
  1 -> missing / incomplete / mismatched, must (re)run
  2 -> usage error
"""
import argparse
import json
import sys
from pathlib import Path

SOURCE_FIELD = {
    "translator": "source_results_path",
    "simulation": "source_translator_results_path",
    # "scoring" only registers a valid --stage choice; the scoring output has no
    # source-path field, so this value is never read (the scoring branch returns
    # before SOURCE_FIELD is consulted).
    "scoring":    None,
}


def paths_match(recorded: str, actual: Path) -> bool:
    a, b = Path(recorded), actual
    if a.as_posix() == b.as_posix():
        return True
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return False


def fail(msg: str) -> None:
    print(f"[check_complete] incomplete: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(SOURCE_FIELD))
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[check_complete] input missing: {args.input}", file=sys.stderr)
        sys.exit(2)

    if args.stage == "scoring":
        if args.output_dir is None:
            print("[check_complete] scoring requires --output-dir", file=sys.stderr)
            sys.exit(2)
        # Must match the score filename written by run_action_plan_scoring.py.
        output_path = args.output_dir / f"scores_{args.input.stem}.json"
    else:
        if args.output is None:
            print(f"[check_complete] {args.stage} requires --output", file=sys.stderr)
            sys.exit(2)
        output_path = args.output

    if not output_path.is_file():
        sys.exit(1)

    try:
        out = json.loads(output_path.read_text(encoding="utf-8"))
        inp = json.loads(args.input.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"json load failed: {e}")

    results = out.get("results")
    input_results = inp.get("results")
    if not isinstance(results, list) or not isinstance(input_results, list):
        fail("output/input missing results list")

    # Scoring output has no `total` or source-path field; it records one result
    # row per input row plus evaluated_count/error_count.
    if args.stage == "scoring":
        if not isinstance(out.get("evaluated_count"), (int, float)):
            fail("scoring output missing evaluated_count")
        if len(results) != len(input_results):
            fail(f"scored {len(results)} rows != input {len(input_results)}")
        sys.exit(0)

    total = out.get("total")
    if not isinstance(total, int):
        fail("output missing total")
    if len(results) != total or len(input_results) != total:
        fail(f"size mismatch (input={len(input_results)}, output={len(results)}, total={total})")

    recorded = out.get(SOURCE_FIELD[args.stage])
    if not isinstance(recorded, str) or not paths_match(recorded, args.input):
        fail(f"source path mismatch (recorded={recorded!r})")

    if args.stage == "translator" and any(r is None for r in results):
        fail("translator output has null entries (checkpoint, not finished)")

    sys.exit(0)


if __name__ == "__main__":
    main()
