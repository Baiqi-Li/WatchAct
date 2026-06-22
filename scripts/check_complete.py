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
    "scoring":    "source_action_execution_path",
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
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from run_action_plan_scoring import infer_dataset_name
        output_path = args.output_dir / f"scores_{infer_dataset_name(args.input)}.json"
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

    total = out.get("total")
    results = out.get("results")
    if not isinstance(total, int) or not isinstance(results, list):
        fail("output missing total/results")
    if len(results) != total:
        fail(f"len(results)={len(results)} != total={total}")

    input_results = inp.get("results")
    if not isinstance(input_results, list) or len(input_results) != total:
        n = len(input_results) if isinstance(input_results, list) else "N/A"
        fail(f"input/output size mismatch (input={n}, output total={total})")

    recorded = out.get(SOURCE_FIELD[args.stage])
    if not isinstance(recorded, str) or not paths_match(recorded, args.input):
        fail(f"source path mismatch (recorded={recorded!r})")

    if args.stage == "translator" and any(r is None for r in results):
        fail("translator output has null entries (checkpoint, not finished)")

    sys.exit(0)


if __name__ == "__main__":
    main()
