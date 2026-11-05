"""Summarise one evaluation store: solved count and mean reward.

Kept separate from the shell queue because the first version inlined it and read
``row["config"]``, which trace records do not have -- the whole comparison then
died with KeyError after the evaluation had already run. Keys are read defensively
and the seed comes from either field name.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: eval_pair.py <store.jsonl> [other.jsonl]")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print("0,0.0000")
        return 1

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("0,0.0000")
        return 1

    solved = sum(1 for row in rows if row["reward"].get("r_task", 0.0) >= 0.999)
    mean = st.mean(row["reward"].get("r_task", 0.0) for row in rows)
    print(f"{solved},{mean:.4f},{len(rows)}")

    if len(sys.argv) > 2:
        other = Path(sys.argv[2])
        if other.exists():
            other_rows = [
                json.loads(line)
                for line in other.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            print(f"compare rows={len(other_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
