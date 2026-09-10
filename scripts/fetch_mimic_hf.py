#!/usr/bin/env python3
"""Stream the five tables the cohort builder reads from a public HF mirror.

CHARTEVENTS is 35.3 GB and is never materialised. Each byte range is requested,
filtered against the cohort and the 17-variable itemid map, and discarded; what
lands on disk is only the rows the builder would have kept, a few GB.

The filter reads the first five comma-separated fields only. In CHARTEVENTS those
are ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID and all five are unquoted
integers, so splitting on the comma is exact regardless of how VALUE later in the
same line is quoted. Matched lines are written whole, so pandas still sees a
well-formed CSV with the original header.

Ranges are requested sequentially with retries and a progress file, so an
interrupted 35 GB pass resumes at the last completed chunk instead of restarting.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{name}.csv"
CHUNK = 512 * 1024 * 1024

# Which zero-based field carries the itemid, and which carries the key the cohort
# is matched on. LABEVENTS has no icustay_id, so it is matched on hadm_id -- the
# same substitution the builder makes in collect_events.
LAYOUT = {
    "CHARTEVENTS": {"itemid": 4, "key": 3, "keyname": "icustay_id"},
    "LABEVENTS": {"itemid": 3, "key": 2, "keyname": "hadm_id"},
}


def remote_size(repo: str, name: str) -> int:
    url = BASE.format(repo=repo, name=name)
    out = subprocess.run(
        ["curl", "-sIL", "--max-time", "120", url], capture_output=True, text=True, check=True
    ).stdout
    for line in reversed(out.splitlines()):
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"no content-length for {name}")


def fetch_range(repo: str, name: str, start: int, end: int, attempts: int = 5) -> bytes:
    url = BASE.format(repo=repo, name=name)
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            ["curl", "-sL", "--max-time", "1800", "-r", f"{start}-{end}", url],
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        time.sleep(5 * attempt)
    raise RuntimeError(f"{name}: range {start}-{end} failed after {attempts} attempts")


def stream_table(repo: str, name: str, dest: Path, itemids: set[bytes], keys: set[bytes]) -> dict:
    """Request the table in ranges, keep only cohort rows, never store the whole file."""
    layout = LAYOUT[name]
    i_item, i_key = layout["itemid"], layout["key"]
    total = remote_size(repo, name)
    out_path = dest / f"{name}.csv"
    state_path = dest / f".{name}.progress.json"

    state = {"offset": 0, "kept": 0, "seen": 0, "total": total}
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved.get("total") == total and out_path.exists():
            state = saved
            print(f"{name}: resuming at {state['offset'] / 1e9:.2f} GB", flush=True)

    mode = "ab" if state["offset"] else "wb"
    started = time.time()
    with out_path.open(mode) as sink:
        remainder = b""
        while state["offset"] < total:
            start = state["offset"]
            end = min(start + CHUNK, total) - 1
            payload = fetch_range(repo, name, start, end)
            data = remainder + payload
            # A range boundary lands mid-line; carry the tail to the next chunk so
            # no row is split, and no row is counted twice.
            cut = data.rfind(b"\n")
            remainder = data[cut + 1 :] if cut >= 0 else data
            body = data[: cut + 1] if cut >= 0 else b""

            lines = body.split(b"\n")
            keep = []
            if start == 0 and lines:
                keep.append(lines[0])  # the header, verbatim
                lines = lines[1:]
            for line in lines:
                if not line:
                    continue
                fields = line.split(b",", i_item + 1)
                if len(fields) <= i_item:
                    continue
                if fields[i_item] in itemids and fields[i_key] in keys:
                    keep.append(line)
            state["seen"] += len(lines)
            state["kept"] += len(keep) - (1 if start == 0 else 0)
            if keep:
                sink.write(b"\n".join(keep) + b"\n")
            sink.flush()

            state["offset"] = end + 1
            state_path.write_text(json.dumps(state))
            done = state["offset"] / total
            rate = state["offset"] / max(time.time() - started, 1e-9) / 1e6
            eta = (total - state["offset"]) / 1e6 / max(rate, 1e-9) / 60
            print(
                f"{name}: {done * 100:5.1f}%  {state['offset'] / 1e9:6.2f}/{total / 1e9:.2f} GB  "
                f"kept {state['kept']:,} of {state['seen']:,}  {rate:.1f} MB/s  eta {eta:.0f} min",
                flush=True,
            )

    return {
        "table": name,
        "remote_bytes": total,
        "rows_seen": state["seen"],
        "rows_kept": state["kept"],
        "local_bytes": out_path.stat().st_size,
        "output": str(out_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ntphuc149/MIMIC-III-Clinical-Database")
    parser.add_argument("--dest", default="data/raw/mimic-iii-hf-1.4")
    parser.add_argument("--keys", default="/tmp/mimichf", help="directory of filter key files")
    parser.add_argument("--tables", default="CHARTEVENTS,LABEVENTS")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    keys = Path(args.keys)

    def load(name: str) -> set[bytes]:
        return {line.encode() for line in keys.joinpath(name).read_text().split() if line}

    stays, hadms = load("stays.txt"), load("hadms.txt")
    chart, lab = load("chart_itemids.txt"), load("lab_itemids.txt")
    print(f"cohort stays={len(stays):,} hadm={len(hadms):,} chart_ids={len(chart)} lab_ids={len(lab)}")

    reports = []
    for name in (t.strip() for t in args.tables.split(",")):
        if not name:
            continue
        itemids, matched = (chart, stays) if name == "CHARTEVENTS" else (lab, hadms)
        reports.append(stream_table(args.repo, name, dest, itemids, matched))
        print(json.dumps(reports[-1], indent=2), flush=True)

    (dest / "hf_fetch_manifest.json").write_text(
        json.dumps({"repo": args.repo, "tables": reports}, indent=2) + "\n"
    )
    print(f"manifest={dest / 'hf_fetch_manifest.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
