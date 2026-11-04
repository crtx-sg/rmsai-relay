"""Simulate a batch of alarms from held-out real ECG (ecg_sigma) and score the call gate.

The companion app's Simulate tab runs this; it is also the CLI harness:

  python -m cli.simulate --samples 10 --fp-percent 20

`samples` events are drawn from the ecg_sigma package's held-out `test` split: `fp_percent` of them
are NORMAL_SINUS (a false alarm: the device fired but the rhythm is normal), the rest are drawn
uniformly from the five classes that ring the clinician (criticality >= High). They are written
with `ingest.real_samples` (stamped with a run id), then `cli.ingest --emit bus` classifies and
publishes them, so the running consumer pushes the worklist and places the calls as usual.

Two hospitals: `SIMULATE_TARGETS=h1=redis://localhost:6379/0,h2=redis://localhost:6380/0` splits
each run between the stacks by patient (a stable hash, so a patient always belongs to one hospital)
and ingests both at once, so both hospitals' phones ring together. Unset: this hospital only.

The summary scores what the gate *will* do with each one (`cli.ingest`'s per-event `call`), so it
is ready before the calls finish. The model must be the one trained on this package, otherwise
its test split is not held out (refused, as in `cli.real_samples pick`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.config import DEFAULT

#: The arrhythmias that ring by default (criticality High/Critical, common.criticality).
CALL_CLASSES = ("ATRIAL_FIBRILLATION", "ATRIAL_FLUTTER", "SVT",
                "VENTRICULAR_TACHYCARDIA", "VENTRICULAR_FIBRILLATION")
FP_CLASS = "NORMAL_SINUS"
MAX_SAMPLES = 10  # one click rings the phone up to this many times; SVT has only 25 test windows
OUT_DIR = "data/simulate"  # + /<this hospital>/<target hospital>/
HISTORY = "data/simulate/runs.jsonl"  # one line per published run (the Simulate tab's history)
_RUN_COUNTER = "rmsai:simulate:runs"
_DEFAULT_PACKAGE = "../ecg_sigma/packages/ecg_pkg_v2"


def plan(samples: int, fp_percent: int, rng: random.Random,
         max_samples: int | None = MAX_SAMPLES) -> dict[str, int]:
    """`{label: count}`: round(samples x fp%) NORMAL_SINUS, the rest uniform over CALL_CLASSES.
    `max_samples=None`: no cap (the command-line pick, which rings nothing by itself)."""
    if samples < 1 or (max_samples is not None and samples > max_samples):
        raise ValueError(f"samples must be 1..{max_samples or 'any'}, got {samples}")
    if not 0 <= fp_percent <= 100:
        raise ValueError(f"fp_percent must be 0..100, got {fp_percent}")
    fps = round(samples * fp_percent / 100)
    pick = Counter(rng.choice(CALL_CLASSES) for _ in range(samples - fps))
    if fps:
        pick[FP_CLASS] = fps
    return dict(pick)


def tally(lines: list[dict]) -> dict:
    """Score `cli.ingest` event lines (ground truth + the gate's `call`) for the results panel."""
    out = Counter()
    for e in lines:
        fp = e.get("ground_truth") == FP_CLASS
        if not fp:
            out["expected_calls"] += 1
            # A vitals-driven alert (rhythm below the confidence gate) still reaches the worklist.
            out["tp_calls" if e.get("call") else
                "missed_worklist" if e.get("gate") == "vitals_alert" else "missed"] += 1
        elif not e.get("call"):
            out["fp_suppressed"] += 1
        else:  # a normal rhythm rang: the vitals carried it (a confident NORMAL_SINUS overridden, or
            # an unsure rhythm re-based on the vitals), or the model called an arrhythmia ("ok")
            vitals = e.get("gate") in ("fp_override", "vitals_alert")
            out["fp_called_vitals" if vitals else "fp_called_model"] += 1
    keys = ("expected_calls", "tp_calls", "missed_worklist", "missed", "fp_suppressed",
            "fp_called_model",
            "fp_called_vitals")
    return {k: out[k] for k in keys}


def targets(config=DEFAULT) -> dict[str, str]:
    """`{hospital: redis_url}` from SIMULATE_TARGETS ("h1=redis://…,h2=redis://…"), else just us."""
    raw = os.environ.get("SIMULATE_TARGETS", "").strip()
    if not raw:
        return {config.hospital_id or "default": config.redis_url}
    out = {}
    for part in raw.split(","):
        name, sep, url = part.strip().partition("=")
        if not (sep and name and url.startswith("redis://")):
            raise ValueError(f"SIMULATE_TARGETS entry must be hospital=redis://…, got {part!r}")
        out[name] = url
    return out


def hospital_for(patient: str, hospitals: list[str]) -> str:
    """A patient's hospital: stable across runs, so one patient never appears in two stacks."""
    return sorted(hospitals)[int(hashlib.sha256(patient.encode()).hexdigest(), 16) % len(hospitals)]


def _next_run_number(redis_url: str) -> int:
    import redis  # noqa: PLC0415

    try:
        return int(redis.Redis.from_url(redis_url).incr(_RUN_COUNTER))
    except redis.RedisError as exc:
        raise RuntimeError(f"run counter (Redis {redis_url}): {type(exc).__name__}") from None


def _ingest(out_dir: str, redis_url: str, hospital: str, emit: str = "bus") -> list[dict]:
    """Run `cli.ingest --emit <emit>` on one hospital's files; return its per-event JSON lines.
    Scored with that hospital's policy (HOSPITAL_ID), not the policy of the app that ran it."""
    env = None
    if hospital != "default":
        env = {**os.environ, "HOSPITAL_ID": hospital}
        if hospital != DEFAULT.hospital_id:  # its events are audited in its own log, not ours
            env["AUDIT_LOG_PATH"] = f"data/audit-{hospital}.jsonl"
    try:
        proc = subprocess.run([sys.executable, "-m", "cli.ingest", "--dir", out_dir, "--emit", emit,
                               "--redis-url", redis_url],
                              capture_output=True, text=True, timeout=600, check=False, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"cli.ingest for {hospital} timed out") from None
    if proc.returncode != 0:
        raise RuntimeError(f"cli.ingest failed ({proc.returncode}): {proc.stderr.strip()[-400:]}")
    lines = []
    for raw in proc.stdout.splitlines():
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if isinstance(d, dict) and "event_id" in d:
            lines.append(d)
    return lines


def run(samples: int, fp_percent: int, *, seed: int | None = None, package: str | None = None,
        out_dir: str | None = None, checkpoints=None, run_number=None, ingest=_ingest) -> dict:
    """One Simulate click: pick, write (run-stamped), split by hospital, ingest each hospital's share
    to its bus in parallel, score. Raises ValueError on bad input or a package/model mismatch,
    RuntimeError if an ingest fails."""
    from cli.real_samples import checkpoint_manifest  # noqa: PLC0415
    from ingest.real_samples import (  # noqa: PLC0415
        SampleError,
        load_package,
        pseudonym_for,
        select_events,
        write_samples,
    )

    buses = targets()
    # Per calling hospital: two gateways simulating at once must not clear each other's files.
    out = Path(out_dir or f"{OUT_DIR}/{DEFAULT.hospital_id or 'default'}")

    seed = random.SystemRandom().randrange(2**31) if seed is None else seed
    pick = plan(samples, fp_percent, random.Random(seed))
    try:
        pkg = load_package(package or os.environ.get("ECGPKG_DIR", _DEFAULT_PACKAGE))
    except SampleError as exc:
        raise ValueError(f"ecg_sigma package: {exc}") from None
    checkpoints = DEFAULT.ecg_checkpoints if checkpoints is None else checkpoints
    if not checkpoints:  # the stub's answers would make the scores meaningless
        raise ValueError("no ECG_CHECKPOINTS: the model would be the stub")
    model_sha = checkpoint_manifest(checkpoints)
    if model_sha and not pkg.manifest_sha256.startswith(model_sha[:12]):
        raise ValueError(f"package {pkg.version} is not the model's training package: its test "
                         "split is not held out for this model")
    sel = select_events(pkg, pick, seed=seed)
    n = (run_number or (lambda: _next_run_number(DEFAULT.redis_url)))()
    run_id = f"r{n}-{time.strftime('%H%M')}-n{samples}-fp{fp_percent}"
    try:
        files = write_samples(pkg, sel, out, run_id=run_id)
    except SampleError as exc:
        raise ValueError(str(exc)) from None
    # write_samples names each file after its source record; move it to its patient's hospital.
    patient_of = {Path(r["h5_relpath"]).name: pseudonym_for(r.get("subject_id") or r["h5_relpath"])
                  for r in sel.rows}
    for h in buses:
        shutil.rmtree(out / h, ignore_errors=True)
        (out / h).mkdir(parents=True)
    for f in files:
        f.rename(out / hospital_for(patient_of[f.name], list(buses)) / f.name)
    shares = [h for h in buses if any((out / h).glob("*.h5"))]
    def _one(h):  # an exception per hospital: one stack failing must not hide the other's calls
        try:
            return ingest(str(out / h), buses[h], h), None
        except Exception as exc:  # noqa: BLE001 - reported per hospital below
            return [], f"{type(exc).__name__}: {exc}"[:300]

    with ThreadPoolExecutor(max_workers=len(shares) or 1) as pool:
        results = list(pool.map(_one, shares))
    errors = {h: err for h, (_, err) in zip(shares, results) if err}
    if errors and len(errors) == len(shares):
        raise RuntimeError("; ".join(f"{h}: {e}" for h, e in errors.items()))
    events = [{**e, "hospital": h} for h, (lines, _) in zip(shares, results) for e in lines]
    return {"run": run_id, "seed": seed, "pick": pick, "package": pkg.version, "errors": errors,
            "shortfall": {k: list(v) for k, v in sel.shortfall.items()},
            "events": [{k: e.get(k) for k in ("hospital", "patient", "event_type", "ground_truth",
                                              "confidence", "outcome", "gate", "call")}
                       for e in events],
            "summary": tally(events)}


def record(result: dict, path: str | Path = HISTORY) -> None:
    """Append one published run to the history (gate decisions; delivery is per hospital)."""
    line = {"run": result["run"], "ts": time.time(), "seed": result.get("seed"),
            "package": result.get("package"), "pick": result.get("pick"),
            "hospitals": dict(Counter(e.get("hospital") or "-" for e in result["events"])),
            "summary": result["summary"], "errors": result.get("errors") or {}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def history(path: str | Path = HISTORY) -> dict:
    """Every recorded run (newest first), the summed tiles and two rates with Wilson 95 % intervals:
    arrhythmias the gate rang, and false positives it filtered. A bad line is skipped."""
    from inference.metrics import wilson  # noqa: PLC0415

    runs = []
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                try:
                    runs.append(json.loads(raw))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    totals = Counter()
    for r in runs:
        totals.update({k: v for k, v in (r.get("summary") or {}).items() if isinstance(v, int)})
    fps = totals["fp_suppressed"] + totals["fp_called_model"] + totals["fp_called_vitals"]
    return {"runs": runs[::-1], "totals": dict(totals),
            "rates": {"rang": wilson(totals["tp_calls"], totals["expected_calls"]),
                      "fp_filtered": wilson(totals["fp_suppressed"], fps)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, required=True, help=f"1..{MAX_SAMPLES}")
    parser.add_argument("--fp-percent", type=int, default=0, help="0..100 NORMAL_SINUS share")
    parser.add_argument("--seed", type=int, default=None, help="repeat an earlier draw")
    parser.add_argument("--package", default=None, help="ecgpkg root (default ECGPKG_DIR)")
    parser.add_argument("--dry-run", action="store_true",
                        help="classify and score only: publish nothing (no worklist, no calls)")
    args = parser.parse_args(argv)
    ingest = (lambda d, url, h: _ingest(d, url, h, "stdout")) if args.dry_run else _ingest
    try:  # a dry run publishes nothing, so it takes no run number (and needs no Redis)
        result = run(args.samples, args.fp_percent, seed=args.seed, package=args.package,
                     ingest=ingest, run_number=(lambda: 0) if args.dry_run else None)
    except (ValueError, RuntimeError) as exc:
        print(f"[simulate] {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        record(result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
