"""Simulate a batch of alarms from held-out real ECG (ecg_sigma) and score the call gate.

The companion app's Simulate tab runs this; it is also the CLI harness:

  python -m cli.simulate --samples 10 --fp-percent 20

`samples` events are drawn from the ecg_sigma package's held-out `test` split: `fp_percent` of them
are NORMAL_SINUS (a false alarm: the device fired but the rhythm is normal), the rest are drawn
uniformly from the five classes that ring the clinician (criticality >= High). They are written
with `ingest.real_samples` (stamped with a run id), then `cli.ingest --emit bus` classifies and
publishes them, so the running consumer pushes the worklist and places the calls as usual.

The summary scores what the gate *will* do with each one (`cli.ingest`'s per-event `call`), so it
is ready before the calls finish. The model must be the one trained on this package, otherwise
its test split is not held out (refused, as in `cli.real_samples pick`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

from common.config import DEFAULT

#: The arrhythmias that ring by default (criticality High/Critical, common.criticality).
CALL_CLASSES = ("ATRIAL_FIBRILLATION", "ATRIAL_FLUTTER", "SVT",
                "VENTRICULAR_TACHYCARDIA", "VENTRICULAR_FIBRILLATION")
FP_CLASS = "NORMAL_SINUS"
MAX_SAMPLES = 10  # one click rings the phone up to this many times; SVT has only 25 test windows
OUT_DIR = "data/simulate"
_RUN_COUNTER = "rmsai:simulate:runs"
_DEFAULT_PACKAGE = "../ecg_sigma/packages/ecg_pkg_v2"


def plan(samples: int, fp_percent: int, rng: random.Random) -> dict[str, int]:
    """`{label: count}`: round(samples x fp%) NORMAL_SINUS, the rest uniform over CALL_CLASSES."""
    if not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be 1..{MAX_SAMPLES}, got {samples}")
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
        else:  # a normal rhythm rang: the vitals override (decision 1), or the model got it wrong
            out["fp_called_vitals" if e.get("gate") == "fp_override" else "fp_called_model"] += 1
    keys = ("expected_calls", "tp_calls", "missed_worklist", "missed", "fp_suppressed",
            "fp_called_model",
            "fp_called_vitals")
    return {k: out[k] for k in keys}


def _next_run_number(redis_url: str) -> int:
    import redis  # noqa: PLC0415

    return int(redis.Redis.from_url(redis_url).incr(_RUN_COUNTER))


def _ingest(out_dir: str, emit: str = "bus") -> list[dict]:
    """Run `cli.ingest --emit <emit>` on the run's files; return its per-event JSON lines."""
    proc = subprocess.run([sys.executable, "-m", "cli.ingest", "--dir", out_dir, "--emit", emit],
                          capture_output=True, text=True, timeout=600, check=False)
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
        out_dir: str = OUT_DIR, checkpoints=None, run_number=None, ingest=_ingest) -> dict:
    """One Simulate click: pick, write (run-stamped), ingest to the bus, score. Raises ValueError on
    bad input or a package/model mismatch, RuntimeError if the ingest fails."""
    from cli.real_samples import checkpoint_manifest  # noqa: PLC0415
    from ingest.real_samples import SampleError, load_package, select_events, write_samples  # noqa: PLC0415

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
        write_samples(pkg, sel, out_dir, run_id=run_id)
    except SampleError as exc:
        raise ValueError(str(exc)) from None
    events = ingest(out_dir)
    return {"run": run_id, "seed": seed, "pick": pick, "package": pkg.version,
            "shortfall": {k: list(v) for k, v in sel.shortfall.items()},
            "events": [{k: e.get(k) for k in ("patient", "event_type", "ground_truth", "confidence",
                                              "outcome", "gate", "call")} for e in events],
            "summary": tally(events)}


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
    ingest = (lambda d: _ingest(d, "stdout")) if args.dry_run else _ingest
    try:
        result = run(args.samples, args.fp_percent, seed=args.seed, package=args.package,
                     ingest=ingest)
    except (ValueError, RuntimeError) as exc:
        print(f"[simulate] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
