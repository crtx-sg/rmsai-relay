"""Phase 0 synthetic-data CLI.

Two jobs, both deterministic given a seed:
  1. `hdf5`   — generate synthetic event HDF5 via the vendored ecgtranscnn simulator.
  2. `cohort` — build a patient -> (unit, bed) + history cohort from the stub services, so
                Phase 2B has demographics/beds to ingest.

Usage:
  python -m cli.gen_synthetic hdf5 --num-files 2 --events-per-file 3 --seed 1 \
      --output-dir data/synthetic
  python -m cli.gen_synthetic cohort --patients PT1000 PT1001 PT1002
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common.bed_assignment import BedAssignmentStub
from common.patient_history import PatientHistoryStub

_REPO = Path(__file__).resolve().parents[1]
_GEN_SCRIPT = _REPO / "external" / "ecgtranscnn" / "scripts" / "generate_inference_data.py"


def build_cohort(patient_ids: list[str], seed: int | None = None) -> list[dict]:
    """Map each patient to a bed + synthetic history (stable, seeded by patient_id)."""
    beds = BedAssignmentStub()
    hist = PatientHistoryStub()
    cohort = []
    for pid in patient_ids:
        unit, bed = beds.assign(pid)
        cohort.append(
            {"patient_id": pid, "unit": unit, "bed": bed, "history": hist.get(pid, seed).to_dict()}
        )
    return cohort


def _cmd_hdf5(args: argparse.Namespace) -> int:
    if not _GEN_SCRIPT.exists():
        print(
            f"error: vendored generator not found at {_GEN_SCRIPT}\n"
            "  clone it first: git clone https://github.com/crtx-sg/ecgtranscnn "
            "external/ecgtranscnn",
            file=sys.stderr,
        )
        return 2
    cmd = [
        sys.executable, str(_GEN_SCRIPT),
        "--num-files", str(args.num_files),
        "--events-per-file", str(args.events_per_file),
        "--output-dir", args.output_dir,
        "--conditions", args.conditions,
        "--noise-level", args.noise_level,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return subprocess.call(cmd)


def _cmd_cohort(args: argparse.Namespace) -> int:
    cohort = build_cohort(args.patients, args.seed)
    print(json.dumps(cohort, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hdf5", help="generate synthetic event HDF5")
    h.add_argument("--num-files", type=int, default=2)
    h.add_argument("--events-per-file", type=int, default=3)
    h.add_argument("--output-dir", default="data/synthetic")
    h.add_argument("--conditions", default="balanced")
    h.add_argument("--noise-level", default="medium")
    h.add_argument("--seed", type=int, default=None)
    h.set_defaults(func=_cmd_hdf5)

    c = sub.add_parser("cohort", help="build patient/bed/history cohort")
    c.add_argument("--patients", nargs="+", required=True)
    c.add_argument("--seed", type=int, default=None)
    c.set_defaults(func=_cmd_cohort)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
