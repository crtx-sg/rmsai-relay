"""Curate held-out real-ECG samples from an ecg_sigma package into reader-ready HDF5.

  python -m cli.real_samples list
  python -m cli.real_samples pick --pick VENTRICULAR_TACHYCARDIA:3,ATRIAL_FIBRILLATION:2,NORMAL_SINUS:2
  python -m cli.ingest --dir data/real --emit bus

`--package` defaults to `ECGPKG_DIR` (else `../ecg_sigma/packages/ecg_pkg_v2`). The package must be
the one the configured checkpoint was trained on — its `test` split is only held out relative to
that package — so `pick` refuses a manifest mismatch unless `--allow-mismatch` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from common.config import DEFAULT
from ingest.real_samples import (
    SPLITS,
    SampleError,
    available,
    load_package,
    parse_pick,
    select_events,
    write_samples,
)

_DEFAULT_PACKAGE = "../ecg_sigma/packages/ecg_pkg_v2"


def checkpoint_manifest(checkpoints: tuple[str, ...] | list[str]) -> str | None:
    """The training-package manifest sha recorded in the first checkpoint, or None if unknowable."""
    if not checkpoints:
        return None
    try:
        import torch  # noqa: PLC0415

        ckpt = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — a missing/unloadable checkpoint just skips the check
        print(f"[real_samples] cannot read {checkpoints[0]} ({exc}); manifest not checked",
              file=sys.stderr)
        return None
    sha = ckpt.get("package_manifest_sha256") if isinstance(ckpt, dict) else None
    return str(sha) if sha else None


def _cmd_list(pkg, args) -> int:
    avail = available(pkg, args.split)
    print(f"package {pkg.version} (manifest {pkg.manifest_sha256[:12]}), split={args.split}")
    for label in pkg.classes:
        by_ds = avail.get(label)
        total = sum(by_ds.values()) if by_ds else 0
        detail = ", ".join(f"{ds}:{n}" for ds, n in sorted(by_ds.items())) if by_ds else "-"
        print(f"  {label:<26} {total:>6}  ({detail})")
    return 0


def _cmd_pick(pkg, args) -> int:
    pick = parse_pick(args.pick)
    checkpoints = DEFAULT.ecg_checkpoints if args.checkpoint is None else tuple(args.checkpoint)
    if not checkpoints:
        print("[real_samples] WARNING no ECG_CHECKPOINTS/--checkpoint: cannot confirm this package "
              "is the model's training package, and cli.ingest will run the stub", file=sys.stderr)
    model_sha = checkpoint_manifest(checkpoints)
    if model_sha and not pkg.manifest_sha256.startswith(model_sha[:12]):
        msg = (f"package manifest {pkg.manifest_sha256[:12]} != checkpoint's training package "
               f"{model_sha[:12]}: its '{args.split}' split is not held out for this model")
        if not args.allow_mismatch:
            print(f"[real_samples] {msg} (pass --allow-mismatch to proceed)", file=sys.stderr)
            return 2
        print(f"[real_samples] WARNING {msg}", file=sys.stderr)
    if args.split != "test":
        print(f"[real_samples] WARNING split={args.split}: the model has seen these events; "
              "results are not a performance estimate", file=sys.stderr)

    datasets = set(args.dataset) if args.dataset else None
    sel = select_events(pkg, pick, split=args.split, datasets=datasets, seed=args.seed)
    for label, (want, got) in sel.shortfall.items():
        print(f"[real_samples] {label}: wanted {want}, only {got} in split={args.split}",
              file=sys.stderr)
    if not sel.rows:
        print("[real_samples] nothing selected", file=sys.stderr)
        return 1
    files = write_samples(pkg, sel, args.out)
    for r in sel.rows:
        print(json.dumps({"label": r["label"], "dataset": r["dataset"], "subject": r["subject_id"],
                          "event_key": r["event_key"], "file": r["h5_relpath"]}))
    print(json.dumps({"events": len(sel.rows), "files": [str(f) for f in files],
                      "package": pkg.version, "split": args.split, "seed": args.seed}))
    return 0 if files else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", default=os.environ.get("ECGPKG_DIR", _DEFAULT_PACKAGE),
                        help="ecgpkg root (package.json + manifest.csv + data/).")
    parser.add_argument("--split", choices=SPLITS, default="test")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="events available per label in the split")

    p = sub.add_parser("pick", help="write the selected events as reader-ready HDF5")
    p.add_argument("--pick", required=True, help="LABEL:N[,LABEL:N...] using class names.")
    p.add_argument("--dataset", nargs="*", help="restrict to these datasets (e.g. mitbih vfdb).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/real", help="output directory (default data/real).")
    p.add_argument("--checkpoint", nargs="*", default=None,
                   help="checkpoint(s) to check the package against; defaults to ECG_CHECKPOINTS.")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="proceed even if the package is not the checkpoint's training package.")
    args = parser.parse_args(argv)

    try:
        pkg = load_package(args.package)
        return _cmd_list(pkg, args) if args.cmd == "list" else _cmd_pick(pkg, args)
    except SampleError as exc:
        print(f"[real_samples] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
