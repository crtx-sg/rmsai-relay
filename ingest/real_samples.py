"""Curate real-ECG demo/test samples from an ecg_sigma ``ecgpkg`` package → reader-ready HDF5.

The relay reads ecg_sigma HDF5 as-is, but a converted record is hundreds of mostly-unlabelled
windows, and most of them sit in the split the model was *trained* on. This module picks a small,
labelled, **held-out** set instead:

- reads ``package.json`` + ``manifest.csv`` directly (csv/json only — no ``ecg_transcovnet``
  import, so no torch pull; same reasoning as ``_meta_field`` in ``hdf5_reader``);
- selects events by label from one split (default ``test``), deterministically for a seed;
- copies each chosen ``event_*`` group out of its source file, one output file per source record
  (a file carries one ``patient_id``, so records are never merged under one patient);
- stamps the manifest label onto the event's ``condition`` attr (the reader's ground truth) and the
  package identity + split onto ``/metadata`` attrs, so a run can be traced back and scored.

The source signals are copied byte-for-byte; nothing about the conversion is reimplemented.
Resilience: a manifest row whose file or event group is missing is skipped with a logged error.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import h5py

from common.event_types import CLASS_NAMES
from common.redacting_logger import get_redacting_logger

_log = get_redacting_logger("rmsai.ingest.real_samples")

SPLITS = ("train", "val", "test")


class SampleError(Exception):
    """Raised for an unusable package or selection request (fail loud, before writing)."""


@dataclass(frozen=True)
class Package:
    root: Path
    version: str
    manifest_sha256: str
    classes: tuple[str, ...]
    rows: tuple[dict, ...]


@dataclass
class Selection:
    rows: list[dict]
    #: label -> (requested, found) for every label that came up short.
    shortfall: dict[str, tuple[int, int]] = field(default_factory=dict)


def load_package(root: str | Path) -> Package:
    """Read an ecgpkg's ``package.json`` + ``manifest.csv`` (no HDF5 is opened)."""
    root = Path(root)
    meta_path, manifest_path = root / "package.json", root / "manifest.csv"
    if not meta_path.is_file() or not manifest_path.is_file():
        raise SampleError(f"{root} is not an ecgpkg (needs package.json + manifest.csv)")
    meta = json.loads(meta_path.read_text())
    if meta.get("format") != "ecgpkg":
        raise SampleError(f"{meta_path}: format={meta.get('format')!r}, expected 'ecgpkg'")
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        rows = tuple(csv.DictReader(fh))
    return Package(
        root=root,
        version=str(meta.get("package_version", "?")),
        manifest_sha256=str(meta.get("manifest_sha256", "")),
        classes=tuple(meta.get("classes", ())),
        rows=rows,
    )


def parse_pick(spec: str) -> dict[str, int]:
    """``"VENTRICULAR_TACHYCARDIA:3,ATRIAL_FIBRILLATION:2"`` → ``{label: count}`` (count ≥ 1)."""
    pick: dict[str, int] = {}
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        label, _, count = part.partition(":")
        label = label.strip().upper()
        if label not in CLASS_NAMES:
            raise SampleError(f"unknown class {label!r}; expected one of {', '.join(CLASS_NAMES)}")
        try:
            n = int(count) if count else 1
        except ValueError:
            raise SampleError(f"bad count in {part!r}") from None
        if n < 1:
            raise SampleError(f"count must be >= 1 in {part!r}")
        pick[label] = pick.get(label, 0) + n
    if not pick:
        raise SampleError("empty --pick")
    return pick


def available(pkg: Package, split: str = "test") -> dict[str, Counter]:
    """``{label: Counter(dataset -> events)}`` for one split — what a ``--pick`` can draw on."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in pkg.rows:
        if r.get("split") == split and r.get("label"):
            out[r["label"]][r.get("dataset", "?")] += 1
    return dict(out)


def select_events(
    pkg: Package,
    pick: dict[str, int],
    *,
    split: str = "test",
    datasets: set[str] | None = None,
    seed: int = 42,
) -> Selection:
    """Deterministically choose ``pick[label]`` events per label from ``split``.

    Candidates are sorted before sampling, so the same package + arguments + seed always yield the
    same events. A label with too few candidates returns what exists and is reported as a
    shortfall rather than raising — a partial demo set is still useful.
    """
    if split not in SPLITS:
        raise SampleError(f"split must be one of {SPLITS}, got {split!r}")
    rng = random.Random(seed)
    chosen: list[dict] = []
    shortfall: dict[str, tuple[int, int]] = {}
    for label, n in pick.items():
        cands = sorted(
            (
                r for r in pkg.rows
                if r.get("split") == split
                and r.get("label") == label
                and (datasets is None or r.get("dataset") in datasets)
            ),
            key=lambda r: (r.get("h5_relpath", ""), r.get("event_key", "")),
        )
        if len(cands) < n:
            shortfall[label] = (n, len(cands))
        chosen.extend(rng.sample(cands, min(n, len(cands))))
    return Selection(rows=chosen, shortfall=shortfall)


def write_samples(pkg: Package, selection: Selection, out_dir: str | Path) -> list[Path]:
    """Copy the selected events into ``out_dir``, one file per source record. Returns the files.

    An output file of the same name is replaced — each run is a fresh curation.
    """
    out_dir = Path(out_dir)
    if out_dir.resolve().is_relative_to(pkg.root.resolve()):
        raise SampleError("refusing to write inside the package (it is checksummed)")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in selection.rows:
        by_file[r["h5_relpath"]].append(r)

    written: list[Path] = []
    for relpath, rows in sorted(by_file.items()):
        src_path = pkg.root / relpath
        if not src_path.is_file():
            _log.error("skipping %s: source file missing (%d event(s))", relpath, len(rows))
            continue
        dst_path = out_dir / src_path.name
        n = 0
        with h5py.File(src_path, "r") as src:
            if "metadata" not in src:
                _log.error("skipping %s: no /metadata group", relpath)
                continue
            with h5py.File(dst_path, "w") as dst:
                src.copy(src["metadata"], dst, name="metadata")
                md = dst["metadata"]
                md.attrs["ecgpkg_version"] = pkg.version
                md.attrs["ecgpkg_manifest_sha256"] = pkg.manifest_sha256
                md.attrs["ecgpkg_split"] = rows[0].get("split", "")
                for r in sorted(rows, key=lambda r: r["event_key"]):
                    key = r["event_key"]
                    if key not in src:
                        _log.error("skipping %s/%s: event group missing", relpath, key)
                        continue
                    src.copy(src[key], dst, name=key)
                    grp = dst[key]
                    # The reader's ground truth is the `condition` attr. ecg_sigma's own value is the
                    # raw annotation (often OTHER); the manifest label is what the model is scored on.
                    grp.attrs["source_condition"] = grp.attrs.get("condition", "")
                    grp.attrs["condition"] = r["label"]
                    n += 1
        if n:
            written.append(dst_path)
        else:
            dst_path.unlink(missing_ok=True)
    return written
