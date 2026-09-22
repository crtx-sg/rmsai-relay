"""`ECGModel` wrapper over the vendored `ecgtranscnn` 7-lead classifier.

`EcgTransConvModel` wraps the upstream preprocessing pipeline + `ECGTransCovNet`. Everything that
varies between checkpoints — the class head, the lead order, the filter preset — is read **from the
checkpoint** via `ecg_transcovnet.checkpoint.load_models`, never hard-coded here. That is what lets
the 13-class real-ECG release (`models/real_v2`) and the 16-class simulator checkpoints both load
through the same wrapper.

Pass several checkpoints to ensemble them (upstream averages their softmax outputs); the `real_v2`
5-fold ensemble is the recommended artifact. With no checkpoint, `get_ecg_model()` returns the
deterministic `StubECGModel` so the whole pipeline runs and tests stay weight-free.

torch / ecgtranscnn are imported **lazily** (inside the real wrapper) so importing this module
does not pull torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from common.ecg_model_stub import StubECGModel
from common.event_types import CLASS_NAMES
from common.interfaces import ECGModel
from common.redacting_logger import get_redacting_logger
from common.schemas import SignalWindow
from ingest.hdf5_reader import ECG_LEADS

_log = get_redacting_logger("rmsai.inference.ecg")

SIGNAL_LENGTH = 2400


def window_to_lead_matrix(window: SignalWindow):
    """Stack the 7 ECG leads into a `(num_leads, SIGNAL_LENGTH)` float32 array (torch-free).

    Missing leads are zero-filled; over/under-length leads are truncated/padded to SIGNAL_LENGTH.
    """
    import numpy as np  # local: numpy is light but keep module import torch-free in spirit

    rows = []
    for lead in ECG_LEADS:
        samples = window.signals.get(lead, [])
        row = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
        n = min(len(samples), SIGNAL_LENGTH)
        if n:
            row[:n] = np.asarray(samples[:n], dtype=np.float32)
        rows.append(row)
    return np.stack(rows, axis=0)


def _as_paths(checkpoint_path: str | Path | Sequence[str | Path]) -> list[Path]:
    """Normalize one path or a sequence of them to a list of `Path`."""
    if isinstance(checkpoint_path, (str, Path)):
        return [Path(checkpoint_path)]
    return [Path(p) for p in checkpoint_path]


class EcgTransConvModel(ECGModel):
    """Real wrapper: ecgtranscnn preprocessing + ECG_TransCovNet classifier (single or ensemble).

    Head, leads and filter preset come from the checkpoint and are validated against this repo's
    expectations at load time. A mismatch raises, so `get_ecg_model` falls back to the stub loudly
    rather than emitting silently mislabelled predictions.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | Sequence[str | Path],
        filter_preset: str | None = None,
    ) -> None:
        import torch  # noqa: PLC0415
        from ecg_transcovnet import FILTER_PRESETS, PreprocessingPipeline  # noqa: PLC0415
        from ecg_transcovnet.checkpoint import load_models  # noqa: PLC0415

        if filter_preset is not None:
            # Pre-real_v2 callers passed this; the checkpoint is now the authority. Accepted so the
            # `get_ecg_model(**kwargs)` signature does not break, but deliberately ignored —
            # honouring it would let a caller silently mis-filter a model's input.
            _log.warning(
                "filter_preset=%r ignored: the preset is read from the checkpoint", filter_preset,
            )

        paths = _as_paths(checkpoint_path)
        device = torch.device("cpu")
        # Sizes the head from the checkpoint, and for >1 path returns a softmax-averaging ensemble
        # after checking the members agree on head, leads and preset.
        loaded = load_models(paths, device)

        if loaded.leads != list(ECG_LEADS):
            raise ValueError(
                f"checkpoint lead order {loaded.leads} != reader's {list(ECG_LEADS)}; "
                "window_to_lead_matrix would feed the model its leads out of order",
            )
        labels = list(loaded.class_spec.names)
        unknown = [name for name in labels if name not in CLASS_NAMES]
        if unknown:
            raise ValueError(f"checkpoint predicts classes outside our vocabulary: {unknown}")

        self._torch = torch
        self._model = loaded.model
        self._device = device
        self._labels = labels
        self._pipeline = PreprocessingPipeline(FILTER_PRESETS[loaded.filter_preset])

        ckpt = loaded.checkpoint
        _log.info(
            "ECG model loaded: %d checkpoint(s), %d classes, filter_preset=%s, "
            "package=%s manifest=%s",
            len(paths),
            len(labels),
            loaded.filter_preset,
            ckpt.get("package_version", "?"),
            str(ckpt.get("package_manifest_sha256", "?"))[:12],
        )

    @property
    def labels(self) -> list[str]:
        """The checkpoint's class head, in softmax output order."""
        return list(self._labels)

    def predict(self, window: SignalWindow) -> tuple[str, float]:
        torch = self._torch
        signal = self._pipeline(window_to_lead_matrix(window))
        with torch.no_grad():
            x = torch.from_numpy(signal).unsqueeze(0).to(self._device)
            logits = self._model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            return self._labels[idx], float(probs[idx].item())


def get_ecg_model(
    checkpoint_path: str | Path | Sequence[str | Path] | None = None, **kwargs,
) -> ECGModel:
    """Return the real wrapper if every checkpoint exists, else the deterministic stub (O1).

    `checkpoint_path` takes one path or several; several load as a softmax-averaging ensemble
    (`models/real_v2/fold{0..4}.pt` is the recommended artifact).
    """
    if not checkpoint_path:
        return StubECGModel()
    paths = _as_paths(checkpoint_path)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        _log.error("ECG checkpoint(s) not found, using stub: %s", ", ".join(missing))
        return StubECGModel()
    try:
        return EcgTransConvModel(paths, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.error("failed to load ECG checkpoint, using stub: %s", exc)
    return StubECGModel()
