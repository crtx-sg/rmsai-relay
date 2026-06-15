"""`ECGModel` wrapper over the vendored `ecgtranscnn` 7-lead classifier.

`EcgTransConvModel` wraps the upstream preprocessing pipeline + `ECGTransCovNet` and the provided
checkpoints. Until real weights are present (O1), `get_ecg_model()` returns the deterministic
`StubECGModel` so the whole pipeline runs and tests stay weight-free.

torch / ecgtranscnn are imported **lazily** (inside the real wrapper) so importing this module
does not pull torch.
"""

from __future__ import annotations

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


class EcgTransConvModel(ECGModel):
    """Real wrapper: ecgtranscnn preprocessing + ECG_TransCovNet classifier."""

    def __init__(self, checkpoint_path: str | Path, filter_preset: str = "none") -> None:
        import torch  # noqa: PLC0415
        from ecg_transcovnet import (  # noqa: PLC0415
            ALL_LEADS,
            FILTER_PRESETS,
            NUM_CLASSES,
            ECGTransCovNet,
            PreprocessingPipeline,
        )

        self._torch = torch
        device = torch.device("cpu")
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        saved = ckpt.get("args", {})
        leads = ckpt.get("leads", ALL_LEADS)
        model = ECGTransCovNet(
            num_classes=NUM_CLASSES,
            in_channels=len(leads),
            signal_length=SIGNAL_LENGTH,
            embed_dim=saved.get("embed_dim", 128),
            nhead=saved.get("nhead", 8),
            num_encoder_layers=saved.get("num_encoder_layers", 3),
            num_decoder_layers=saved.get("num_decoder_layers", 3),
            dim_feedforward=saved.get("dim_feedforward", 512),
            dropout=saved.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self._model = model
        self._device = device
        self._pipeline = PreprocessingPipeline(FILTER_PRESETS[filter_preset])

    def predict(self, window: SignalWindow) -> tuple[str, float]:
        torch = self._torch
        signal = self._pipeline(window_to_lead_matrix(window))
        with torch.no_grad():
            x = torch.from_numpy(signal).unsqueeze(0).to(self._device)
            logits = self._model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            return CLASS_NAMES[idx], float(probs[idx].item())


def get_ecg_model(checkpoint_path: str | Path | None = None, **kwargs) -> ECGModel:
    """Return the real wrapper if a checkpoint exists, else the deterministic stub (O1)."""
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            return EcgTransConvModel(checkpoint_path, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _log.error("failed to load ECG checkpoint, using stub: %s", exc)
    return StubECGModel()
