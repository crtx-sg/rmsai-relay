"""Render an event's ECG waveform to a strip image (the artifact behind `MonitoredEvent.ecg_plot_ref`).

The raw multi-lead ECG lives only in the HDF5 source and, briefly, in `SignalWindow.signals` at the
producer (the bus drops it). So the producer renders the strip here, while the samples are in hand,
and persists just the file path downstream — mirroring how the report markdown is materialized.

`render_ecg_strip` is pure-ish (filesystem + matplotlib) and unit-tested with synthetic samples.
"""

from __future__ import annotations

from pathlib import Path

from common.config import DEFAULT, Config
from common.schemas import SignalWindow

# Preference order for the lead to plot when several are present.
_PREFERRED_LEADS = ("II", "ii", "I", "i", "V5", "v5", "ECG", "ecg")


def _pick_lead(signals: dict[str, list[float]]) -> str | None:
    """Choose the ECG lead to plot: a preferred lead if present, else the first non-empty one."""
    for lead in _PREFERRED_LEADS:
        if signals.get(lead):
            return lead
    return next((k for k, v in signals.items() if v), None)


def render_ecg_strip(window: SignalWindow, *, config: Config = DEFAULT) -> str | None:
    """Render the event's primary ECG lead to `{plot_dir}/{event_id}.png`. Returns the path or None.

    None when there are no samples (e.g. the bus path, where signals were stripped) — callers then
    leave `ecg_plot_ref` as whatever the producer already set.
    """
    lead = _pick_lead(window.signals)
    if lead is None:
        return None
    samples = window.signals[lead]

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless — no display on the worker/CI
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rate = window.sample_rates.get("ecg")
    xs = [i / float(rate) for i in range(len(samples))] if rate else list(range(len(samples)))
    path = Path(config.plot_dir) / f"{window.event_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(xs, samples, linewidth=0.6, color="#b00020")
    ax.set_title(f"{window.patient_ref} · event {window.event_id[:8]} · lead {lead}")
    ax.set_xlabel("seconds" if rate else "sample")
    ax.set_ylabel(window.waveform_units)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return str(path)
