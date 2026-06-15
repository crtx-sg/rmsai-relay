"""MQTT/stream reader parity: a stream packet yields the same SignalWindow as the HDF5 event."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest.hdf5_reader import read_hdf5_file
from ingest.mqtt_reader import iter_packet_payloads, packet_to_window, window_to_packet

_FIXTURE = next((Path(__file__).resolve().parents[1] / "data" / "fixtures").glob("*.h5"))


@pytest.fixture(scope="module")
def hdf5_window():
    return next(read_hdf5_file(_FIXTURE))


def test_packet_roundtrip_equals_hdf5_window(hdf5_window):
    packet = window_to_packet(hdf5_window)
    # Packet must be JSON-serializable (it crosses the wire).
    packet = json.loads(json.dumps(packet))
    reconstructed = packet_to_window(packet)
    assert reconstructed == hdf5_window  # HDF5 and MQTT yield the SAME SignalWindow


def test_packet_preserves_rational_rate(hdf5_window):
    packet = json.loads(json.dumps(window_to_packet(hdf5_window)))
    w = packet_to_window(packet)
    assert w.sample_rates["resp"] == hdf5_window.sample_rates["resp"]


def test_iter_payloads(hdf5_window):
    payload = json.dumps(window_to_packet(hdf5_window))
    windows = list(iter_packet_payloads([payload, payload.encode()]))
    assert len(windows) == 2
    assert windows[0] == hdf5_window
