# ingest

HDF5 + MQTT readers → `SignalWindow` (per Appendix A). Format changes are a one-file edit here, never in `common/`. Phase 1.

- `hdf5_reader.py`: simulator HDF5 **and** ecg_sigma real-ECG HDF5 (metadata as datasets or attributes, `_meta_field`); one `SignalWindow` per `event_*` group, bad events skipped.
- `real_samples.py`: curate held-out, labelled events from an ecg_sigma `ecgpkg` (test split, deterministic per seed) into reader-ready files, with `PT9#####` pseudonyms and the manifest label as ground truth. CLI: `cli.real_samples`; runbook: `DEMO.md`.
- `mqtt_reader.py`: the MQTT path. `rates.py`: rational sample-rate handling.
