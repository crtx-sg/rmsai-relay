"""Phase 9 companion-app server-side glue: inbox publisher + scoped artifact tokens (+ gateway).

The companion app is an interactive *facility* surface (per-hospital worklist + acknowledge +
in-app text/voice chat + inline artifacts). This package holds the pieces the app talks to that
are NOT the voice worker itself:

* `inbox` — publish critical-event notifications + status changes into the per-hospital LiveKit
  inbox room (``rmsai-inbox-<hospital_id>``) via the LiveKit server API. Notifications carry only
  pseudonyms + short-lived, single-event scoped artifact links; artifact *bytes* never ride the
  data channel.
* `artifact_tokens` — mint/verify the scoped short-lived tokens those links use (Redis-backed).

Bedside MQTT waveform streaming and camera relay remain out of scope.
"""
