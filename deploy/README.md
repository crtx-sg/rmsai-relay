# Deploying the companion app (app-only public edge)

Goal: let a **remote clinician** reach the companion app on a **public server**, while the
rmsai-relay services (consumer, worker, Neo4j, Qdrant, Redis, LiveKit, Ollama, artifact files) stay
**private**. The public edge is a thin **nginx** that serves the static app and reverse-proxies the
API to the on-prem gateway. It is **stateless and PHI-free** — no datastore, no artifact files, no
secrets live on the public box.

```
remote browser ──HTTP(S)──> PUBLIC EDGE (nginx, deploy/edge)          [public host]
                              ├─ /                       = static app (app/)
                              └─ /session /ack /artifact ─► on-prem GATEWAY :8080   [private]
                 ──WebRTC───► LiveKit (public ingress, wss + TURN)                  [private/on-prem]
```

Two moving parts differ between local and public:
- **`GATEWAY_UPSTREAM`** (edge → gateway address). Local: `host.docker.internal:8080`. Public: the
  gateway's address reachable from the edge over your VPN/tunnel.
- **`LIVEKIT_PUBLIC_URL`** (browser → LiveKit). Set on the **gateway** (`.env`); returned by
  `POST /session`. Local: blank (falls back to `LIVEKIT_URL=ws://localhost:7880`). Public: the
  `wss://` ingress.

Everything is config-driven and defaults to localhost, so you can test the exact edge path now and
flip to a public host later.

---

## Local test (no public host, no TLS)

Prereq: the relay + gateway + LiveKit running on your box as usual (see repo README / earlier phases):
worker, `cli.gateway` on :8080, LiveKit on :7880, and events pushed via `cli.consume`.

```bash
cd deploy/edge
cp .env.example .env            # defaults are fine for localhost
docker compose up -d
```

Browse **http://localhost:8081/** → PIN → the worklist, acknowledge, inline artifacts, and chat all
work **through the edge** (static + proxy). Quick endpoint check:

```bash
curl -s localhost:8081/session -X POST -H 'content-type: application/json' -d '{"pin":"1234"}'
# -> JSON with a token; "url" is the browser LiveKit URL (LIVEKIT_PUBLIC_URL or LIVEKIT_URL)
```

This proves the split-URL + proxy without any public infrastructure. The direct flow at
`http://localhost:8080/` (gateway) is unchanged.

---

## Going public — what needs to be done

Provided here (`deploy/livekit/`) as config templates; validate on the real host.

1. **Connectivity (blocker — no VPN yet).** The public edge must reach the on-prem gateway (`:8080`)
   and the browser must reach LiveKit. Any VPN works; **WireGuard** is recommended (site-to-site: the
   edge dials the on-prem private IPs). Alternatives: an SSH reverse tunnel, or `cloudflared`. Until a
   link exists, run the edge co-located with the relay. Then set `GATEWAY_UPSTREAM` to the gateway's
   private address over that link.

2. **TLS (mandatory for real PHI).** Terminate HTTPS at the edge:
   - No domain yet → self-signed cert for IP testing; get a domain for Let's Encrypt before real use.
   - Uncomment the 443 server block in `edge/templates/default.conf.template`, the `EDGE_HTTPS_PORT`
     port + `./certs` mount in `edge/docker-compose.yml`, drop `fullchain.pem`/`privkey.pem` in
     `edge/certs/`, and redirect 80→443.
   - LiveKit ingress must be `wss://` (TLS), matching the TURN cert.

3. **Public LiveKit + TURN (choice 2c).** `deploy/livekit/`:
   - Set `deploy/livekit/.env` (real `LIVEKIT_API_KEY`/`SECRET` — NOT devkey/secret; must match the
     gateway's LiveKit creds).
   - Edit the `# EDIT` lines in `livekit.public.yaml` (domain, public IP) — it advertises the public
     IP and runs built-in TURN over TLS. Or use standalone **coturn** (`coturn/turnserver.conf`,
     set `turn.enabled: false` in the yaml).
   - `docker compose -f docker-compose.public.yml up -d` on the LiveKit host; open the media/TURN
     ports (7881/tcp, 50000-50100/udp, 3478, 5349) on the firewall.
   - Set the gateway's **`LIVEKIT_PUBLIC_URL=wss://<your-ingress>`** and restart it.

4. **Auth hardening (real PHI).** The 4-digit shared PIN on a public `/session` is brute-forceable and
   the gateway does **no HTTP rate limiting**. Before real use:
   - Enable the `limit_req` throttle shown in `edge/templates/default.conf.template`.
   - Use a long shared secret (`INBOUND_AUTH_PIN`) — or move to per-user auth (a later story).
   - Persist the gateway's audit log volume (`data/audit.jsonl`).

5. **Firewall.** Expose only the edge (443) and the LiveKit signal/media + TURN ports. Keep the
   gateway, Redis, Neo4j, Qdrant, and artifact files private.

---

## What stays private (never on the public edge)
The gateway, Redis, Neo4j, Qdrant, Ollama, the voice worker, the consumer, and the artifact files
(`data/plots`, `data/reports`). The edge only forwards HTTP and serves static assets; the app and
all data messages/links are pseudonym-only by construction.
