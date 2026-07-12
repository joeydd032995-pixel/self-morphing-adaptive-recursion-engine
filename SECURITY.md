# Security Policy

## Project status

This is an experimental, actively-developed **v0.1 alpha** project maintained
by a single developer. It is not covered by a formal security program, and
there is no guaranteed response time (SLA) for reports. Reports are handled
on a best-effort basis.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository
(**Security tab → "Report a vulnerability"**) rather than opening a public
issue. This lets us discuss and fix the issue before it's publicly disclosed.

## Security-relevant surface

For context when reporting, the parts of this project that matter most from
a security standpoint are:

- **`fastapi_server.py`** — HTTP API surface. Authenticated via the
  `X-API-Key` header, checked against the `ENGINE_API_KEY` environment
  variable (the server refuses to start if this is unset). Optional rate
  limiting is available via `slowapi`. CORS is not configured by default.
- **Optional LLM provider keys** (`OPENAI_API_KEY`, `GROQ_API_KEY`) — read
  only from environment variables, never logged or persisted to disk.
- **Local persistence** — SQLite database, FAISS index file, and GNN model
  weights are stored on the local filesystem with no encryption at rest by
  default.
- **LLM-generated content entering the knowledge graph** — already gated:
  every path from LLM output (self-teaching proposals, PoG-derived nodes) to
  a graph or knowledge-graph write passes through `symbolic_verifier` and
  `self_auditor_verify` in `organized_self_morphing_engine.py` before being
  written. This is an existing safeguard, not a gap — no need to re-report it
  as a first-order finding unless you've found a way around it.

## Out of scope

- Denial-of-service via local resource exhaustion on a self-hosted,
  single-operator deployment.
- Vulnerabilities in third-party dependencies that are already tracked
  upstream (please report those to the upstream project).
