# enterprise-rag-core — Solution Status

**Repo:** https://github.com/gsachin/enterprise-rag-core
**Date:** 2026-08-29
**Scope:** Standalone, pluggable Enterprise RAG/MCP Core Engine — implemented from
`docs/TRD_ENTERPRISE_RAG_MCP_CORE.md` (v2026.5), migrated from the
`universityDemo` repository where the blueprint was authored.

## ✅ DONE

### Blueprint verification (pre-implementation)
The TRD v2026.5 was written and every code block verified against the pinned
SDK versions before any implementation started (5 verification suites, all
green: smoke, Qdrant filter regression, live ONNX reranker, live Redis Stack
semantic cache, live MCP boot). Revision log (17 rows) lives in the TRD.

### Implementation phases (all committed, each gated by a green test run)

| Phase | Content | Gate |
|---|---|---|
| 1 | Repo seed from the verified design-doc harness: core modules, packaging, venv | 31 passed |
| 2 | Pluggable vector-store adapter layer (Qdrant/Chroma/in-memory + ES/BM25/none keyword legs) + hardened core with lazy SDK imports | 57 passed |
| 3 | `config.py` (`EngineConfig.from_env` / `build_stack`) + full `server.py` rewrite — lifespan wiring, dual auth modes (`oidc` / `none`) | 57 passed + zero-infra smoke |
| 4 | CLI (`download-model`, `serve`, `serve-stdio`) | 65 passed |
| 5 | PDF ingestion (dual representation: structured JSON + markdown matrix) + `ingest` CLI + docker-compose.yml | 67 passed |
| 6 | Redis Stack tests against live Docker container | 69 passed, 0 skipped |

**Current suite: 69 passed, 0 skipped** (with Redis Stack running; without it,
the 2 redis-marked tests auto-skip). Model-dependent reranker tests auto-skip
until `enterprise-rag-core download-model` has run.

### Migration (2026-08-29)
- Moved from `universityDemo` to this repo (user-created, `gsachin`). Remote
  tree verified byte-identical to the local build before re-pointing.
- Design docs moved here (`docs/`); `universityDemo` keeps a pointer.
- LICENSE: all rights reserved (public repo, proprietary code).
- CI: GitHub Actions runs the full suite on Python 3.11/3.12 with a Redis
  Stack service container.

## ⏳ PENDING / NOT DONE

| # | Item | Blocker / what's needed |
|---|---|---|
| 1 | TEST-SEC-01 (10,000 cross-tenant × department × backend requests) | Requires live Qdrant + Elasticsearch + test corpus (docker-compose.yml provides the infra; the load driver is not yet written) |
| 2 | TEST-PERF-01 (500 concurrent HTTP/2 sessions, P95 ≤ 35 ms) | Same full infra + load generator; latency figures in the TRD are SLO *targets* |
| 3 | TEST-INGEST-01 at scale (50 multi-page PDFs) | Converted subset implemented and green (fpdf2-generated PDFs); full corpus not yet assembled |
| 4 | JWKS signature verification end-to-end | `OIDCJWTVerifier` statically verified; needs a real (or stubbed) OIDC issuer — the MCP boot test uses a stub `TokenVerifier` |
| 5 | Live Elasticsearch round-trip | `AsyncElasticsearch.search` verified by signature + keyword-only params; no live ES run yet |
| 6 | Live Qdrant round-trip | Filter regression covered against qdrant-client models API; live server run pending |
| 7 | Semantic cache live check on CI | Covered locally (Phase 6) + CI Redis service; entry needed if the CI service proves unreliable |

## Development notes

- Windows venv at repo root `.venv` (pinned versions in `requirements-dev.txt`).
- Reranker model: `enterprise-rag-core download-model` (HF
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, file `onnx/model_quint8_avx2.onnx`,
  22 MiB, gitignored).
- Redis-marked tests: `docker run -d --name rag-redis-stack -p 6379:6379
  redis/redis-stack-server:latest`, then `pytest -m redis`.
