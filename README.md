# enterprise-rag-core

Standalone, pluggable **Enterprise RAG/MCP Core Engine** — hybrid retrieval
(vector + keyword), weighted RRF fusion, ONNX reranking, tenant-scoped semantic
cache, PDF/table ingestion, and an MCP server (OIDC or no-auth).

Implemented from the verified design blueprint
[`docs/TRD_ENTERPRISE_RAG_MCP_CORE.md`](docs/TRD_ENTERPRISE_RAG_MCP_CORE.md)
(v2026.5). Every design claim was verified against the exact pinned SDK
versions listed in `requirements-dev.txt`. See
[`docs/ent-solution-status.md`](docs/ent-solution-status.md) for the current
status and [`docs/ent-chat-context.md`](docs/ent-chat-context.md) for project
context.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Windows; else .venv/bin/pip
.venv/Scripts/pip install -e . --no-deps

# Download the INT8 reranker model (22 MiB, Hugging Face)
.venv/Scripts/enterprise-rag-core download-model
```

Zero-infrastructure mode (no Qdrant / Elasticsearch / Redis):

```bash
.venv/Scripts/enterprise-rag-core serve            # Streamable HTTP on :8000/mcp
.venv/Scripts/enterprise-rag-core serve-stdio      # stdio transport
```

## Backend matrix

| Concern | Backends | Default |
|---|---|---|
| Vector store | `qdrant`, `chroma`, `memory` | `chroma` |
| Keyword leg | `elasticsearch`, `bm25` (in-memory), `none` | `bm25` |
| Semantic cache | `redisvl` (Redis Stack), `memory`, `none` | `none` |
| Embeddings | `ollama` (`OLLAMA_URL` / `EMBED_MODEL`), `mlx` (OpenAI-compatible `/v1/embeddings`) | `auto` — MLX on macOS Apple Silicon, Ollama elsewhere (CUDA/Linux/Windows) |
| Auth | `none` (default tenant), `oidc` (JWT/JWKS) | `none` |

Environment variables (all `RAG_CORE_*`):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_CORE_VECTOR_BACKEND` | `chroma` | `qdrant` \| `chroma` \| `memory` |
| `RAG_CORE_QDRANT_URL` | — | Qdrant server URL |
| `RAG_CORE_CHROMA_PATH` | — | ChromaDB persistent path |
| `RAG_CORE_CHROMA_COLLECTION` | `langchain` | ChromaDB collection |
| `RAG_CORE_KEYWORD_BACKEND` | `bm25` | `elasticsearch` \| `bm25` \| `none` |
| `RAG_CORE_ES_URL` | — | Elasticsearch URL |
| `RAG_CORE_INDEX` | `rag-chunks` | Index / collection name |
| `RAG_CORE_CACHE_BACKEND` | `none` | `redisvl` \| `memory` \| `none` |
| `RAG_CORE_REDIS_URL` | — | Redis Stack URL (RediSearch + RedisJSON required) |
| `RAG_CORE_RERANK_MODEL_PATH` | — | Path to `minilm-int8.onnx` |
| `RAG_CORE_AUTH_MODE` | `none` | `none` \| `oidc` |
| `RAG_CORE_OIDC_ISSUER` / `RAG_CORE_OIDC_AUDIENCE` | — | OIDC verification endpoints |
| `RAG_CORE_DEFAULT_TENANT` | `default` | Tenant used in `none` auth mode |
| `RAG_CORE_DEFAULT_CLEARANCE` | `0` | Clearance used in `none` auth mode |
| `RAG_CORE_EMBED_BACKEND` | `auto` | `auto` \| `ollama` \| `mlx` — auto picks MLX on macOS Apple Silicon, Ollama elsewhere |
| `RAG_CORE_MLX_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible MLX embedding server (vllm-mlx, mlx-serve, ...) |
| `OLLAMA_URL` / `EMBED_MODEL` | `http://localhost:11434` / `nomic-embed-text` | Ollama embedding endpoint (app-compatible); `EMBED_MODEL` is also the model name for `mlx` |

## Usage as a library

```python
from enterprise_rag.config import EngineConfig
from enterprise_rag.security import SecurityContext

config = EngineConfig.from_env()          # RAG_CORE_* env vars
stack = config.build_stack()              # engine, stores, cache, reranker, embeddings

sec = SecurityContext(
    principal_id="u1", tenant_id="acme", roles=["interviewer"],
    departments=[], clearance_level=3, allowed_groups=[],
)
chunks = await stack.engine.retrieve_parallel("leadership rubric", sec, top_k=5)
```

## Ingestion

```bash
.venv/Scripts/enterprise-rag-core ingest --pdf doc.pdf --doc-id hr-policy \
    --tenant acme --department hr --clearance 3
```

Extracts text and tables (dual representation: structured JSON + markdown
matrix), chunks, embeds, and upserts into the configured vector + keyword legs.

## Optional infrastructure (docker compose)

```bash
docker compose up -d     # Qdrant + Elasticsearch + Redis Stack + Ollama
```

## Development

```bash
.venv/Scripts/python -m pytest tests/              # skips redis-marked tests
.venv/Scripts/python -m pytest tests/ -m redis     # needs Redis Stack running
```

The 5 converted verification suites (smoke, Qdrant filter regression, ONNX
reranker, RedisVL cache, MCP boot) preserve the original design-doc assertions.

## License

Proprietary — all rights reserved. See [LICENSE.md](LICENSE.md).
