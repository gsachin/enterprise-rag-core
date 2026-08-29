"""§2 Low-Latency Hybrid Retrieval — weighted RRF over pluggable legs.

The engine fans the dense (VectorStore) and sparse (KeywordStore) legs out
concurrently; latency is max(leg) + fusion overhead. Security filters are the
adapters' responsibility on BOTH legs (design doc §1: identical, unbypassable).
"""
import asyncio
from dataclasses import replace

import httpx

from enterprise_rag.model import Chunk
from enterprise_rag.security import SecurityContext
from enterprise_rag.adapters.protocols import EmbeddingClient, VectorStore, KeywordStore


# ── Embedding (real call, budgeted at <= 6 ms p95) ─────────────────────

class OllamaEmbeddingClient:
    """Concrete default: Ollama nomic-embed-text (mirrors EMBED_MODEL of the
    repository this design is archived in)."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        import os

        self._base_url = base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self._model = model or os.environ.get("EMBED_MODEL", "nomic-embed-text")

    async def embed(self, text: str) -> list[float]:
        # target: <= 6 ms p95 (warm model, local network)
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_sync(self, text: str) -> list[float]:
        """Sync variant — for components that must probe vector dimensions at
        construction time (the §5 RedisVL CustomVectorizer). Loop-independent."""
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
        return resp.json()["embedding"]


# ── Weighted RRF fusion ────────────────────────────────────────────────

def _wrrf(rank: int, k: int = 60) -> float:
    """Reciprocal-rank contribution for a 0-based rank."""
    return 1.0 / (k + rank)


def fuse_wrrf(dense: list[Chunk], sparse: list[Chunk],
              alpha: float = 0.3, k: int = 60) -> list[Chunk]:
    """Weighted RRF: ranks from each leg are deduplicated per leg, weighted,
    and accumulated by chunk_id. No raw-score normalization exists to get wrong."""
    acc: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for leg, weight in ((dense, alpha), (sparse, 1.0 - alpha)):
        seen: set[str] = set()
        for rank, chunk in enumerate(leg):          # 0-based ranks
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            acc[chunk.chunk_id] = acc.get(chunk.chunk_id, 0.0) + weight * _wrrf(rank, k)
            chunks.setdefault(chunk.chunk_id, chunk)
    return sorted(
        (replace(chunks[cid], score=acc[cid]) for cid in acc),
        key=lambda c: -c.score,
    )


# ── Hybrid engine over adapter protocols ───────────────────────────────

class AsyncParallelHybridEngine:
    def __init__(self, dense: VectorStore, keyword: KeywordStore | None,
                 embeddings: EmbeddingClient, *, alpha: float = 0.3,
                 rrf_k: int = 60):
        self._dense = dense
        self._keyword = keyword
        self._embeddings = embeddings
        self._alpha = alpha
        self._rrf_k = rrf_k

    async def embed_query(self, text: str) -> list[float]:
        return await self._embeddings.embed(text)

    async def retrieve_parallel(self, query_text: str, sec_ctx: SecurityContext,
                                top_k: int, fetch_k: int | None = None,
                                query_vector: list[float] | None = None) -> list[Chunk]:
        """Fan the legs out concurrently; latency = max(leg) + fusion overhead.
        Accepts a precomputed query_vector so callers that already embedded the
        query (e.g. for the semantic cache check) do not embed twice. With no
        keyword leg configured, runs dense-only."""
        if query_vector is None:
            query_vector = await self._embeddings.embed(query_text)
        fetch_k = fetch_k or top_k * 2
        if self._keyword is None:
            dense_hits = await self._dense.search(query_vector, sec_ctx, fetch_k)
            sparse_hits: list[Chunk] = []
        else:
            dense_hits, sparse_hits = await asyncio.gather(
                self._dense.search(query_vector, sec_ctx, fetch_k),
                self._keyword.search(query_text, sec_ctx, fetch_k),
            )
        return fuse_wrrf(dense_hits, sparse_hits,
                         alpha=self._alpha, k=self._rrf_k)[:top_k]
