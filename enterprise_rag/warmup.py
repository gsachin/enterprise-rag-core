"""Keyword-leg warm-up.

The BM25 keyword leg is in-memory (zero infra), so a fresh service process
starts with an empty sparse index even though the vector store is persisted.
``warm_keyword_from_vector_store`` repopulates it at boot from the persistent
vector store — the launcher/CLI runs it before serving (env-gated by
``RAG_CORE_WARM_KEYWORD``, default on).
"""
from enterprise_rag.model import UpsertRecord


async def warm_keyword_from_vector_store(stack) -> int:
    """Upserts every persisted chunk of the default tenant into the keyword
    leg. Returns the number of chunks warmed (0 when the store is empty)."""
    chunks = await stack.vector_store.get_all(stack.config.default_tenant)
    if not chunks:
        return 0
    records = [
        UpsertRecord(
            chunk_id=c.chunk_id,
            parent_id=c.parent_id or "",
            tenant_id=c.tenant_id,
            content=c.content,
            section_title=c.section_title,
            required_clearance=c.required_clearance,
            department=c.department,
            vector=[],      # keyword legs never read the vector
        )
        for c in chunks
    ]
    await stack.keyword_store.upsert(records)
    return len(records)
