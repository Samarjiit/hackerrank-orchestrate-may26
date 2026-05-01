from __future__ import annotations

from typing import Optional

import chromadb

import config
from ingestion import get_chroma_collection, index_corpus
from schemas import RetrievedDoc


class Retriever:
    """Semantic retriever backed by ChromaDB + sentence-transformers embeddings."""

    def __init__(self, collection: Optional[chromadb.Collection] = None) -> None:
        if collection is not None:
            self._collection = collection
        else:
            self._collection = get_chroma_collection()

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = config.TOP_K,
    ) -> list[RetrievedDoc]:
        """
        Retrieve the top-k most relevant chunks for *query*.

        If *company* is provided, restrict results to that company's corpus.
        Falls back to cross-corpus retrieval if company-filtered results are sparse.
        """
        if not query.strip():
            return []

        docs = self._query(query, company=company, n=top_k)

        # If company filter returned too few results, supplement with cross-corpus
        if company and len(docs) < 2:
            fallback = self._query(query, company=None, n=top_k)
            seen = {d.source for d in docs}
            for d in fallback:
                if d.source not in seen:
                    docs.append(d)
                if len(docs) >= top_k:
                    break

        return docs[:top_k]

    def format_context(self, docs: list[RetrievedDoc]) -> str:
        """Format retrieved documents as a readable context block for the prompt."""
        if not docs:
            return "(No relevant documentation found.)"

        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            header = f"[{i}] **{doc.title}** (company: {doc.company}, category: {doc.category})"
            parts.append(f"{header}\n{doc.content}")

        return "\n\n---\n\n".join(parts)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _query(
        self,
        query: str,
        company: Optional[str],
        n: int,
    ) -> list[RetrievedDoc]:
        """Run a ChromaDB query and return RetrievedDoc objects."""
        where: Optional[dict] = {"company": company} if company else None

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n, self._collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        docs: list[RetrievedDoc] = []
        if not results["documents"] or not results["documents"][0]:
            return docs

        for content, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity score (0-1, higher = better)
            score = max(0.0, 1.0 - float(dist))
            docs.append(
                RetrievedDoc(
                    content=content,
                    source=meta.get("source", ""),
                    company=meta.get("company", "unknown"),
                    category=meta.get("category", "general"),
                    title=meta.get("title", ""),
                    score=round(score, 4),
                )
            )

        # Sort by score descending
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs


def build_retriever(force_reindex: bool = False) -> Retriever:
    """Convenience factory: index corpus if needed, then return a Retriever."""
    collection = index_corpus(force=force_reindex)
    return Retriever(collection=collection)
