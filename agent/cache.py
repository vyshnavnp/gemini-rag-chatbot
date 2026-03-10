# agent/cache.py
#
# Semantic query-response cache backed by ChromaDB.
#
# How it works:
#   - On every agent turn, before calling the LLM, we embed the user query
#     and search a dedicated "response_cache" collection in ChromaDB.
#   - If a semantically similar query exists (cosine similarity >= THRESHOLD)
#     we return the cached response immediately, saving an API call.
#   - After a live LLM response is produced, we store (query embedding,
#     response text) so future similar queries can hit the cache.
#
# We embed queries with the same multilingual MiniLM model used by the RAG
# retriever, so no extra model is loaded.
#
# The cache is persistent across restarts because ChromaDB writes to disk.
# Cached entries never expire -- they represent factual oncology answers
# that do not change frequently. Stale entries can be cleared manually via
# clear_cache() if the knowledge base is significantly updated.
#
# Thread safety: ChromaDB handles concurrent reads. Writes from multiple
# Streamlit sessions are serialised by the SQLite WAL. No extra locking needed.

import hashlib
import json
import os
from typing import Optional

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Anchor to project root via __file__ so the cache works regardless of CWD
# (local dev launched from any directory, Docker WORKDIR /app, EC2, etc.).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
CACHE_COLLECTION = "response_cache"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cosine similarity cutoff: 1.0 = identical, 0.0 = completely unrelated.
# 0.92 is tight enough to avoid false hits on different clinical questions
# while catching paraphrases such as "side effects of cisplatin" vs
# "what are cisplatin's adverse effects".
SIMILARITY_THRESHOLD = 0.92

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once per process)
# ---------------------------------------------------------------------------

_embed_model: Optional[HuggingFaceEmbeddings] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_cache_collection = None


def _get_embed_model() -> HuggingFaceEmbeddings:
    """Return (and lazily load) the embedding model singleton."""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    """Return (and lazily open) the ChromaDB cache collection."""
    global _chroma_client, _cache_collection
    if _cache_collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # get_or_create ensures idempotent startup.
        _cache_collection = _chroma_client.get_or_create_collection(
            name=CACHE_COLLECTION,
            # cosine distance so similarity = 1 - distance
            metadata={"hnsw:space": "cosine"},
        )
    return _cache_collection


def _query_embedding(text: str) -> list[float]:
    """Return the embedding vector for a query string."""
    return _get_embed_model().embed_query(text)


def _stable_id(query: str) -> str:
    """
    Generate a stable document ID from the query text.
    Using a hash prevents duplicate insertions for the exact same string.
    """
    return hashlib.sha256(query.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached_response(query: str) -> Optional[dict]:
    """
    Look up the cache for a semantically similar previous query.

    Args:
        query: The user's raw query text.

    Returns:
        A result dict matching the shape returned by run_agent():
            {"response": str, "graph_dot": None, "steps": [], "tools_used": [],
             "cache_hit": True}
        Returns None if no sufficiently similar entry exists.
    """
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return None

        embedding = _query_embedding(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance  # ChromaDB cosine distance -> similarity

        if similarity < SIMILARITY_THRESHOLD:
            return None

        # Deserialise the stored metadata payload.
        metadata = results["metadatas"][0][0]
        return {
            "response": results["documents"][0][0],
            "graph_dot": metadata.get("graph_dot") or None,
            "steps": [],        # Steps are not cached; they belong to live runs.
            "tools_used": json.loads(metadata.get("tools_used", "[]")),
            "cache_hit": True,
            "cache_similarity": round(similarity, 4),
        }
    except Exception:
        # Cache is best-effort; never block the main flow on cache errors.
        return None


def store_response(query: str, result: dict) -> None:
    """
    Persist a query-response pair in the cache.

    Only text responses are cached. Image analysis results are excluded
    because they depend on visual content that cannot be reproduced from
    the query text alone.

    Args:
        query:  The user's raw query text.
        result: The dict returned by run_agent().
    """
    # Do not cache image analysis responses or empty responses.
    if not result.get("response"):
        return
    if "analyze_medical_image" in result.get("tools_used", []):
        return

    # Do not cache error/fallback/broken responses.
    # These were generated when RAG was empty, the API was down, or the agent
    # failed.  Caching them would poison future queries for hours/days.
    response_lower = result["response"].lower()
    _BAD_PHRASES = (
        "could not generate a response",
        "please try again",
        "knowledge base is currently empty",
        "background indexer runs every 30",
        "no relevant oncology information found",
        "api daily quota has been reached",
        "agent encountered an error",
        "image analysis failed",
        "diagram generation failed",
    )
    if any(phrase in response_lower for phrase in _BAD_PHRASES):
        return
    # Reject suspiciously short responses (< 60 chars) — real oncology
    # answers always contain more substance than a one-liner error message.
    if len(result["response"].strip()) < 60:
        return

    try:
        collection = _get_collection()
        doc_id = _stable_id(query)
        embedding = _query_embedding(query)

        metadata = {
            "graph_dot": result.get("graph_dot") or "",
            "tools_used": json.dumps(result.get("tools_used", [])),
        }

        # upsert so re-running the same query refreshes the cached answer.
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[result["response"]],
            metadatas=[metadata],
        )
    except Exception:
        # Cache writes are best-effort; never raise to caller.
        pass


def clear_cache() -> int:
    """
    Delete all entries from the response cache.

    Returns:
        The number of entries that were deleted.
    """
    try:
        collection = _get_collection()
        count = collection.count()
        if count > 0:
            all_ids = collection.get(include=[])["ids"]
            collection.delete(ids=all_ids)
        return count
    except Exception:
        return 0


def cache_size() -> int:
    """Return the number of entries currently in the cache."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
