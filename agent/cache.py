# agent/cache.py — Semantic query-response cache backed by ChromaDB.
# Embeds queries with multilingual MiniLM; serves cached answers at cosine >= 0.92.

import hashlib
import json
import os
from typing import Optional

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
CACHE_COLLECTION = "response_cache"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.92

_embed_model: Optional[HuggingFaceEmbeddings] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_cache_collection = None


def _get_embed_model() -> HuggingFaceEmbeddings:
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    global _chroma_client, _cache_collection
    if _cache_collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _cache_collection = _chroma_client.get_or_create_collection(
            name=CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _cache_collection


def _query_embedding(text: str) -> list[float]:
    return _get_embed_model().embed_query(text)


def _stable_id(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:32]


def get_cached_response(query: str) -> Optional[dict]:
    """Return cached result dict if a similar query exists, else None."""
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
        similarity = 1.0 - distance

        if similarity < SIMILARITY_THRESHOLD:
            return None

        metadata = results["metadatas"][0][0]
        return {
            "response": results["documents"][0][0],
            "steps": [],        # Steps are not cached; they belong to live runs.
            "tools_used": json.loads(metadata.get("tools_used", "[]")),
            "cache_hit": True,
            "cache_similarity": round(similarity, 4),
        }
    except Exception:
        return None


def store_response(query: str, result: dict) -> None:
    """Persist a query-response pair in the cache (skips images and errors)."""
    if not result.get("response"):
        return
    if "analyze_medical_image" in result.get("tools_used", []):
        return

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
    )
    if any(phrase in response_lower for phrase in _BAD_PHRASES):
        return
    if len(result["response"].strip()) < 60:
        return

    try:
        collection = _get_collection()
        doc_id = _stable_id(query)
        embedding = _query_embedding(query)

        metadata = {
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
        pass


def clear_cache() -> int:
    """Delete all cache entries. Returns count deleted."""
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
    """Return the number of entries in the cache."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
