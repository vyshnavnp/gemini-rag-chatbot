"""
One-shot script to remove poisoned entries from the response_cache collection.
Safe to run multiple times. Delete this file after use if desired.
"""
import chromadb
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")

BAD_PHRASES = [
    "could not generate a response",
    "knowledge base is currently empty",
    "no relevant oncology information found",
    "background indexer runs every 30",
]

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    col = client.get_collection("response_cache")
except Exception:
    print("response_cache collection does not exist — nothing to clear.")
    raise SystemExit(0)

count = col.count()
print(f"response_cache has {count} entr{'y' if count == 1 else 'ies'}")

if count == 0:
    print("Cache is already empty.")
    raise SystemExit(0)

results = col.get(include=["documents", "metadatas"])
bad_ids = []
for doc_id, doc in zip(results["ids"], results["documents"]):
    lower = doc.lower()
    flagged = any(phrase in lower for phrase in BAD_PHRASES) or len(doc.strip()) < 30
    if flagged:
        bad_ids.append(doc_id)
        print(f"  STALE: {doc_id[:10]}... -> {doc[:100]!r}")

if bad_ids:
    col.delete(ids=bad_ids)
    print(f"\nDeleted {len(bad_ids)} stale cache entr{'y' if len(bad_ids) == 1 else 'ies'}.")
else:
    print("No stale entries — cache looks clean.")
