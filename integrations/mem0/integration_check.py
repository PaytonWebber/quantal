"""End-to-end mem0 x quantal integration: real Memory pipeline, real quantal
store, real SQLite history. Only the embedder and LLM factories are mocked
(deterministic hash-seeded embeddings; LLM unused with infer=False)."""
import hashlib
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

DIMS = 256


def embed(text, memory_action=None):
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(DIMS).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


mock_embedder = MagicMock()
mock_embedder.embed.side_effect = embed

with tempfile.TemporaryDirectory() as tmp, \
     patch("mem0.utils.factory.EmbedderFactory.create", return_value=mock_embedder), \
     patch("mem0.utils.factory.LlmFactory.create", return_value=MagicMock()):
    from mem0 import Memory

    config = {
        "vector_store": {
            "provider": "quantal",
            "config": {"collection_name": "itest", "path": tmp, "embedding_model_dims": DIMS},
        },
        "history_db_path": f"{tmp}/history.db",
    }
    m = Memory.from_config(config)

    m.add("I love sci-fi movies", user_id="alice", infer=False)
    m.add("My favourite food is ramen", user_id="alice", infer=False)
    m.add("I am allergic to peanuts", user_id="bob", infer=False)

    hits = m.search("I love sci-fi movies", filters={"user_id": "alice"})["results"]
    assert hits and hits[0]["memory"] == "I love sci-fi movies", hits
    assert hits[0]["score"] > 0.99, hits[0]["score"]
    print(f"search: top hit '{hits[0]['memory']}' score {hits[0]['score']:.3f}")

    alice_all = m.get_all(filters={"user_id": "alice"})["results"]
    assert {r["memory"] for r in alice_all} == {"I love sci-fi movies", "My favourite food is ramen"}
    print(f"get_all: alice has {len(alice_all)} memories")

    bob_hits = m.search("I love sci-fi movies", filters={"user_id": "bob"})["results"]
    assert all(h["memory"] != "I love sci-fi movies" for h in bob_hits), bob_hits
    print("user isolation: alice's memory not visible to bob")

    target = next(r for r in alice_all if r["memory"] == "My favourite food is ramen")
    m.update(target["id"], "My favourite food is sushi")
    updated = m.get(target["id"])
    assert updated["memory"] == "My favourite food is sushi", updated
    print("update: memory text replaced")

    m.delete(target["id"])
    assert m.get(target["id"]) is None
    remaining = m.get_all(filters={"user_id": "alice"})["results"]
    assert [r["memory"] for r in remaining] == ["I love sci-fi movies"], remaining
    print("delete: memory removed, one remains")

    hist = m.history(remaining[0]["id"])
    assert hist and hist[0]["event"] == "ADD"
    print("history: SQLite log intact")

print("ALL INTEGRATION CHECKS PASSED")
