"""
world_memory.py — ChromaDB-Backed Semantic World Memory (Phase 11)

Provides a persistent, queryable store of objects and locations the robot
has observed. Backs the "find_object" task action with historical knowledge
so the robot doesn't have to re-detect every object from scratch.

Uses ChromaDB as the vector store with text embeddings for semantic queries.

Example queries:
  - "Where did I last see the mug?" → returns (x, y, z) position
  - "What objects are on the table?" → returns [{"class": "mug", "pos": ...}, ...]
  - "Find me something I can drink from" → semantic nearest-neighbour search

Usage:
    from memory.world_memory import WorldMemory

    memory = WorldMemory(persist_dir="data/world_memory")
    memory.remember_object("mug", position=[0.4, 0.1, 0.8], properties={"color": "red"})
    result = memory.find_object("mug")
    print(result.position)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """
    A single entry in the world memory store.

    Attributes:
        object_id: Unique ID for this object observation.
        class_name: Object class (e.g., "025_mug").
        position: Last observed 3D position [x, y, z] in metres (base frame).
        timestamp: Unix timestamp of last observation.
        properties: Additional properties (color, size, etc.).
        confidence: Detection confidence at time of observation.
        n_observations: How many times this object has been seen.
    """
    object_id: str
    class_name: str
    position: list[float]
    timestamp: float = field(default_factory=time.time)
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    n_observations: int = 1


class WorldMemory:
    """
    Persistent semantic world memory using ChromaDB.

    Stores object observations as vector embeddings, enabling:
    1. Exact lookup: "where is the mug I saw 5 minutes ago?"
    2. Semantic search: "find something round on the table"
    3. Temporal queries: "what changed since my last visit?"

    ChromaDB provides both a vector store (for semantic search) and
    metadata filtering (for exact attribute queries).
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/world_memory",
        collection_name: str = "world_objects",
    ) -> None:
        """
        Initialise the world memory with ChromaDB backend.

        Args:
            persist_dir: Directory for ChromaDB persistent storage.
            collection_name: ChromaDB collection name.

        TODO: Phase 11 — implement:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = None       # chromadb.PersistentClient — set in Phase 11
        self._collection = None   # chromadb.Collection — set in Phase 11
        log.info(
            f"WorldMemory created (ChromaDB not yet initialised — TODO: Phase 11). "
            f"Persist dir: {persist_dir}"
        )

    def remember_object(
        self,
        class_name: str,
        position: list[float],
        object_id: str | None = None,
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> str:
        """
        Store or update an object observation in world memory.

        If an object of the same class exists nearby (within 0.2m), updates
        its position and increments n_observations. Otherwise creates a new entry.

        Args:
            class_name: Object class name (e.g., "025_mug").
            position: Observed 3D position [x, y, z] in metres.
            object_id: Optional explicit ID. Auto-generated if None.
            properties: Optional property dict (color, size, etc.).
            confidence: Detection confidence [0, 1].

        Returns:
            object_id of the stored/updated entry.

        TODO: Phase 11 — implement with ChromaDB upsert:
            embedding = self._embed_object(class_name, properties)
            self._collection.upsert(
                ids=[object_id],
                embeddings=[embedding],
                metadatas=[{
                    "class_name": class_name,
                    "position_x": position[0],
                    "position_y": position[1],
                    "position_z": position[2],
                    "timestamp": time.time(),
                    "confidence": confidence,
                }],
                documents=[class_name],
            )
        """
        raise NotImplementedError(
            "TODO: Phase 11 — implement remember_object() with ChromaDB upsert."
        )

    def find_object(
        self,
        class_name: str,
        max_age_seconds: float = 300.0,
    ) -> MemoryEntry | None:
        """
        Find the most recently observed object of a given class.

        Args:
            class_name: Object class to search for.
            max_age_seconds: Ignore observations older than this. Default: 5 minutes.

        Returns:
            MemoryEntry of the most recent observation, or None if not found.

        TODO: Phase 11 — implement with ChromaDB metadata filter:
            results = self._collection.query(
                query_texts=[class_name],
                n_results=1,
                where={"timestamp": {"$gt": time.time() - max_age_seconds}},
            )
        """
        raise NotImplementedError(
            "TODO: Phase 11 — implement find_object() with ChromaDB metadata query."
        )

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[MemoryEntry]:
        """
        Find objects matching a natural language query.

        Args:
            query: Natural language description (e.g., "something to drink from").
            n_results: Maximum number of results to return.

        Returns:
            List of MemoryEntry objects sorted by semantic similarity.

        TODO: Phase 11 — implement with ChromaDB semantic search:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
            )
        """
        raise NotImplementedError(
            "TODO: Phase 11 — implement semantic_search() using ChromaDB vector search."
        )

    def forget_old_entries(self, max_age_seconds: float = 3600.0) -> int:
        """
        Remove observations older than max_age_seconds.

        Args:
            max_age_seconds: Age threshold (default: 1 hour).

        Returns:
            Number of entries removed.

        TODO: Phase 11 — implement with ChromaDB delete:
            cutoff = time.time() - max_age_seconds
            self._collection.delete(where={"timestamp": {"$lt": cutoff}})
        """
        raise NotImplementedError(
            "TODO: Phase 11 — implement forget_old_entries() with ChromaDB delete."
        )

    def get_all_objects(self) -> list[MemoryEntry]:
        """
        Return all objects currently in world memory.

        Returns:
            List of all MemoryEntry objects.

        TODO: Phase 11 — implement:
            results = self._collection.get()
            return [self._to_memory_entry(r) for r in results]
        """
        raise NotImplementedError(
            "TODO: Phase 11 — implement get_all_objects()."
        )

    def clear(self) -> None:
        """
        Clear all entries from world memory.

        Used at the start of each episode or when entering a new room.

        TODO: Phase 11 — implement:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(self.collection_name)
        """
        raise NotImplementedError(
            "TODO: Phase 11 — implement clear() with ChromaDB collection reset."
        )
