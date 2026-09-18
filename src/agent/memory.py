"""Memory system for AgentLens.

Provides short-term and long-term memory capabilities for agents,
including semantic search using embeddings.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from collections import OrderedDict


@dataclass
class MemoryItem:
    """A single memory entry.

    Attributes:
        id: Unique identifier
        content: The memory content
        timestamp: When the memory was created
        metadata: Additional metadata
        access_count: How many times this item was accessed
        last_access: When this item was last accessed
    """

    id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access": self.last_access,
        }


class Memory(ABC):
    """Abstract base class for memory implementations."""

    @abstractmethod
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a memory item.

        Args:
            content: Memory content
            metadata: Optional metadata

        Returns:
            Memory item ID
        """
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID.

        Args:
            id: Memory item ID

        Returns:
            MemoryItem if found, None otherwise
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search memories by content similarity.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching MemoryItems
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass

    @abstractmethod
    def get_all(self) -> List[MemoryItem]:
        """Get all memory items.

        Returns:
            List of all MemoryItems
        """
        pass


class ShortTermMemory(Memory):
    """In-memory short-term storage with LRU eviction.

    Provides fast access to recent memories with automatic
    eviction when capacity is reached.
    """

    def __init__(self, max_items: int = 100):
        """Initialize short-term memory.

        Args:
            max_items: Maximum number of items to store
        """
        self.max_items = max_items
        self._storage: OrderedDict[str, MemoryItem] = OrderedDict()
        self._similarity_func: Optional[Callable[[str, str], float]] = None

    def set_similarity_func(self, func: Callable[[str, str], float]) -> None:
        """Set the similarity function for search.

        Args:
            func: Function that takes two strings and returns similarity score (0-1)
        """
        self._similarity_func = func

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a memory item."""
        item_id = str(uuid.uuid4())
        item = MemoryItem(id=item_id, content=content, metadata=metadata or {})
        self._storage[item_id] = item

        # Evict oldest if over capacity
        while len(self._storage) > self.max_items:
            self._storage.popitem(last=False)

        return item_id

    def get(self, id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item."""
        item = self._storage.get(id)
        if item:
            item.access_count += 1
            item.last_access = time.time()
            # Move to end (most recent)
            self._storage.move_to_end(id)
        return item

    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search using keyword matching or similarity function."""
        if not query:
            return list(self._storage.values())[-limit:]

        results = []
        query_lower = query.lower()

        for item in self._storage.values():
            # Simple keyword matching
            if self._similarity_func:
                score = self._similarity_func(query, item.content)
                if score > 0.3:  # Threshold
                    results.append((score, item))
            else:
                # Fallback to simple contains
                if query_lower in item.content.lower():
                    results.append((1.0, item))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]

    def clear(self) -> None:
        """Clear all memories."""
        self._storage.clear()

    def get_all(self) -> List[MemoryItem]:
        """Get all memory items in insertion order."""
        return list(self._storage.values())

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """Get most recent memories.

        Args:
            limit: Number of recent items to return

        Returns:
            List of recent MemoryItems
        """
        items = list(self._storage.values())
        return items[-limit:] if len(items) <= limit else items[-limit:]


class LongTermMemory(Memory):
    """Persistent memory with vector-like search capabilities.

    In production, this would use a proper vector database.
    For now, implements a simple embedding-based search using
    keyword TF-IDF-like scoring.
    """

    def __init__(self, max_items: int = 10000):
        """Initialize long-term memory.

        Args:
            max_items: Maximum items before eviction
        """
        self.max_items = max_items
        self._storage: Dict[str, MemoryItem] = {}
        self._word_weights: Dict[str, float] = {}
        self._total_words: int = 0

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a memory item and update word weights."""
        item_id = str(uuid.uuid4())
        item = MemoryItem(id=item_id, content=content, metadata=metadata or {})
        self._storage[item_id] = item

        # Update word weights (simplified TF-IDF)
        words = content.lower().split()
        for word in set(words):
            if len(word) > 2:  # Skip short words
                self._word_weights[word] = self._word_weights.get(word, 0) + 1
        self._total_words += len(words)

        # Evict if over capacity
        if len(self._storage) > self.max_items:
            self._evict_oldest()

        return item_id

    def _evict_oldest(self) -> None:
        """Evict oldest memory item."""
        if not self._storage:
            return
        oldest_id = min(self._storage.keys(), key=lambda x: self._storage[x].timestamp)
        del self._storage[oldest_id]

    def get(self, id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item."""
        return self._storage.get(id)

    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search using weighted keyword matching.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching MemoryItems ranked by relevance
        """
        if not query:
            return []

        words = query.lower().split()
        scores: Dict[str, float] = {}

        for item_id, item in self._storage.items():
            content_lower = item.content.lower()
            score = 0.0

            for word in words:
                if len(word) <= 2:
                    continue
                # Count occurrences in content
                count = content_lower.count(word)
                if count > 0:
                    # Weight by inverse document frequency
                    idf = 1.0 / (self._word_weights.get(word, 1) + 1)
                    score += count * idf

            if score > 0:
                scores[item_id] = score

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [self._storage[id] for id in sorted_ids[:limit]]

    def clear(self) -> None:
        """Clear all memories."""
        self._storage.clear()
        self._word_weights.clear()
        self._total_words = 0

    def get_all(self) -> List[MemoryItem]:
        """Get all memory items sorted by timestamp."""
        return sorted(self._storage.values(), key=lambda x: x.timestamp, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        return {
            "total_items": len(self._storage),
            "max_items": self.max_items,
            "unique_words": len(self._word_weights),
            "total_words": self._total_words,
        }


class HybridMemory:
    """Combines short-term and long-term memory.

    Provides seamless transitions between fast short-term
    access and persistent long-term storage.
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        consolidation_threshold: int = 10,
    ):
        """Initialize hybrid memory.

        Args:
            short_term: Short-term memory instance
            long_term: Long-term memory instance
            consolidation_threshold: Access count before moving to long-term
        """
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()
        self.consolidation_threshold = consolidation_threshold

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add to short-term memory primarily."""
        return self.short_term.add(content, metadata)

    def get(self, id: str) -> Optional[MemoryItem]:
        """Get from either memory layer."""
        # Check short-term first
        item = self.short_term.get(id)
        if item:
            return item
        return self.long_term.get(id)

    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search across both memory layers."""
        # Get from short-term
        st_results = self.short_term.search(query, limit=limit)

        # Get from long-term
        lt_results = self.long_term.search(query, limit=limit)

        # Merge and dedupe, prioritizing short-term
        seen_ids = set()
        merged = []

        for item in st_results:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                merged.append(item)

        for item in lt_results:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                merged.append(item)

        return merged[:limit]

    def consolidate(self, item_id: str) -> None:
        """Move item from short-term to long-term.

        Args:
            item_id: ID of item to consolidate
        """
        item = self.short_term.get(item_id)
        if item:
            # Add to long-term
            self.long_term.add(item.content, item.metadata)
            # Remove from short-term
            del self.short_term._storage[item_id]

    def clear(self) -> None:
        """Clear both memory layers."""
        self.short_term.clear()
        self.long_term.clear()
