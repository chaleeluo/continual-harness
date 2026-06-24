"""
Memory store — persistent long-term memory for PokeAgent.

Inherits from BaseStore and adds text search with multi-factor reranking.
Handles migration from legacy ``knowledge_base.json`` and ``category``→``path``.
"""

import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from utils.stores.base_store import BaseStore

logger = logging.getLogger(__name__)

# Reranking weights
_RERANK_WEIGHTS = {
    "importance": 1.0,
    "relevance": 2.0,
    "recency": 0.5,
    "popularity": 0.3,
}

# Default ranking strategy
_RERANK_STRATEGY = "hybrid"  # "importance", "relevance", "hybrid"


@dataclass
class MemoryEntry:
    """A single entry in long-term memory."""
    id: str = ""
    path: str = "uncategorized"
    title: str = ""
    content: str = ""
    location: Optional[str] = None
    coordinates: Optional[tuple] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = None  # type: ignore[assignment]
    updated_at: str = None  # type: ignore[assignment]
    importance: int = 3
    source: str = "orchestrator" # "evolved" | "orchestrator"
    last_modified_step: Optional[int] = None
    mutation_history: List[dict] = field(default_factory=list)
    access_count: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


# Backward-compat aliases
KnowledgeEntry = MemoryEntry


class Memory(BaseStore[MemoryEntry]):
    """Persistent long-term memory store with multi-factor reranking."""

    file_name = "memory.json"
    id_prefix = "mem_"
    store_label = "LONG-TERM MEMORY"
    empty_message = "No memory entries yet."
    entry_class = MemoryEntry

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self._migrate_if_needed()
        self.load()
        self._maybe_migrate_access_count()

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate_if_needed(self) -> None:
        """Auto-migrate from knowledge_base.json -> memory.json,
        and convert legacy ``category`` field to ``path``."""
        legacy_file = os.path.join(self.cache_dir, "knowledge_base.json")
        if not os.path.exists(self.store_file) and os.path.exists(legacy_file):
            try:
                with open(legacy_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry_dict in data.get("entries", {}).values():
                    entry_dict.setdefault("source", "orchestrator")
                    entry_dict.setdefault("last_modified_step", None)
                    entry_dict.setdefault("mutation_history", [])
                    self._migrate_category_to_path(entry_dict)
                with open(self.store_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Migrated {legacy_file} -> {self.store_file}")
            except Exception as e:
                logger.warning(f"Failed to migrate knowledge_base.json: {e}")

    def _maybe_migrate_access_count(self) -> None:
        """Ensure all entries have the ``access_count`` field after load."""
        changed = False
        for e in self.entries.values():
            if getattr(e, "access_count", None) is None:
                e.access_count = 0
                changed = True
        if changed:
            self.save()

    @staticmethod
    def _migrate_category_to_path(entry_dict: dict) -> None:
        """Convert a legacy ``category`` field to ``path``, removing it."""
        if "category" in entry_dict and "path" not in entry_dict:
            entry_dict["path"] = entry_dict.pop("category")
        elif "category" in entry_dict:
            entry_dict.pop("category")

    # ------------------------------------------------------------------
    # Deserialization override
    # ------------------------------------------------------------------

    def _deserialize_entry(self, entry_dict: dict) -> MemoryEntry:
        if entry_dict.get("coordinates"):
            entry_dict["coordinates"] = tuple(entry_dict["coordinates"])
        entry_dict.setdefault("source", "orchestrator")
        entry_dict.setdefault("last_modified_step", None)
        entry_dict.setdefault("mutation_history", [])
        entry_dict.setdefault("access_count", 0)
        self._migrate_category_to_path(entry_dict)
        entry_dict.setdefault("path", "uncategorized")
        return MemoryEntry(**entry_dict)

    # ------------------------------------------------------------------
    # Backward-compat add() — translates legacy ``category`` kwarg
    # ------------------------------------------------------------------

    def add(self, **fields) -> str:
        if "category" in fields and "path" not in fields:
            fields["path"] = fields.pop("category")
        elif "category" in fields:
            fields.pop("category")
        return super().add(**fields)

    # ------------------------------------------------------------------
    # Memory-specific: text search
    # ------------------------------------------------------------------

    def search(
        self,
        path: Optional[str] = None,
        location: Optional[str] = None,
        tags: Optional[List[str]] = None,
        query: Optional[str] = None,
        min_importance: int = 1,
        rerank: bool = True,
        rerank_strategy: str = _RERANK_STRATEGY,
        top_k: Optional[int] = None,
        # Legacy kwarg
        category: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Search memory with optional multi-factor reranking.

        Args:
            path: Filter by path prefix.
            location: Filter by exact location match.
            tags: Filter by tag intersection.
            query: Text query (keyword matching).
            min_importance: Minimum importance threshold.
            rerank: If True (default), apply weighted reranking.
            rerank_strategy: One of ``"importance"``, ``"relevance"``, or ``"hybrid"``.
            top_k: If set, return only the top K results after reranking.
            category: Legacy alias for *path*.
        """
        effective_path = path or category
        results = []

        for entry in self.entries.values():
            if entry.importance < min_importance:
                continue
            if effective_path and not entry.path.startswith(effective_path):
                continue
            if location and entry.location != location:
                continue
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            if query:
                query_lower = query.lower()
                if (query_lower not in entry.title.lower() and
                        query_lower not in entry.content.lower()):
                    continue
            results.append(entry)

        if not results:
            return results

        if rerank and len(results) > 1:
            results = self._rerank(results, query or "", strategy=rerank_strategy)

        if top_k is not None and top_k < len(results):
            results = results[:top_k]

        return results

    def _rerank(
        self,
        entries: List[MemoryEntry],
        query: str,
        strategy: str = "hybrid",
    ) -> List[MemoryEntry]:
        """Multi-factor reranking of search results.

        Computes a composite score for each entry using:
        - **Importance**: 1–5 rating (normalized to 0–1).
        - **Relevance**: TF-overlap between query tokens and (title + content).
        - **Recency**: Exponential decay over time (half-life ≈ 7 days).
        - **Popularity**: Normalised access_count.

        Returns entries sorted by descending composite score.
        """
        now = datetime.now()
        query_tokens = set(re.findall(r"\w+", query.lower())) if query else set()

        idf = self._compute_idf(query_tokens) if query_tokens else {}

        scored = []
        for e in entries:
            importance_norm = e.importance / 5.0

            relevance = self._compute_relevance(e, query_tokens, idf) if query_tokens else 0.0

            recency = self._compute_recency(e, now)

            popularity = min(e.access_count / 50.0, 1.0)

            if strategy == "importance":
                score = importance_norm
            elif strategy == "relevance":
                score = relevance if relevance > 0 else importance_norm * 0.5
            else:
                score = (
                    _RERANK_WEIGHTS["importance"] * importance_norm
                    + _RERANK_WEIGHTS["relevance"] * relevance
                    + _RERANK_WEIGHTS["recency"] * recency
                    + _RERANK_WEIGHTS["popularity"] * popularity
                )

            scored.append((score, e))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored]

    def _compute_idf(self, query_tokens: set) -> Dict[str, float]:
        """Compute inverse document frequency for query tokens across all entries.

        Uses smoothed IDF: ``log(1 + N / (1 + df))`` to avoid negative values
        when a token appears in every document.
        """
        if not self.entries:
            return {}
        n = len(self.entries)
        df: Counter = Counter()
        for e in self.entries.values():
            text = (e.title + " " + e.content).lower()
            for token in query_tokens:
                if token in text:
                    df[token] += 1
        return {
            t: math.log(1 + n / (1 + df[t]))
            for t in query_tokens
        }

    @staticmethod
    def _compute_relevance(entry: MemoryEntry, query_tokens: set, idf: Dict[str, float]) -> float:
        """TF-IDF-style relevance score: sum of TF · IDF per query token.

        TF is raw token count divided by total token count in the entry
        (title + content), so longer documents with repeated query terms
        score higher.
        """
        if not query_tokens:
            return 0.0
        text = (entry.title + " " + entry.content).lower()
        tokens = re.findall(r"\w+", text)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return 0.0
        tf: Counter = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            w = idf.get(token, 1.0)
            score += (tf.get(token, 0) / total_tokens) * w
        return score / len(query_tokens)

    @staticmethod
    def _compute_recency(entry: MemoryEntry, now: datetime) -> float:
        """Exponential recency decay. Half-life ≈ 7 days."""
        try:
            updated = datetime.fromisoformat(entry.updated_at)
        except (ValueError, TypeError):
            return 0.5
        delta_hours = (now - updated).total_seconds() / 3600
        half_life = 168.0
        return math.exp(-delta_hours / half_life) if delta_hours >= 0 else 1.0

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        entry = super().get(entry_id)
        if entry is not None:
            entry.access_count += 1
            self.save()
        return entry

    def get_all(self, path: Optional[str] = None, category: Optional[str] = None,
                rerank: bool = False, query: str = "") -> List[MemoryEntry]:
        """Return all entries, optionally filtered by path prefix.

        If *rerank* is True, applies the hyrid reranker (useful for
        orchestrator-level recall where all entries are candidates).
        """
        effective_path = path or category
        results = [e for e in self.entries.values()
                   if not effective_path or e.path.startswith(effective_path)]
        if rerank and len(results) > 1:
            results = self._rerank(results, query)
        return results

    def get_summary(self, max_entries: int = 20, min_importance: int = 3) -> str:
        """Legacy summary format — retained for backward compat."""
        important = [e for e in self.entries.values() if e.importance >= min_importance]
        important.sort(key=lambda e: (e.importance, e.updated_at or ""), reverse=True)
        important = important[:max_entries]

        if not important:
            return "No memory entries yet."

        by_path: Dict[str, list] = {}
        for entry in important:
            by_path.setdefault(entry.path, []).append(entry)

        lines = ["=== LONG-TERM MEMORY SUMMARY ==="]
        for path_key in sorted(by_path.keys()):
            lines.append(f"\n[{path_key.upper()}]")
            for entry in by_path[path_key]:
                location_str = f" @ {entry.location}" if entry.location else ""
                coords_str = f" ({entry.coordinates[0]}, {entry.coordinates[1]})" if entry.coordinates else ""
                lines.append(f"  • {entry.title}{location_str}{coords_str}")
                if len(entry.content) <= 100:
                    lines.append(f"    {entry.content}")
                else:
                    lines.append(f"    {entry.content[:97]}...")

        lines.append(f"\nTotal: {len(important)} important entries (showing importance {min_importance}+)")
        return "\n".join(lines)


# Backward-compat aliases
KnowledgeBase = Memory

# Global singleton
_memory_store: Optional[Memory] = None


def get_memory_store() -> Memory:
    """Get or create the global Memory instance (persistent across runs)."""
    global _memory_store
    if _memory_store is None:
        _memory_store = Memory()
    return _memory_store


get_knowledge_base = get_memory_store
