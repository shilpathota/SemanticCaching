"""
Simplified Semantic Cache Wrapper

This module provides a clean, simplified interface for semantic caching with optional reranking.

Example usage with custom reranker:

    # Create cache wrapper
    cache = SemanticCacheWrapper()

    # Define a simple reranker function
    def my_reranker(query: str, candidates: List[dict]) -> List[dict]:
        # Filter candidates based on some criteria
        filtered = [c for c in candidates if c.get("vector_distance", 1.0) < 0.5]
        # Sort by custom logic (e.g., prefer shorter responses)
        return sorted(filtered, key=lambda x: len(x.get("response", "")))

    # Register the reranker
    cache.register_reranker(my_reranker)

    # Use the cache - reranker will be applied automatically
    results = cache.check("What is Python?")

    # Clear reranker to use default behavior
    cache.clear_reranker()
"""

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import redis
from pydantic import BaseModel, Field
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

from cache.config import config as default_config


class CacheResult(BaseModel):
    """
    Standardized result for cache wrapper outputs with rich reranker metadata.

    Core Fields:
    - prompt: cache key text
    - response: cached response text
    - vector_distance: semantic distance from vector index (lower = more similar)
    - cosine_similarity: cosine similarity score (higher = more similar)

    Reranker Metadata:
    - reranker_type: type of reranker used ("cross_encoder", "llm")
    - reranker_score: raw score from reranker (interpretation depends on type)
    - reranker_reason: explanation from reranker (for LLM rerankers)
    """

    # Core fields
    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float

    # Reranker metadata
    reranker_type: Optional[str] = None
    reranker_score: Optional[float] = None
    reranker_reason: Optional[str] = None


class CacheResults(BaseModel):
    query: str
    matches: List[CacheResult]

    def __repr__(self):
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"


def try_connect_to_redis(redis_url: str):
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print("✅ Redis is running and accessible!")
    except redis.ConnectionError:
        print(
            """
            ❌ Cannot connect to Redis. Please make sure Redis is running on localhost:6379
                Try: docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
            """
        )
        raise

    return r


class SemanticCacheWrapper:
    def __init__(
        self,
        name: str = "semantic-cache",
        distance_threshold: float = 0.3,
        ttl: int = 3600,
        redis_url: Optional[str] = None,
    ):
        redis_conn_url = redis_url or getattr(
            default_config, "redis_url", "redis://localhost:6379"
        )
        self.redis = try_connect_to_redis(redis_conn_url)

        self.embeddings_cache = EmbeddingsCache(redis_client=self.redis, ttl=ttl * 24)
        self.langcache_embed = HFTextVectorizer(
            model="redis/langcache-embed-v1", cache=self.embeddings_cache
        )
        self.cache = SemanticCache(
            name=name,
            vectorizer=self.langcache_embed,
            redis_client=self.redis,
            distance_threshold=distance_threshold,
            ttl=ttl,
        )

        # Optional reranker function. When set, all check results will be post-processed.
        # Expected signature: reranker(query: str, candidates: List[dict]) -> List[dict]
        self._reranker: Optional[Callable[[str, List[dict]], List[dict]]] = None

    def pair_distance(self, question: str, answer: str) -> float:
        """Compute semantic distance between question and answer using the vectorizer."""
        q_emb = self.langcache_embed.embed(question)
        a_emb = self.langcache_embed.embed(answer)

        distance = cosine(q_emb, a_emb)
        return distance.item()

    def set_cache_entries(self, question_answer_pairs: List[Tuple[str, str]]):
        self.cache.clear()
        for question, answer in question_answer_pairs:
            self.cache.store(prompt=question, response=answer)

    # --------------------------
    # New constructors & helpers
    # --------------------------
    @classmethod
    def from_config(
        cls,
        config,
    ) -> "SemanticCacheWrapper":
        """
        Construct a SemanticCacheWrapper from a config object with optional overrides.

        The config object should have attributes like:
        - redis_url (default: "redis://localhost:6379")
        - cache_name (default: "semantic-cache")
        - cache_distance_threshold (default: 0.3)
        - cache_ttl_seconds (default: 3600)

        Any of these can be overridden via kwargs:
        - redis_url, name, distance_threshold, ttl

        Example:
            # Use all config values
            wrapper = SemanticCacheWrapper.from_config(config)

            # Override specific values
            wrapper = SemanticCacheWrapper.from_config(config, name="custom-cache")
        """
        return cls(
            redis_url=config["redis_url"],
            name=config["cache_name"],
            distance_threshold=float(config["distance_threshold"]),
            ttl=int(config["ttl_seconds"]),
        )

    def hydrate_from_df(
        self,
        df: pd.DataFrame,
        *,
        q_col: str = "question",
        a_col: str = "answer",
        clear: bool = True,
        ttl_override: Optional[int] = None,
        return_id_map: bool = False,
    ) -> Optional[Dict[str, int]]:
        if clear:
            self.cache.clear()
        question_to_id: Dict[str, int] = {}
        idx = 0
        for row in df[[q_col, a_col]].itertuples(index=False, name=None):
            q, a = row
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            if return_id_map and q not in question_to_id:
                question_to_id[q] = idx
            idx += 1
        return question_to_id if return_id_map else None

    def hydrate_from_pairs(
        self,
        pairs: Iterable[Tuple[str, str]],
        *,
        clear: bool = True,
        ttl_override: Optional[int] = None,
        return_id_map: bool = False,
    ) -> Optional[Dict[str, int]]:
        if clear:
            self.cache.clear()
        question_to_id: Dict[str, int] = {}
        idx = 0
        for q, a in pairs:
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            if return_id_map and q not in question_to_id:
                question_to_id[q] = idx
            idx += 1
        return question_to_id if return_id_map else None

    # --------------------------
    # Reranker registration
    # --------------------------
    def register_reranker(self, reranker: Callable[[str, List[dict]], List[dict]]):
        """
        Register a reranking function.

        The reranker function should have the signature:
        reranker(query: str, candidates: List[dict]) -> List[dict]

        Where:
        - query: The search query string
        - candidates: List of cache hit dictionaries from semantic search
        - Returns: Filtered/reordered list of candidates

        Each candidate dict contains keys like: prompt, response, vector_distance, etc.
        """
        if not callable(reranker):
            raise TypeError("Reranker must be a callable function")
        self._reranker = reranker

    def clear_reranker(self):
        """Remove any registered reranking function and revert to default behavior."""
        self._reranker = None

    def has_reranker(self) -> bool:
        """Check if a reranker function is currently registered."""
        return self._reranker is not None

    # -------------------
    # Cache check methods
    # -------------------
    def check(
        self,
        query: str,
        distance_threshold: Optional[float] = None,
        num_results: int = 1,
        use_reranker_distance: bool = False,
    ) -> List[CacheResult]:
        """
        Check semantic cache for a single query.

        Args:
            query: The query string to search for
            distance_threshold: Maximum semantic distance (lower = more similar)
            num_results: Maximum number of results to return

        Returns:
            List of CacheResult objects (empty list if no matches)
        """

        # Get candidates from semantic cache
        # If no reranker is registered, use the provided num_results
        # If reranker is registered, get larger pool of candidates from initial cache retrieval
        _num_results = (
            num_results if not self.has_reranker() else max(10, 3 * num_results)
        )
        candidates = self.cache.check(
            query, distance_threshold=distance_threshold, num_results=_num_results
        )

        # Exit early if nothing to show
        if not candidates:
            return CacheResults(query=query, matches=[])

        # Apply reranker if registered
        if self.has_reranker():
            candidates = self._reranker(query, candidates)

        results: List[CacheResult] = []
        # Convert to CacheResult objects and apply final filtering
        for item in candidates[:num_results]:
            # Create cache result with metadata
            result = dict(item)
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            result["query"] = query

            # Set reranker metadata
            if self.has_reranker():
                result["reranker_type"] = result.get("reranker_type")
                result["reranker_score"] = result.get("reranker_score")
                result["reranker_reason"] = result.get("reranker_reason")
                if use_reranker_distance:
                    result["vector_distance"] = result["reranker_distance"]

            results.append(CacheResult(**result))

        return CacheResults(query=query, matches=results)

    def check_many(
        self,
        queries: List[str],
        distance_threshold: Optional[float] = None,
        show_progress: bool = False,
        num_results: int = 1,
        use_reranker_distance=False,
    ) -> List[Optional[CacheResult]]:
        """
        Check semantic cache for multiple queries in batch.

        Args:
            queries: List of query strings to search for
            distance_threshold: Maximum semantic distance (lower = more similar)
            num_results: Maximum number of results per query (returns top result)

        Returns:
            List of CacheResult objects or None for each query (maintains order)
        """
        results: List[Optional[CacheResult]] = []
        for q in tqdm(queries, disable=not show_progress):
            cache_results = self.check(
                q, distance_threshold, num_results, use_reranker_distance
            )
            results.append(cache_results)
        return results
