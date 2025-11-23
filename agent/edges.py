"""
Routing and edge logic for the deep research agent workflow.

This module contains the conditional routing functions that determine
the flow of execution through the agentic workflow graph.
"""

import logging

from typing import Literal, Dict, Any

# Configure logger
logger = logging.getLogger("agentic-workflow")

# Global cache variable that will be set by the notebook
cache = None


def initialize_edges(semantic_cache):
    """
    Initialize the edges module with required dependencies.

    Args:
        semantic_cache: The semantic cache instance
    """
    global cache
    cache = semantic_cache


def route_after_cache_check(state: Dict[str, Any]) -> Literal["research", "synthesize"]:
    """
    Intelligent routing based on cache results.

    Determines whether we need to route to the researcher agent
    (if cache misses exist) or can proceed directly to synthesis
    (if all sub-questions were cached).
    """
    cache_hits = state.get("cache_hits", {})

    # Check if any sub-questions had cache misses
    cache_misses = [sq for sq, hit in cache_hits.items() if not hit]

    if cache_misses:
        logger.info(
            f"🔀 Routing to researcher: {len(cache_misses)} cache misses detected"
        )
        for miss in cache_misses:
            logger.info(f"   🔍 Will research: '{miss[:50]}...'")
        return "research"
    else:
        logger.info("🔀 Routing to synthesis: all sub-questions cached!")
        return "synthesize"


def route_after_quality_evaluation(
    state: Dict[str, Any],
) -> Literal["research", "synthesize"]:
    """
    Intelligent routing after quality evaluation - decide if more research is needed.

    Determines whether we need additional research iterations for inadequate answers
    or can proceed to synthesis with the current research quality.
    """
    quality_scores = state.get("research_quality_scores", {})
    research_iterations = state.get("research_iterations", {})
    max_iterations = state.get("max_research_iterations", 2)

    # Find sub-questions that need more research
    needs_more_research = []
    for sub_question, score in quality_scores.items():
        current_iteration = research_iterations.get(sub_question, 0)
        # Need more research if quality is low and we haven't hit max iterations
        if score < 0.7 and current_iteration < max_iterations:
            needs_more_research.append(sub_question)

    if needs_more_research:
        logger.info(
            f"🔄 Routing to additional research: {len(needs_more_research)} questions need improvement"
        )
        for sq in needs_more_research:
            iteration = research_iterations.get(sq, 0)
            score = quality_scores.get(sq, 0)
            logger.info(
                f"   🔍 Improve: '{sq[:40]}...' (score: {score:.2f}, iteration: {iteration + 1})"
            )
        return "research"
    else:
        logger.info("🔀 Routing to synthesis: all research quality is adequate!")
        # Cache validated research results now that quality is confirmed
        cache_validated_research(state)
        return "synthesize"


def cache_validated_research(state: Dict[str, Any]):
    """
    Cache research results that have passed quality validation.
    This ensures we only cache high-quality, validated responses.
    """
    if not cache:
        logger.warning("⚠️ Cache not initialized - skipping research caching")
        return

    if not state.get("cache_enabled", True):
        return  # Skip if caching disabled for A/B testing

    cache_hits = state.get("cache_hits", {})
    sub_answers = state.get("sub_answers", {})
    quality_scores = state.get("research_quality_scores", {})

    cached_count = 0
    try:
        for sub_question, answer in sub_answers.items():
            # Only cache if it wasn't already cached and quality is adequate
            if (
                not cache_hits.get(sub_question, False)
                and quality_scores.get(sub_question, 0) >= 0.7
            ):
                cache.cache.store(prompt=sub_question, response=answer)
                cached_count += 1
                logger.info(
                    f"   💾 Cached validated research: '{sub_question[:40]}...'"
                )

        if cached_count > 0:
            logger.info(f"✅ Cached {cached_count} validated research results")
    except Exception as e:
        logger.warning(f"⚠️ Research caching failed: {e}")
