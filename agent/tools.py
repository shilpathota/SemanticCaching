"""
Research tools for the deep research agent workflow.

This module contains the tools used by the research agent to search
the knowledge base and gather information.
"""

import logging
import numpy as np
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.query import VectorQuery

# Configure logger
logger = logging.getLogger("agentic-workflow")

# Global variables that will be set by the notebook
kb_index = None
embeddings = None


def initialize_tools(
    knowledge_base_index: Any, openai_embeddings: OpenAITextVectorizer
):
    """
    Initialize the tools with required dependencies.

    Args:
        knowledge_base_index: Redis search index for the knowledge base
        openai_embeddings: OpenAI embeddings instance
    """
    global kb_index, embeddings
    kb_index = knowledge_base_index
    embeddings = openai_embeddings


@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the Redis knowledge base for relevant information.

    This tool performs semantic vector search through the Redis knowledge base
    to find the most relevant information for answering user questions.

    Args:
        query: The search query
        top_k: Number of top results to return

    Returns:
        Formatted search results with relevance scores
    """
    if not kb_index or not embeddings:
        return "Error: Knowledge base not initialized. Please call initialize_tools() first."

    logger.info(
        f"🔍 Using search_knowledge_base tool for query: '{query}' (top_k={top_k})"
    )

    try:
        # Generate query embedding
        query_vector = embeddings.embed(query)

        # Perform vector search
        search_query = VectorQuery(
            vector=query_vector,
            vector_field_name="content_vector",
            return_fields=["content", "vector_distance"],
            num_results=top_k,
        )

        results = kb_index.query(search_query)

        if not results:
            return f"No relevant information found for query: {query}"

        # Format results with relevance scores
        formatted_results = []
        for i, result in enumerate(results, 1):
            relevance = 1.0 - float(result["vector_distance"])
            formatted_results.append(
                f"Result {i} (relevance: {relevance:.3f}):\n{result['content']}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
