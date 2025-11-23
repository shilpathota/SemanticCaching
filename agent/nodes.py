"""
Workflow nodes for the deep research agent.

This module contains all the node functions that implement the core
logic of the agentic workflow with semantic caching.
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from .tools import search_knowledge_base

# Configure logger
logger = logging.getLogger("agentic-workflow")

# Global cache variable that will be set by the notebook
cache = None

# Global LLMs
_analysis_llm = None
_research_llm = None


def get_analysis_llm():
    """Get the configured analysis LLM instance."""
    global _analysis_llm
    if _analysis_llm is None:
        # _analysis_llm = ChatOpenAI(model="gpt-4.1", temperature=0.1, max_tokens=400)
        _analysis_llm = ChatOllama(model="qwen2:0.5b", temperature=0.1, max_tokens=400)
    return _analysis_llm


def get_research_llm():
    """Get the configured research LLM instance."""
    global _research_llm
    if _research_llm is None:
        _research_llm = ChatOllama(
            model="qwen2:0.5b", temperature=0.2, max_tokens=400
        )
    return _research_llm




class WorkflowMetrics(TypedDict):
    """
    Consolidated metrics for workflow performance tracking.
    
    Groups related metrics for cleaner organization and easier analysis.
    """
    # Performance timing (all in milliseconds)
    total_latency: float
    decomposition_latency: float
    cache_latency: float
    research_latency: float
    synthesis_latency: float
    
    # Cache effectiveness
    cache_hit_rate: float  # 0.0 to 1.0
    cache_hits_count: int
    
    # Research efficiency
    questions_researched: int
    total_research_iterations: int
    
    # LLM usage for cost tracking
    llm_calls: Dict[str, int]  # {"analysis_llm": X, "research_llm": Y}
    
    # Workflow structure
    sub_question_count: int
    execution_path: str  # Space-separated path like "decomposed → cache_checked → researched"


class WorkflowState(TypedDict):
    """
    State for agentic workflow with semantic caching and quality evaluation.

    Tracks query processing, caching, research iterations, and performance metrics.
    """

    # Core query management
    original_query: str
    sub_questions: List[str]
    sub_answers: Dict[str, str]
    final_response: Optional[str]

    # Cache management (granular per sub-question)
    cache_hits: Dict[str, bool]
    cache_confidences: Dict[str, float]
    cache_enabled: bool  # Toggle for A/B testing

    # Research iteration and quality control
    research_iterations: Dict[str, int]  # Track iterations per sub-question
    max_research_iterations: int
    research_quality_scores: Dict[str, float]  # Quality score per sub-question
    research_feedback: Dict[str, str]  # Improvement suggestions per sub-question
    current_research_strategy: Dict[str, str]  # Current strategy per sub-question

    # Agent coordination
    execution_path: List[str]
    active_sub_question: Optional[str]

    # Consolidated metrics tracking
    metrics: WorkflowMetrics
    timestamp: str
    comparison_mode: bool  # For cache vs no-cache comparison

    # LLM usage tracking for ROI analysis
    llm_calls: Dict[str, int]  # Track calls to analysis_llm and research_llm


def initialize_metrics() -> WorkflowMetrics:
    """
    Initialize a clean metrics structure with default values.
    """
    return {
        "total_latency": 0.0,
        "decomposition_latency": 0.0,
        "cache_latency": 0.0,
        "research_latency": 0.0,
        "synthesis_latency": 0.0,
        "cache_hit_rate": 0.0,
        "cache_hits_count": 0,
        "questions_researched": 0,
        "total_research_iterations": 0,
        "llm_calls": {},
        "sub_question_count": 0,
        "execution_path": "",
    }


def update_metrics(current_metrics: WorkflowMetrics, **updates) -> WorkflowMetrics:
    """
    Helper to cleanly update metrics with new values.
    """
    updated = current_metrics.copy()
    for key, value in updates.items():
        if key == "llm_calls" and isinstance(value, dict):
            # Merge LLM call counts
            updated["llm_calls"] = {**updated["llm_calls"], **value}
        else:
            updated[key] = value
    return updated


def initialize_nodes(semantic_cache):
    """
    Initialize the nodes with required dependencies.

    Args:
        semantic_cache: The semantic cache instance
    """
    global cache
    cache = semantic_cache


def decompose_question_node(state: WorkflowState) -> WorkflowState:
    """
    Decompose complex queries into focused, cacheable sub-questions.

    This function uses GPT-4 to intelligently break down user queries into
    2-4 independent sub-questions that can be cached and researched separately.
    This maximizes cache hit potential and enables granular caching.
    """
    start_time = time.perf_counter()
    query = state["original_query"]

    logger.info(f"🧠 Supervisor: Decomposing query: '{query[:50]}...'")

    try:
        # Intelligent decomposition - only break down when beneficial
        decomposition_prompt = f"""
        Analyze this customer support query and determine if it needs to be broken down into sub-questions.
        
        Original query: {query}
        
        Rules:
        - If the query is simple and focused on ONE topic, respond with: SINGLE_QUESTION
        - If the query has multiple distinct aspects that would benefit from separate research, break it into 2-4 specific sub-questions
        - Each sub-question should be self-contained and cacheable
        
        If breaking down, provide ONLY the sub-questions, one per line, no numbering.
        If keeping as single question, respond with exactly: SINGLE_QUESTION
        """

        response = get_analysis_llm().invoke(
            [HumanMessage(content=decomposition_prompt)]
        )

        # Track LLM usage
        llm_calls = state.get("llm_calls", {}).copy()
        llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + 1

        # Check if decomposition is needed
        response_content = response.content.strip()
        if response_content == "SINGLE_QUESTION":
            sub_questions = [query]
            logger.info("🧠 Query is simple - keeping as single question")
        else:
            sub_questions = [
                line.strip()
                for line in response_content.split("\n")
                if line.strip()
                and not line.strip().startswith(("1.", "2.", "3.", "4.", "-", "*"))
                and line.strip() != "SINGLE_QUESTION"
            ]

            # Ensure we have reasonable decomposition
            if not sub_questions or len(sub_questions) == 1:
                sub_questions = [query]  # Fallback to original query
            elif len(sub_questions) > 4:
                sub_questions = sub_questions[:4]  # Limit complexity

        # Initialize research tracking for each sub-question
        research_iterations = {sq: 0 for sq in sub_questions}
        research_quality_scores = {sq: 0.0 for sq in sub_questions}
        research_feedback = {sq: "" for sq in sub_questions}
        current_research_strategy = {sq: "initial" for sq in sub_questions}

        # Track performance
        decomposition_time = (time.perf_counter() - start_time) * 1000

        if len(sub_questions) == 1:
            logger.info(
                f"🧠 Kept as single question in {decomposition_time:.2f}ms"
            )
        else:
            logger.info(
                f"🧠 Decomposed into {len(sub_questions)} sub-questions in {decomposition_time:.2f}ms"
            )
            for i, sq in enumerate(sub_questions, 1):
                logger.info(f"   {i}. {sq}")

        # Update metrics cleanly
        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            decomposition_latency=decomposition_time,
            sub_question_count=len(sub_questions),
            llm_calls={"analysis_llm": llm_calls.get("analysis_llm", 0)}
        )

        # Update state immutably
        return {
            **state,
            "sub_questions": sub_questions,
            "sub_answers": {},
            "cache_hits": {},
            "cache_confidences": {},
            "research_iterations": research_iterations,
            "research_quality_scores": research_quality_scores,
            "research_feedback": research_feedback,
            "current_research_strategy": current_research_strategy,
            "execution_path": state["execution_path"] + ["decomposed"],
            "llm_calls": llm_calls,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ Decomposition failed: {e}")
        # Graceful fallback
        return {
            **state,
            "sub_questions": [query],
            "sub_answers": {},
            "cache_hits": {},
            "cache_confidences": {},
            "research_iterations": {query: 0},
            "research_quality_scores": {query: 0.0},
            "research_feedback": {query: ""},
            "current_research_strategy": {query: "initial"},
            "execution_path": state["execution_path"] + ["decomposition_failed"],
        }


def check_cache_node(state: WorkflowState) -> WorkflowState:
    """
    Check semantic cache for each sub-question independently.
    
    This node performs semantic similarity lookups against the Redis-based cache
    for each decomposed sub-question. It serves as a critical decision point in
    the workflow, determining which questions need research vs. which can be
    answered from cache.
    
    Workflow Position: decompose_query_node → check_cache_node → [research_node OR synthesize_response_node]
    
    Key Responsibilities:
    - Semantic similarity matching using vector embeddings
    - Confidence score calculation based on vector distance
    - A/B testing support with cache enable/disable toggle
    - Comprehensive metrics tracking (hit rates, latency, etc.)
    - Graceful error handling with fallback to cache misses
    
    Cache Strategy:
    - Each sub-question is checked independently for maximum granularity
    - Confidence threshold determines cache hit vs miss
    - Results populate sub_answers for immediate synthesis if all cached
    - Cache misses trigger research workflow for those specific questions
    
    This enables granular caching where individual sub-questions can be
    cached and reused across different complex queries, maximizing
    cache efficiency and cost savings.
    """
    if not cache:
        raise RuntimeError(
            "Cache not initialized. Please call initialize_nodes() first."
        )

    start_time = time.perf_counter()
    sub_questions = state.get("sub_questions", [])

    logger.info(f"🔍 Supervisor: Checking cache for {len(sub_questions)} sub-questions")

    cache_hits = {}
    cache_confidences = {}
    sub_answers = {}
    total_hits = 0

    try:
        # Check if caching is enabled (for A/B testing)
        if not state.get("cache_enabled", True):
            logger.info(
                "🔀 Cache disabled - treating all as cache misses for comparison"
            )
            # Update metrics for disabled cache
            updated_metrics = update_metrics(
                state.get("metrics", initialize_metrics()),
                cache_latency=0.0,
                cache_hit_rate=0.0,
                cache_hits_count=0
            )
            
            return {
                **state,
                "cache_hits": {sq: False for sq in sub_questions},
                "cache_confidences": {sq: 0.0 for sq in sub_questions},
                "execution_path": state["execution_path"] + ["cache_disabled"],
                "metrics": updated_metrics,
            }

        # Check each item in the semantic cache
        for sub_question in sub_questions:
            cache_results = cache.check(sub_question, num_results=1)

            if cache_results.matches:
                result = cache_results.matches[0]
                confidence = (2.0 - result.vector_distance) / 2.0
                cache_hits[sub_question] = True
                cache_confidences[sub_question] = confidence
                sub_answers[sub_question] = result.response
                total_hits += 1

                logger.info(
                    f"   ✅ Cache HIT: '{sub_question[:40]}...' (confidence: {confidence:.3f})"
                )
            else:
                cache_hits[sub_question] = False
                cache_confidences[sub_question] = 0.0

                logger.info(f"   ❌ Cache MISS: '{sub_question[:40]}...'")

        cache_latency = (time.perf_counter() - start_time) * 1000
        hit_rate = total_hits / len(sub_questions) if sub_questions else 0

        logger.info(
            f"🔍 Cache check complete: {total_hits}/{len(sub_questions)} hits ({hit_rate:.1%}) in {cache_latency:.2f}ms"
        )

        # Update metrics cleanly
        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            cache_latency=cache_latency,
            cache_hit_rate=hit_rate,
            cache_hits_count=total_hits
        )

        return {
            **state,
            "cache_hits": cache_hits,
            "cache_confidences": cache_confidences,
            "sub_answers": {**state.get("sub_answers", {}), **sub_answers},
            "execution_path": state["execution_path"] + ["cache_checked"],
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ Cache check failed: {e}")
        # Graceful fallback - assume all cache misses
        return {
            **state,
            "cache_hits": {sq: False for sq in sub_questions},
            "cache_confidences": {sq: 0.0 for sq in sub_questions},
            "execution_path": state["execution_path"] + ["cache_check_failed"],
        }


def synthesize_response_node(state: WorkflowState) -> WorkflowState:
    """
    Synthesize sub-answers into a coherent, comprehensive response.

    The supervisor agent uses GPT-4 to intelligently combine answers from
    cached results and research findings into a natural, helpful response.
    """
    start_time = time.perf_counter()
    sub_questions = state.get("sub_questions", [])
    sub_answers = state.get("sub_answers", {})

    logger.info(
        f"🔗 Supervisor: Synthesizing {len(sub_answers)} answers into final response"
    )

    try:
        # Build context from sub-questions and answers
        qa_pairs = []
        for sq in sub_questions:
            if sq in sub_answers:
                qa_pairs.append(f"Q: {sq}\nA: {sub_answers[sq]}")

        if not qa_pairs:
            logger.warning("⚠️ No answers available for synthesis")
            return {
                **state,
                "final_response": "I apologize, but I couldn't find answers to your question. Please try rephrasing or contact support directly.",
                "execution_path": state["execution_path"] + ["synthesis_failed"],
            }

        synthesis_prompt = f"""
        You are a helpful customer support assistant. Combine the following question-answer pairs 
        into a single, coherent, and comprehensive response to the user's original query.
        
        Original query: {state['original_query']}
        
        Information gathered:
        {chr(10).join(qa_pairs)}
        
        Provide a natural, conversational response that:
        - Directly addresses the user's question
        - Integrates all relevant information smoothly
        - Is helpful and actionable
        - Maintains a professional, friendly tone
        """

        messages = [
            SystemMessage(content="You are a helpful customer support assistant."),
            HumanMessage(content=synthesis_prompt),
        ]

        response = get_analysis_llm().invoke(messages)
        final_response = response.content.strip()

        # Track LLM usage
        llm_calls = state.get("llm_calls", {}).copy()
        llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + 1

        synthesis_time = (time.perf_counter() - start_time) * 1000

        logger.info(f"🔗 Response synthesized in {synthesis_time:.2f}ms")
        logger.info(f"📝 Final response: {final_response[:100]}...")

        # Update metrics cleanly
        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            synthesis_latency=synthesis_time,
            llm_calls={"analysis_llm": llm_calls.get("analysis_llm", 0)}
        )

        return {
            **state,
            "final_response": final_response,
            "execution_path": state["execution_path"] + ["synthesized"],
            "llm_calls": llm_calls,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ Response synthesis failed: {e}")
        return {
            **state,
            "final_response": "I apologize, but I encountered an error while preparing your response. Please try again or contact support.",
            "execution_path": state["execution_path"] + ["synthesis_error"],
        }


def evaluate_quality_node(state: WorkflowState) -> WorkflowState:
    """
    Evaluate the quality and adequacy of research results before synthesis.

    Uses the research LLM to review each research result and determine if it's sufficient
    to answer the sub-question or if additional research with different strategies
    is needed. This enables iterative improvement of research quality.
    """
    start_time = time.perf_counter()
    sub_questions = state.get("sub_questions", [])
    sub_answers = state.get("sub_answers", {})
    cache_hits = state.get("cache_hits", {})

    logger.info(
        f"🎯 Quality Evaluation: Evaluating research quality for {len(sub_answers)} answers"
    )

    quality_scores = state.get("research_quality_scores", {}).copy()
    feedback = state.get("research_feedback", {}).copy()
    needs_more_research = []

    # Track LLM usage - initialize outside try block
    llm_calls = state.get("llm_calls", {}).copy()
    research_llm = get_research_llm()

    try:
        for sub_question in sub_questions:
            # Skip cached answers - they're already validated
            if cache_hits.get(sub_question, False):
                quality_scores[sub_question] = 1.0  # Cached answers are high quality
                continue

            # Skip if no answer yet
            if sub_question not in sub_answers:
                needs_more_research.append(sub_question)
                continue

            answer = sub_answers[sub_question]
            current_iteration = state.get("research_iterations", {}).get(
                sub_question, 0
            )

            # Evaluate research quality
            evaluation_prompt = f"""
            Evaluate the quality and completeness of this research result for answering the user's question.
            
            Original sub-question: {sub_question}
            Research result: {answer}
            Research iteration: {current_iteration + 1}
            
            Provide:
            1. A quality score from 0.0 to 1.0 (where 1.0 is perfect, 0.7+ is adequate)
            2. Brief feedback on what's missing or could be improved (if score < 0.7)
            
            Format your response as:
            SCORE: 0.X
            FEEDBACK: [your feedback or "Adequate" if score >= 0.7]
            """

            evaluation = research_llm.invoke([HumanMessage(content=evaluation_prompt)])

            # Track LLM usage
            llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1

            # Parse the evaluation
            try:
                lines = evaluation.content.strip().split("\n")
                score_line = [l for l in lines if l.startswith("SCORE:")][0]
                feedback_line = [l for l in lines if l.startswith("FEEDBACK:")][0]

                score = float(score_line.split("SCORE:")[1].strip())
                feedback_text = feedback_line.split("FEEDBACK:")[1].strip()

                quality_scores[sub_question] = score
                feedback[sub_question] = feedback_text

                # If quality is insufficient and we haven't hit max iterations, mark for more research
                max_iterations = state.get("max_research_iterations", 2)
                if score < 0.7 and current_iteration < max_iterations:
                    needs_more_research.append(sub_question)
                    logger.info(
                        f"   📊 {sub_question[:40]}... - Score: {score:.2f} - Needs improvement"
                    )
                else:
                    logger.info(
                        f"   ✅ {sub_question[:40]}... - Score: {score:.2f} - Adequate"
                    )

            except Exception as parse_error:
                logger.warning(
                    f"Failed to parse evaluation for {sub_question}: {parse_error}"
                )
                # Default to adequate if we can't parse
                quality_scores[sub_question] = 0.8
                feedback[sub_question] = "Evaluation parsing failed - assuming adequate"

        evaluation_time = (time.perf_counter() - start_time) * 1000

        logger.info(f"🎯 Quality evaluation complete in {evaluation_time:.2f}ms")
        logger.info(
            f"📊 {len(needs_more_research)} sub-questions need additional research"
        )

        # Update LLM usage in metrics
        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            llm_calls={"research_llm": llm_calls.get("research_llm", 0)}
        )

        return {
            **state,
            "research_quality_scores": quality_scores,
            "research_feedback": feedback,
            "execution_path": state["execution_path"] + ["quality_evaluated"],
            "llm_calls": llm_calls,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ Quality evaluation failed: {e}")
        # Graceful fallback - assume all research is adequate
        return {
            **state,
            "research_quality_scores": {sq: 0.8 for sq in sub_questions},
            "research_feedback": {
                sq: "Evaluation failed - assuming adequate" for sq in sub_questions
            },
            "execution_path": state["execution_path"] + ["evaluation_failed"],
            "llm_calls": llm_calls,
        }


def research_node(state: WorkflowState) -> WorkflowState:
    """
    Research with strategy adaptation and iteration support.

    This node handles sub-questions that need research, adapting search strategies
    based on previous iteration feedback. Supports multiple research iterations
    with different approaches based on quality evaluation feedback.
    """
    start_time = time.perf_counter()
    cache_hits = state.get("cache_hits", {})
    sub_answers = state.get("sub_answers", {}).copy()
    research_iterations = state.get("research_iterations", {}).copy()
    current_strategies = state.get("current_research_strategy", {}).copy()
    feedback = state.get("research_feedback", {})
    questions_researched = 0

    # Track LLM usage
    llm_calls = state.get("llm_calls", {}).copy()

    from langgraph.prebuilt import create_react_agent

    research_llm = get_research_llm()
    researcher_agent = create_react_agent(
        model=research_llm, tools=[search_knowledge_base]
    )

    logger.info("🔬 Research: Starting investigation with strategy adaptation")

    try:
        for sub_question, is_cached in cache_hits.items():
            if not is_cached:  # Research cache misses or inadequate previous research
                current_iteration = research_iterations.get(sub_question, 0)
                strategy = current_strategies.get(sub_question, "initial")

                logger.info(
                    f"🔍 Researching: '{sub_question[:50]}...' (iteration {current_iteration + 1}, strategy: {strategy})"
                )

                # Adapt research strategy based on iteration and feedback
                research_prompt = sub_question
                if current_iteration > 0 and feedback.get(sub_question):
                    research_prompt = f"""
                    Previous research was insufficient. Feedback: {feedback[sub_question]}
                    
                    Original question: {sub_question}
                    
                    Please research this more thoroughly, focusing on the specific improvements mentioned in the feedback.
                    Use different search terms and approaches than before.
                    """

                # Use ReAct agent with enhanced prompting
                research_result = researcher_agent.invoke(
                    {"messages": [HumanMessage(content=research_prompt)]}
                )

                # Track research LLM usage (ReAct agent makes multiple internal calls)
                llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1

                # Extract the final answer from the agent's response
                if research_result and "messages" in research_result:
                    answer = research_result["messages"][-1].content
                    sub_answers[sub_question] = answer
                    questions_researched += 1

                    # Update iteration count and strategy
                    research_iterations[sub_question] = current_iteration + 1
                    current_strategies[sub_question] = (
                        f"iteration_{current_iteration + 1}"
                    )

                    logger.info(
                        f"   ✅ Research complete (iteration {current_iteration + 1}): '{answer[:60]}...'"
                    )
                    # Note: Caching happens after quality validation, not here
                else:
                    logger.warning(
                        f"   ❌ No response from researcher for: {sub_question}"
                    )
                    sub_answers[sub_question] = (
                        "I couldn't find specific information about this. Please contact support for detailed assistance."
                    )

        research_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"🔬 Research complete: {questions_researched} questions researched in {research_time:.2f}ms"
        )

        # Update metrics cleanly
        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            research_latency=research_time,
            questions_researched=questions_researched,
            total_research_iterations=sum(research_iterations.values()),
            llm_calls={"research_llm": llm_calls.get("research_llm", 0)}
        )

        return {
            **state,
            "sub_answers": sub_answers,
            "research_iterations": research_iterations,
            "current_research_strategy": current_strategies,
            "execution_path": state["execution_path"] + ["researched"],
            "llm_calls": llm_calls,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        # Provide fallback answers for failed research
        for sub_question, is_cached in cache_hits.items():
            if not is_cached and sub_question not in sub_answers:
                sub_answers[sub_question] = (
                    "I encountered an error while researching this question. Please contact support for assistance."
                )

        return {
            **state,
            "sub_answers": sub_answers,
            "execution_path": state["execution_path"] + ["research_failed"],
            "llm_calls": llm_calls,
        }


def decompose_query_node(state: WorkflowState) -> WorkflowState:
    """
    Decompose complex queries into focused, cacheable sub-questions.
    
    This node serves as the entry point of the workflow, intelligently breaking down
    user queries into 2-4 independent sub-questions that can be cached and researched 
    separately. This maximizes cache hit potential and enables granular caching.
    
    Workflow Position: Entry point → check_cache_node
    
    The decomposition strategy:
    - Simple queries stay as single questions
    - Complex queries are broken into focused sub-questions
    - Each sub-question is designed to be independently cacheable
    - Maintains comprehensive metrics for performance tracking
    """
    logger.info("🧠 Decomposing query...")
    
    # Delegate to the existing decomposition logic
    state = decompose_question_node(state)
    
    logger.info("🧠 Query decomposition complete")
    return state
