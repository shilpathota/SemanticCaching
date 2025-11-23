"""
Deep research agent workflow package.

This package provides a complete agentic workflow implementation with semantic caching,
including intelligent query decomposition, research capabilities, quality evaluation,
and conditional routing.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logger
logger = logging.getLogger("agentic-workflow")

# Import core workflow components
from .nodes import (
    WorkflowState,
    initialize_nodes,
    initialize_metrics,
    decompose_question_node,
    decompose_query_node,
    check_cache_node,
    synthesize_response_node,
    evaluate_quality_node,
    research_node,
)

from .edges import (
    initialize_edges,
    route_after_cache_check,
    route_after_quality_evaluation,
    cache_validated_research,
)

from .tools import (
    initialize_tools,
    search_knowledge_base,
)

from .demo import (
    create_demo,
    launch_demo,
    ResearchDemo,
)

from .knowledge_base_utils import (
    KnowledgeBaseManager,
    create_knowledge_base_from_texts,
)


# Initialize functions for easy setup
def initialize_agent(semantic_cache, knowledge_base_index, openai_embeddings):
    """
    Initialize all agent components with required dependencies.

    Args:
        semantic_cache: The semantic cache instance
        knowledge_base_index: Redis search index for the knowledge base
        openai_embeddings: OpenAI embeddings instance
    """
    initialize_nodes(semantic_cache)
    initialize_edges(semantic_cache)
    initialize_tools(knowledge_base_index, openai_embeddings)


def run_agent(agent, query: str, enable_caching: bool = True) -> Dict[str, Any]:
    """Execute a query through the agentic workflow with comprehensive monitoring."""
    start_time = time.perf_counter()

    # Initialize state for the workflow
    initial_state: WorkflowState = {
        "original_query": query,
        "sub_questions": [],
        "sub_answers": {},
        "cache_hits": {},
        "cache_confidences": {},
        "cache_enabled": enable_caching,
        "research_iterations": {},
        "max_research_iterations": 2,
        "research_quality_scores": {},
        "research_feedback": {},
        "current_research_strategy": {},
        "final_response": None,
        "execution_path": [],
        "active_sub_question": None,
        "metrics": initialize_metrics(),
        "timestamp": datetime.now().isoformat(),
        "comparison_mode": False,
        "llm_calls": {},
    }

    logger.info("=" * 80)

    try:
        # Execute the agentic workflow
        final_state = agent.invoke(initial_state)

        # Calculate total execution time and finalize metrics
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Update final metrics with total latency and execution path
        final_metrics = final_state["metrics"].copy()
        final_metrics["total_latency"] = total_latency
        final_metrics["execution_path"] = " → ".join(final_state["execution_path"])

        # Compile comprehensive results with cleaner structure
        result = {
            "original_query": query,
            "sub_questions": final_state["sub_questions"],
            "sub_answers": final_state["sub_answers"],
            "cache_hits": final_state["cache_hits"],
            "cache_confidences": final_state["cache_confidences"],
            "final_response": final_state["final_response"],
            "execution_path": final_metrics["execution_path"],
            "total_latency": f"{total_latency:.2f}ms",
            "metrics": final_metrics,
            "timestamp": final_state["timestamp"],
            "llm_calls": final_state.get("llm_calls", {}),
        }

        logger.info("=" * 80)
        logger.info(f"✅ Workflow query completed in {total_latency:.2f}ms")

        return result

    except Exception as e:
        logger.error(f"❌ Workflow query failed: {e}")
        return {
            "original_query": query,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def display_results(result):
    """Enhanced display function showing workflow execution, caching details, and final response."""
    from IPython.display import display, Markdown

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"\\n🎯 SEMANTIC CACHING WORKFLOW ANALYSIS")
    print("=" * 80)

    # Query & Execution Path
    print(
        f"📝 **Original Query:** {result['original_query'][:100]}{'...' if len(result['original_query']) > 100 else ''}"
    )
    execution_path = result.get("execution_path", "")
    print(f"🔄 **Execution Path:** {execution_path}")

    # Sub-question Analysis with Cache Status
    sub_questions = result.get("sub_questions", [])
    cache_hits = result.get("cache_hits", {})
    cache_confidences = result.get("cache_confidences", {})

    if sub_questions:
        print(f"\\n🧠 **Query Decomposition:** {len(sub_questions)} sub-questions")
        for i, sq in enumerate(sub_questions, 1):
            hit_status = cache_hits.get(sq, False)
            confidence = cache_confidences.get(sq, 0.0)

            if hit_status:
                print(f"   {i}. ✅ **CACHE HIT** ({confidence:.3f}): {sq}")
            else:
                print(f"   {i}. 🔍 **RESEARCH**: {sq}")

    # Performance Metrics
    print(f"\\n📊 **Performance Metrics:**")

    # Cache Performance
    if cache_hits:
        hits = sum(1 for hit in cache_hits.values() if hit)
        hit_rate = hits / len(cache_hits) if cache_hits else 0
        print(
            f"   💾 Cache Hit Rate: **{hit_rate:.1%}** ({hits}/{len(cache_hits)} questions)"
        )

    # LLM Usage and Cost Analysis
    llm_calls = result.get("llm_calls", {})
    if llm_calls:
        analysis_calls = llm_calls.get("analysis_llm", 0)
        research_calls = llm_calls.get("research_llm", 0)
        total_calls = analysis_calls + research_calls
        print(
            f"   🤖 LLM Calls: **{total_calls}** (GPT-4: {analysis_calls}, GPT-4-Mini: {research_calls})"
        )

    # Timing
    metrics = result.get("metrics", {})
    total_latency = result.get("total_latency", "0ms")
    print(f"   ⚡ Total Latency: **{total_latency}**")

    if metrics:
        cache_latency = metrics.get("cache_latency", 0)
        research_latency = metrics.get("research_latency", 0)
        print(f"   ⏱️  Cache: {cache_latency:.0f}ms, Research: {research_latency:.0f}ms")

    # Final Response
    final_response = result.get("final_response", "")
    if final_response:
        print(f"\\n📋 **AI Response:**")
        print("-" * 80)
        display(Markdown(final_response))
        print("-" * 80)

    print("=" * 80)


def analyze_agent_results(results):
    """Analyze the caching performance across scenarios with visual charts"""
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display
    
    scenario_names = [
        "Enterprise Evaluation",
        "Implementation Planning", 
        "Final Review",
    ]

    # Collect data for visualization
    scenario_data = []
    cumulative_questions = 0
    cumulative_hits = 0

    for i, (result, name) in enumerate(zip(results, scenario_names), 1):
        cache_hits = result.get("cache_hits", {})
        sub_questions = result.get("sub_questions", [])
        llm_calls = result.get("llm_calls", {})
        metrics = result.get("metrics", {})
        
        hits = sum(1 for hit in cache_hits.values() if hit)
        total_qs = len(sub_questions)
        hit_rate = (hits / total_qs * 100) if total_qs > 0 else 0

        cumulative_questions += total_qs
        cumulative_hits += hits
        cumulative_rate = (
            (cumulative_hits / cumulative_questions * 100)
            if cumulative_questions > 0
            else 0
        )

        # LLM usage analysis
        analysis_calls = llm_calls.get("analysis_llm", 0)
        research_calls = llm_calls.get("research_llm", 0)
        total_calls = analysis_calls + research_calls

        # Extract timing data
        total_latency_str = result.get("total_latency", "0ms")
        total_latency = float(total_latency_str.replace("ms", "")) if isinstance(total_latency_str, str) else total_latency_str
        cache_latency = metrics.get("cache_latency", 0)
        research_latency = metrics.get("research_latency", 0)

        scenario_data.append({
            'name': name,
            'scenario_num': i,
            'total_questions': total_qs,
            'cache_hits': hits,
            'hit_rate': hit_rate,
            'cumulative_hit_rate': cumulative_rate,
            'total_llm_calls': total_calls,
            'analysis_calls': analysis_calls,
            'research_calls': research_calls,
            'total_latency': total_latency,
            'cache_latency': cache_latency,
            'research_latency': research_latency,
        })

    # Create comprehensive visualization
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("🎯 Semantic Caching Performance Analysis", fontsize=16, fontweight="bold", y=0.98)

    # 1. Cache Hit Rate Progression
    scenarios = [d['scenario_num'] for d in scenario_data]
    hit_rates = [d['hit_rate'] for d in scenario_data]
    cumulative_rates = [d['cumulative_hit_rate'] for d in scenario_data]
    
    ax1.bar(scenarios, hit_rates, alpha=0.7, color='skyblue', label='Per-Scenario Hit Rate')
    ax1.plot(scenarios, cumulative_rates, marker='o', color='darkblue', linewidth=2, label='Cumulative Hit Rate')
    ax1.set_title("💾 Cache Hit Rate Evolution", fontweight="bold")
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Hit Rate (%)")
    ax1.set_xticks(scenarios)
    ax1.set_xticklabels([f"S{i}" for i in scenarios])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (rate, cum_rate) in enumerate(zip(hit_rates, cumulative_rates)):
        ax1.text(scenarios[i], rate + 2, f'{rate:.0f}%', ha='center', fontsize=9)
        ax1.text(scenarios[i], cum_rate + 5, f'{cum_rate:.1f}%', ha='center', fontsize=9, color='darkblue')

    # 2. LLM Calls Breakdown
    analysis_calls = [d['analysis_calls'] for d in scenario_data]
    research_calls = [d['research_calls'] for d in scenario_data]
    
    width = 0.35
    x_pos = np.array(scenarios)
    ax2.bar(x_pos - width/2, analysis_calls, width, label='GPT-4 (Analysis)', color='lightcoral')
    ax2.bar(x_pos + width/2, research_calls, width, label='GPT-4-Mini (Research)', color='lightgreen')
    
    ax2.set_title("🤖 LLM Calls by Type", fontweight="bold")
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Number of Calls")
    ax2.set_xticks(scenarios)
    ax2.set_xticklabels([f"S{i}" for i in scenarios])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (a_calls, r_calls) in enumerate(zip(analysis_calls, research_calls)):
        ax2.text(scenarios[i] - width/2, a_calls + 0.1, str(a_calls), ha='center', fontsize=9)
        ax2.text(scenarios[i] + width/2, r_calls + 0.1, str(r_calls), ha='center', fontsize=9)

    # 3. Latency Breakdown
    cache_latencies = [d['cache_latency'] for d in scenario_data]
    research_latencies = [d['research_latency'] for d in scenario_data]
    
    ax3.bar(scenarios, cache_latencies, label='Cache Latency', color='gold', alpha=0.8)
    ax3.bar(scenarios, research_latencies, bottom=cache_latencies, label='Research Latency', color='orange', alpha=0.8)
    
    ax3.set_title("⚡ Latency Breakdown", fontweight="bold")
    ax3.set_xlabel("Scenario")
    ax3.set_ylabel("Latency (ms)")
    ax3.set_xticks(scenarios)
    ax3.set_xticklabels([f"S{i}" for i in scenarios])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add total latency labels
    for i, (cache_lat, research_lat) in enumerate(zip(cache_latencies, research_latencies)):
        total = cache_lat + research_lat
        ax3.text(scenarios[i], total + 100, f'{total:.0f}ms', ha='center', fontsize=9)

    # 4. Performance Summary
    ax4.axis('off')
    
    # Calculate summary metrics
    total_questions_all = sum(d['total_questions'] for d in scenario_data)
    total_hits_all = sum(d['cache_hits'] for d in scenario_data)
    overall_hit_rate = (total_hits_all / total_questions_all * 100) if total_questions_all > 0 else 0
    total_llm_calls_all = sum(d['total_llm_calls'] for d in scenario_data)
    avg_latency = np.mean([d['total_latency'] for d in scenario_data])
    
    # Create summary text
    summary_text = f"""
📊 OVERALL PERFORMANCE SUMMARY

📝 Total Questions Processed: {total_questions_all}
💾 Total Cache Hits: {total_hits_all} ({overall_hit_rate:.1f}%)
🤖 Total LLM Calls: {total_llm_calls_all}
⚡ Average Latency: {avg_latency:.0f}ms

🚀 Research Calls Avoided: {total_hits_all}
🎯 Key Insight: Cache effectiveness increases with each interaction!
✅ Result: Semantic caching delivers progressive intelligence and cost savings

💡 Cache Evolution:
  • Scenario 1: {scenario_data[0]['hit_rate']:.0f}% hit rate
  • Scenario 2: {scenario_data[1]['hit_rate']:.0f}% hit rate  
  • Scenario 3: {scenario_data[2]['hit_rate']:.0f}% hit rate
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return cumulative_questions, cumulative_hits



__all__ = [
    # State and workflow
    "WorkflowState",
    "initialize_agent",
    "run_agent",
    "analyze_agent_results",
    # Node functions
    "decompose_question_node",
    "check_cache_node",
    "synthesize_response_node",
    "evaluate_quality_node",
    "research_node",
    "process_query_node",
    # Edge/routing functions
    "route_after_cache_check",
    "route_after_quality_evaluation",
    "cache_validated_research",
    # Tools
    "search_knowledge_base",
    # Simple demo
    "create_demo",
    "launch_demo",
    "ResearchDemo",
    # Knowledge base utilities
    "KnowledgeBaseManager", 
    "create_knowledge_base_from_texts",
]
