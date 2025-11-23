"""
Clean and Simple Gradio Demo for Deep Research Agent

A streamlined interface for demonstrating semantic caching with web content analysis.
Requires Tavily API for web content extraction.
"""

import gradio as gr
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any
from getpass import getpass

import redis
from tavily import TavilyClient
from .knowledge_base_utils import KnowledgeBaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")


def _get_tavily_api_key() -> str:
    """Get Tavily API key from environment or user input"""
    # First try to get from environment
    api_key = os.getenv("TAVILY_API_KEY")
    
    if api_key:
        logger.info("Using Tavily API key from environment")
        return api_key
    
    # If not in environment, prompt user
    logger.info("Tavily API key not found in environment, requesting from user...")
    api_key = getpass("Please enter your Tavily API key: ")
    
    if not api_key.strip():
        raise ValueError("Tavily API key is required to run the demo")
    
    return api_key.strip()


class ResearchDemo:
    """Simple demo state manager"""
    
    def __init__(self, workflow_app, semantic_cache, tavily_api_key, redis_url="redis://localhost:6379"):
        self.workflow_app = workflow_app
        self.semantic_cache = semantic_cache
        self.tavily_client = TavilyClient(api_key=tavily_api_key, api_base_url=os.getenv("DLAI_TAVILY_BASE_URL"))
        self.current_url = None
        self.run_log = []  # Keep accumulating log of all runs
        
        # Initialize Redis and knowledge base manager
        redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.kb_manager = KnowledgeBaseManager(redis_client)
    
    def _reset_all_data(self) -> str:
        """Clear all knowledge bases and semantic cache for fresh start"""
        try:
            # Clear all knowledge bases
            kb_status = self.kb_manager.clear_knowledge_base()
            
            # Clear semantic cache
            self.semantic_cache.cache.clear()
            
            # Reset demo state
            self.current_url = None
            self.run_log = []
            
            logger.info("🧹 Complete reset: cleared knowledge bases and semantic cache")
            return f"🧹 Reset complete: {kb_status}, semantic cache cleared"
            
        except Exception as e:
            error_msg = f"❌ Reset failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    def process_url(self, url: str) -> str:
        """Extract content from URL using Tavily and build knowledge base"""
        if not url.strip():
            return "❌ Please enter a URL"
        
        try:
            # First, reset all data for fresh start
            reset_status = self._reset_all_data()
            logger.info(f"Reset status: {reset_status}")
            
            # Extract content using Tavily
            result = self.tavily_client.extract(
                urls=[url], 
                extract_depth="advanced"
            )
            
            if not result or 'results' not in result or not result['results']:
                return "❌ No content extracted from URL"
            
            # Get the extracted content
            extracted_content = result['results'][0].get('raw_content', '') or result['results'][0].get('content', '')
            if not extracted_content:
                return "❌ No readable content found"
            
            # Create knowledge base from extracted content
            success, message, kb_index = self.kb_manager.create_knowledge_base(
                source_id=url,
                content=extracted_content,
                chunk_size=4500,
                chunk_overlap=250
            )
            
            if success:
                self.current_url = url
                # Update workflow tools to use new knowledge base
                from . import initialize_tools
                initialize_tools(kb_index, self.kb_manager.embeddings)
                return f"✅ Extracted {len(extracted_content)} chars, {message}"
            else:
                return f"❌ {message}"
                
        except Exception as e:
            return f"❌ Processing failed: {e}"
    
    def ask_question(self, question: str) -> tuple[str, str]:
        """Ask a research question"""
        if not question.strip():
            return "❌ Please enter a question", ""
        
        if not self.current_url:
            return "❌ Please process a URL first", ""
        
        try:
            # Import run_agent locally to avoid circular imports
            from . import run_agent, initialize_agent
            
            # Initialize agent with current components
            kb_index = self.kb_manager.get_index_for_source(self.current_url)
            initialize_agent(self.semantic_cache, kb_index, self.kb_manager.embeddings)
            
            # Run the agent with caching always enabled
            start_time = time.perf_counter()
            result = run_agent(self.workflow_app, question, enable_caching=True)
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Extract answer and metrics
            answer = result.get('final_response', 'No response generated')
            
            # Debug logging
            logger.info(f"Answer extracted: {answer[:100]}...")
            logger.info(f"Result keys: {list(result.keys())}")
            
            # Calculate metrics
            cache_hits = len([h for h in result.get('cache_hits', {}).values() if h])
            total_questions = len(result.get('sub_questions', []))
            total_llm_calls = sum(result.get('llm_calls', {}).values()) if result.get('llm_calls') else 0
            estimated_tokens = total_llm_calls * 150
            
            # Create single-line log entry for this run
            run_number = len(self.run_log) + 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format question preview (first 50 chars)
            question_preview = question[:50] + "..." if len(question) > 50 else question
            
            log_entry = f"#{run_number:2d} | {timestamp} | {total_time:4.0f}ms | {cache_hits}/{total_questions} hits | {total_llm_calls:2d} LLM | ~{estimated_tokens:4d} tokens | Q: {question_preview}"
            
            # Add to running log
            self.run_log.append(log_entry)
            
            # Create scrollable metrics display with header
            metrics_display = "\n".join(reversed(self.run_log))
            
            # Ensure we return proper strings
            return str(answer), str(metrics_display)
            
        except Exception as e:
            return f"❌ Error: {e}", ""
    
    
    def get_status(self) -> str:
        """Get simple status"""
        if self.current_url:
            kb_status = self.kb_manager.get_status()
            kb_count = kb_status['total_indexes']
            return f"✅ Ready | URL: {self.current_url} | KB: {kb_count} loaded"
        else:
            return "⏳ No content loaded"


def create_demo(workflow_app, semantic_cache, tavily_api_key):
    """Create a clean, simple Gradio demo"""
    
    demo_state = ResearchDemo(workflow_app, semantic_cache, tavily_api_key)
    
    # Minimal, no-CSS, default-components demo
    with gr.Blocks(title="Deep Research Agent Demo") as demo:
        gr.Markdown("# Deep Research Agent with Semantic Caching")
        gr.Markdown("Semantic caching builds intelligence progressively as you ask more questions.")

        # Load content
        gr.Markdown("## Load Web Content")
        url_input = gr.Textbox(label="Website URL", placeholder="https://example.com/article", lines=1)
        process_btn = gr.Button("Process URL")
        status_output = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=6)

        # Ask question
        gr.Markdown("## Ask Research Questions")
        question_input = gr.Textbox(
            label="Research Question",
            placeholder="What are the main points discussed in this content?",
            lines=2, max_lines=6
        )
        ask_btn = gr.Button("Ask Question")

        # Answer (plain text)
        gr.Markdown("## Answer")
        answer_output = gr.Textbox(interactive=False, lines=10, max_lines=1000, show_label=False)

        # Performance Log (plain text)
        gr.Markdown("## Performance Log")
        metrics_output = gr.Textbox(interactive=False, lines=10, max_lines=1000, show_label=False)

        # Wire up handlers
        process_btn.click(fn=demo_state.process_url, inputs=[url_input], outputs=[status_output])
        ask_btn.click(fn=demo_state.ask_question, inputs=[question_input], outputs=[answer_output, metrics_output])
        demo.load(fn=demo_state.get_status, outputs=[status_output])
    
    return demo


def launch_demo(workflow_app, semantic_cache, tavily_api_key=None, **kwargs):
    """Launch the simplified demo"""
    # Initialize Tavily API key if not provided
    if tavily_api_key is None:
        tavily_api_key = _get_tavily_api_key()
    
    demo = create_demo(workflow_app, semantic_cache, tavily_api_key)
    
    # Set default height to avoid scrolling in notebooks
    if 'height' not in kwargs:
        kwargs['height'] = 1000
    
    return demo.launch(**kwargs)
