import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn

from cache.wrapper import CacheResults

try:
    import tiktoken
except ImportError:
    tiktoken = None

from cache.vis import plot_metrics, pprint_confusion_matrix

pn.extension()


def load_model_costs() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load model costs from JSON file.

    Returns:
        Dictionary with provider -> model -> {input, output} cost structure
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        costs_file = os.path.join(current_dir, "model_costs.json")

        with open(costs_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to basic OpenAI costs
        return {
            "openai": {
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            }
        }


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken for accurate counting.

    Args:
        text: Text to count tokens for
        model: Model name to determine encoding (defaults to gpt-4o-mini)

    Returns:
        Number of tokens in the text
    """
    if tiktoken is None:
        # Fallback to rough estimation if tiktoken not available
        return len(text.split()) * 1.3

    try:
        # Map model names to tiktoken encodings
        model_encodings = {
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "claude": "cl100k_base",  # Approximation
            "gemini": "cl100k_base",  # Approximation
        }

        # Get the appropriate encoding
        encoding_name = "o200k_base"  # Default for newer models
        for model_prefix, enc in model_encodings.items():
            if model_prefix in model.lower():
                encoding_name = enc
                break

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    except Exception:
        # Fallback to estimation if tiktoken fails
        return int(len(text.split()) * 1.3)


def get_model_cost(provider: str, model: str) -> Dict[str, float]:
    """
    Get cost per 1K tokens for a specific model.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4o-mini")

    Returns:
        Dictionary with "input" and "output" costs per 1K tokens
    """
    costs = load_model_costs()

    if provider in costs and model in costs[provider]:
        return costs[provider][model]

    # Try to find model across all providers
    for p, models in costs.items():
        if model in models:
            return models[model]

    # Fallback to reasonable defaults
    return {"input": 0.001, "output": 0.002}


def _harmonic_mean(a, b):
    if a + b == 0:
        return 0
    return 2 * a * b / (a + b) if (a + b) > 0 else 0


class CacheEvaluator:
    true_labels: List[bool]
    cache_results: List[CacheResults]

    # changes the interpretation of true_labels
    is_from_full_retrieval: bool

    @classmethod
    def from_full_retrieval(cls, true_labels, cache_results) -> "CacheEvaluator":
        return cls(true_labels, cache_results, is_from_full_retrieval=True)

    def __init__(
        self, true_labels, cache_results, is_from_full_retrieval: bool = False
    ):
        self.true_labels = np.array(true_labels)
        self.cache_results = np.array(cache_results)
        self.is_from_full_retrieval = is_from_full_retrieval

    def matches_df(self) -> pd.DataFrame:
        query = [r.query for r in self.cache_results]
        match = [
            r.matches[0].prompt if len(r.matches) > 0 else None
            for r in self.cache_results
        ]
        distance = [
            r.matches[0].vector_distance if len(r.matches) > 0 else None
            for r in self.cache_results
        ]
        true_label = self.true_labels.tolist()

        return pd.DataFrame(
            {
                "query": query,
                "match": match,
                "distance": distance,
                "true_label": true_label,
            }
        )

    def get_metrics(self, distance_threshold: Optional[float] = None):
        # dont apply threshold filtering if no threshold is applied
        T = 1 if distance_threshold is None else distance_threshold

        has_retrieval = np.array(
            [
                len([m for m in it.matches if m.vector_distance < T]) > 0
                for it in self.cache_results
            ]
        )
        true_labels = np.array(self.true_labels)
        if self.is_from_full_retrieval:
            # Switch interpretation of matches that are thresholded out
            true_labels[~has_retrieval] = ~true_labels[~has_retrieval]

        tp = has_retrieval & true_labels
        tn = (~has_retrieval) & true_labels
        fp = has_retrieval & (~true_labels)
        fn = (~has_retrieval) & (~true_labels)

        TP = sum(tp)
        FP = sum(fp)
        FN = sum(fn)
        TN = sum(tn)

        confusion_matrix = np.array([[TN, FP], [FN, TP]])
        cache_hit_rate = (TP + FP) / (TP + FP + FN + TN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1

        return {
            "cache_hit_rate": cache_hit_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * TP / (2 * TP + FP + FN),
            "accuracy": (TP + TN) / (TP + TN + FP + FN),
            "utility": _harmonic_mean(precision, cache_hit_rate),
            "confusion_matrix": confusion_matrix,
            "confusion_mask": np.array([[tn, fp], [fn, tp]]),
        }

    def report_threshold_sweep(
        self,
        metric_to_maximize="f1_score",
        threshold_span=(0, 1),
        num_samples=100,
        metrics_to_plot=[
            "cache_hit_rate",
            "precision",
            "recall",
            "f1_score",
        ],
    ):
        thresholds = []
        all_metrics = {}
        for threshold in np.linspace(*threshold_span, num_samples):
            threshold = threshold.item()
            metrics = self.get_metrics(threshold)
            thresholds.append(threshold)
            for key, metric in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(metric)

        thresholds = np.array(thresholds)
        all_metrics = {key: np.array(value) for key, value in all_metrics.items()}

        best_index = np.argmax(all_metrics[metric_to_maximize])
        best_threshold = thresholds[best_index].item()

        best_metrics_report = self.report_metrics(
            distance_threshold=best_threshold,
            title=f"Best Threshold Metrics @T={best_threshold:.4f}",
            orientation="vertical",
        )

        return pn.Row(
            best_metrics_report,
            pn.pane.Matplotlib(
                plot_metrics(
                    thresholds,
                    all_metrics,
                    best_threshold,
                    metrics_to_plot=metrics_to_plot,
                ),
                format="svg",
                tight=True,
                height=420,
            ),
        )

    def report_metrics(
        self,
        distance_threshold: Optional[float] = None,
        title="Evaluation report",
        orientation="horizontal",
    ):
        metrics = self.get_metrics(distance_threshold)

        metrics_table = (
            pd.DataFrame([metrics])
            .drop(columns=["confusion_matrix", "confusion_mask"])
            .T.rename(columns={0: ""})
        )
        container = pn.Row
        if orientation == "vertical":
            container = pn.Column

        return pn.Column(
            f"### {title}",
            container(
                pn.pane.DataFrame(metrics_table, width=200),
                pprint_confusion_matrix(metrics["confusion_matrix"]),
            ),
        )


class PerfEval:
    def __init__(self):
        self.durations = []  # seconds
        self.durations_by_label: Dict[str, List[float]] = {}
        self.last_time: Optional[float] = None
        self.total_queries: Optional[int] = None
        self.llm_calls: List[Dict] = []  # {model, in_tokens, out_tokens}

    def __enter__(self):
        self.last_time = time.time()
        self.durations = []
        self.durations_by_label = {}
        self.llm_calls = []
        return self

    def start(self):
        self.last_time = time.time()

    def tick(self, label: Optional[str] = None):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
        dt = now - self.last_time
        self.durations.append(dt)
        if label:
            self.durations_by_label.setdefault(label, []).append(dt)
        self.last_time = now

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def set_total_queries(self, n: int):
        self.total_queries = n

    def record_llm_call(
        self, model: str, input_text: str, output_text: str, provider: str = "openai"
    ):
        """
        Record an LLM call with automatic token counting if texts are provided.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            input_text: Input text to count tokens for (optional if input_tokens provided)
            output_text: Output text to count tokens for (optional if output_tokens provided)
            provider: Provider name for cost lookup (default: "openai")
        """
        # Count tokens if not provided
        input_tokens = count_tokens(input_text, model)
        output_tokens = count_tokens(output_text, model)

        # Store the call with provider info for cost calculation
        self.llm_calls.append(
            {
                "model": model,
                "provider": provider,
                "in": input_tokens or 0,
                "out": output_tokens or 0,
            }
        )

    def _stats(self, values: List[float]):
        if len(values) == 0:
            return {
                "count": 0,
                "average_latency": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "average_throughput": 0.0,
            }
        arr = np.array(values, dtype=float)
        total_ms = arr.sum() * 1000.0
        return {
            "count": int(arr.size),
            "average_latency": float(arr.mean() * 1000.0),
            "p50": float(np.percentile(arr, 50) * 1000.0),
            "p90": float(np.percentile(arr, 90) * 1000.0),
            "p95": float(np.percentile(arr, 95) * 1000.0),
            "p99": float(np.percentile(arr, 99) * 1000.0),
            "average_throughput": float(
                (arr.size / (total_ms / 1000.0)) if total_ms > 0 else 0.0
            )
            * 1000.0,
        }

    def get_metrics(self, labels: Optional[List[str]] = None):
        overall = self._stats(self.durations)
        by_label = {}
        if labels:
            for lbl in labels:
                by_label[lbl] = self._stats(self.durations_by_label.get(lbl, []))
        return {"overall": overall, "by_label": by_label}

    def get_costs(self):
        """
        Calculate costs for all LLM calls.

        Returns:
            Dictionary with cost breakdown
        """
        all_costs = load_model_costs()
        total = 0.0
        by_model: Dict[str, float] = {}

        for call in self.llm_calls:
            model = call["model"]
            provider = call.get("provider", "openai")
            rates = get_model_cost(provider, model)

            # Calculate cost (rates are per 1K tokens)
            input_cost = (call["in"] / 1000.0) * rates.get("input", 0.0)
            output_cost = (call["out"] / 1000.0) * rates.get("output", 0.0)
            call_cost = input_cost + output_cost

            by_model[model] = by_model.get(model, 0.0) + call_cost
            total += call_cost

        result = {
            "total_cost": total,
            "by_model": by_model,
            "calls": len(self.llm_calls),
        }

        if self.total_queries:
            result["avg_cost_per_query"] = total / self.total_queries
        if self.llm_calls:
            result["avg_cost_per_call"] = total / len(self.llm_calls)

        return result

    def plot(
        self,
        labels: Optional[List[str]] = None,
        title: str = "Performance Dashboard",
        figsize: tuple = (14, 8),
        show_cost_analysis: bool = True,
    ):
        """
        Create a comprehensive performance dashboard visualization.

        Args:
            labels: List of timing labels to include in metrics (auto-detected if None)
            title: Dashboard title
            figsize: Figure size (width, height)
            show_cost_analysis: Whether to include cost analysis panel
        """
        # Auto-detect available labels if not provided
        if labels is None:
            labels = list(self.durations_by_label.keys())

        # Get metrics and costs
        metrics = self.get_metrics(labels=labels)
        costs = self.get_costs()

        # Set up clean plotting style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10}
        )

        # Create clean 2x2 grid layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        self._plot_hit_miss_distribution(ax1, labels, metrics)
        self._plot_latency_comparison(ax2, labels, metrics)

        if show_cost_analysis and costs["calls"] > 0:
            self._plot_cost_analysis(ax3, costs)
        else:
            ax3.axis("off")

        self._plot_performance_summary(ax4, labels, metrics, costs)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_hit_miss_distribution(self, ax, labels, metrics):
        """Plot cache hit/miss distribution as pie chart."""
        # Look for cache-related labels
        cache_hits = 0
        cache_misses = 0

        # Count actual queries that resulted in hits vs misses
        # Each LLM call represents a cache miss
        for label in labels:
            count = metrics["by_label"].get(label, {}).get("count", 0)

            if "hit" in label.lower():
                cache_hits += count
            elif "llm" in label.lower():
                # LLM calls represent cache misses
                cache_misses += count

        # If we can't detect hits/misses properly, calculate from total queries
        if cache_hits == 0 and cache_misses == 0:
            total_queries = self.total_queries or len(self.durations)
            cache_misses = total_queries

        # Ensure we have some data to plot
        if cache_hits + cache_misses > 0:
            sizes = [cache_hits, cache_misses]
            labels_pie = [
                f"Cache Hits\n({cache_hits})",
                f"Cache Misses\n({cache_misses})",
            ]
            colors = ["#2ecc71", "#e74c3c"]
            explode = (0.02, 0) if cache_hits > 0 else (0, 0.02)

            wedges, texts, autotexts = ax.pie(
                sizes,
                explode=explode,
                labels=labels_pie,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 10},
            )

            # Clean up text styling
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

            ax.set_title("Cache Effectiveness", fontweight="bold", pad=20)
        else:
            ax.text(
                0.5,
                0.5,
                "No hit/miss data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Cache Effectiveness", fontweight="bold")

    def _plot_latency_comparison(self, ax, labels, metrics):
        """Plot latency comparison as clean bar chart."""
        latency_data = []
        latency_labels = []
        colors = []

        # Only show the most relevant metrics for clarity
        relevant_labels = []
        for label in labels:
            if label in metrics["by_label"] and metrics["by_label"][label]["count"] > 0:
                if "hit" in label.lower() or "llm" in label.lower():
                    relevant_labels.append(label)

        for label in relevant_labels:
            latency = metrics["by_label"][label]["average_latency"]
            latency_data.append(latency)

            # Clean up label names
            if "hit" in label.lower():
                latency_labels.append(f"Cache Hit\n{latency:.1f}ms")
                colors.append("#2ecc71")
            elif "llm" in label.lower():
                latency_labels.append(f"LLM Call\n{latency:.1f}ms")
                colors.append("#e74c3c")
            else:
                latency_labels.append(
                    f'{label.replace("_", " ").title()}\n{latency:.1f}ms'
                )
                colors.append("#95a5a6")

        if latency_data:
            bars = ax.bar(latency_labels, latency_data, color=colors, alpha=0.8)
            ax.set_title("Response Time Comparison", fontweight="bold", pad=20)
            ax.set_ylabel("Latency (ms)", fontweight="bold")

            # Clean styling
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_axisbelow(True)

            # Set y-axis to show meaningful scale
            ax.set_ylim(0, max(latency_data) * 1.1)
        else:
            ax.text(
                0.5,
                0.5,
                "No latency data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Response Time Comparison", fontweight="bold")

    def _plot_cost_analysis(self, ax, costs):
        """Plot cost analysis as clean bar chart with integrated labels."""
        cost_data = [costs.get("avg_cost_per_query", 0), costs.get("total_cost", 0)]

        # Format cost values for display
        avg_cost_str = (
            f"${cost_data[0]:.6f}" if cost_data[0] < 0.001 else f"${cost_data[0]:.4f}"
        )
        total_cost_str = (
            f"${cost_data[1]:.6f}" if cost_data[1] < 0.001 else f"${cost_data[1]:.4f}"
        )

        # Create labels with cost values integrated
        cost_labels = [f"Per Query\n{avg_cost_str}", f"Total Cost\n{total_cost_str}"]

        bars = ax.bar(cost_labels, cost_data, color=["#3498db", "#e74c3c"], alpha=0.8)
        ax.set_title("Cost Analysis", fontweight="bold", pad=20)
        ax.set_ylabel("Cost (USD)", fontweight="bold")

        # Clean up the plot
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        # Set y-axis to show meaningful scale
        if max(cost_data) > 0:
            ax.set_ylim(0, max(cost_data) * 1.1)

    def _plot_performance_summary(self, ax, labels, metrics, costs):
        """Plot clean performance summary as text panel."""
        ax.axis("off")

        # Calculate basic stats
        total_queries = self.total_queries or len(self.durations)

        # Detect cache hits and LLM calls
        cache_hits = 0
        llm_calls = 0
        cache_hit_latency = 0
        llm_call_latency = 0

        for label in labels:
            label_data = metrics["by_label"].get(label, {})
            count = label_data.get("count", 0)

            if "hit" in label.lower():
                cache_hits += count
                if cache_hit_latency == 0:
                    cache_hit_latency = label_data.get("average_latency", 0)
            elif "llm" in label.lower():
                llm_calls += count
                if llm_call_latency == 0:
                    llm_call_latency = label_data.get("average_latency", 0)

        # Calculate hit rate and speed improvement
        hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0

        if cache_hits > 0 and llm_calls > 0 and cache_hit_latency > 0:
            speed_improvement = llm_call_latency / cache_hit_latency
            speed_text = f"{speed_improvement:.1f}x faster"
        else:
            speed_text = "N/A"

        # Build clean summary text
        summary_text = f"""Performance Summary

Queries: {total_queries}
Cache Hits: {cache_hits} ({hit_rate:.1f}%)
Cache Misses: {llm_calls}

Average Latency: {metrics['overall']['average_latency']:.1f}ms
Cache Speedup: {speed_text}"""

        if costs["calls"] > 0:
            summary_text += f"""

Total Cost: ${costs['total_cost']:.4f}
Cost/Query: ${costs.get('avg_cost_per_query', 0):.6f}"""

        # Display with clean styling
        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.7",
                facecolor="#f8f9fa",
                edgecolor="#dee2e6",
                linewidth=1,
                alpha=0.9,
            ),
        )
