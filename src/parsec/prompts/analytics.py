from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import statistics

@dataclass
class TemplateMetrics:
    template_name: str
    version: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0 
    retry_counts: List[int] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Success rate as a float between 0 and 1."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def average_latency_ms(self) -> float:
        """Average latency for all calls in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls
    
    @property
    def average_tokens(self) -> float:
        """Average tokens used per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_tokens / self.total_calls
    
    @property
    def average_retries(self) -> float:
        """Average number of retries per call."""
        if not self.retry_counts:
            return 0.0
        return statistics.mean(self.retry_counts)
    
    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(0.95 * len(sorted_latencies))
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(0.99 * len(sorted_latencies))
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
        
    def get_error_breakdown(self) -> Dict[str, int]:
        """Get a breakdown of validation errors."""
        from collections import Counter
        error_count = Counter(self.validation_errors)
        return dict(error_count)
    
    
class TemplateAnalytics:
    """Track performace metrics for prompt templates."""

    def __init__(self):
        self._metrics: Dict[tuple, TemplateMetrics] = {}

    def record_result(
        self,
        template_name: str,
        version: str,
        success: bool,
        tokens_used: int,
        latency_ms: float,
        retry_count: int,
        validation_errors: Optional[List[str]] = None
    ) -> None:
        key = (template_name, version)

        if key not in self._metrics:
            self._metrics[key] = TemplateMetrics(template_name, version)
        
        metric = self._metrics[key]
        metric.total_calls += 1

        if success:
            metric.successful_calls += 1
        else:
            metric.failed_calls += 1
        
        metric.total_tokens += tokens_used
        metric.total_latency_ms += latency_ms
        metric.retry_counts.append(retry_count)
        metric.timestamps.append(datetime.now())
        metric.latencies.append(latency_ms)

        if validation_errors:
            metric.validation_errors.extend(validation_errors)

    def get_metrics(self, template_name: str, version: Optional[str] = None) -> Optional[TemplateMetrics]:
        if version:
            return self._metrics.get((template_name, version))
        
        matching = [
            m for (name, ver), m in self._metrics.items() if name == template_name
        ]

        if not matching:
            return None
        
        aggregated = TemplateMetrics(template_name, "all_versions")
        for metric in matching:
            aggregated.total_calls += metric.total_calls
            aggregated.successful_calls += metric.successful_calls
            aggregated.failed_calls += metric.failed_calls
            aggregated.total_tokens += metric.total_tokens
            aggregated.total_latency_ms += metric.total_latency_ms
            aggregated.retry_counts.extend(metric.retry_counts)
            aggregated.validation_errors.extend(metric.validation_errors)
            aggregated.timestamps.extend(metric.timestamps)
        
        return aggregated
    
    def get_all_metrics(self) -> Dict[str, TemplateMetrics]:
        """Get metrics for all templates and versions."""
        return {
            f"{name}:{version}": metrics
            for (name, version), metrics in self._metrics.items()
        }
    
    def compare_versions(self, template_name: str) -> Dict[str, TemplateMetrics]:
        """Compare metrics across all versions of a template."""
        return {
            version: metrics
            for (name, version), metrics in self._metrics.items()
            if name == template_name
        }
    
    def get_best_performing_version(self, template_name: str, metric: str = "success_rate") -> Optional[TemplateMetrics]:
        versions = self.compare_versions(template_name)
        if not versions:
            return None
        
        higher_is_better = {"success_rate"}
        
        best_version = None
        best_value = float('-inf') if metric in higher_is_better else float('inf')
        
        for version, metrics in versions.items():
            value = getattr(metrics, metric, None)
            if value is None:
                continue
            
            if metric in higher_is_better:
                if value > best_value:
                    best_value = value
                    best_version = metrics
            else:
                if value < best_value:
                    best_value = value
                    best_version = metrics
        
        return best_version