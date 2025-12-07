from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
import math
from parsec.prompts.analytics import TemplateAnalytics, TemplateMetrics

class TrafficSplitStrategy(str, Enum):
    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    EPSILON_GREEDY = "epsilon_greedy"

@dataclass
class Variant:
    """A variant in an A/B test."""

    template_name: str
    version: str
    weight: float = 1.0  # Used for WEIGHTED strategy

@dataclass
class ABTestResult:
    """Results of an A/B test."""

    winner: Optional[Variant]
    confidence: float
    metrics_by_variant: Dict[str, TemplateMetrics]
    is_significant: bool
    sample_size: int

class ABTest:
    """Manages an A/B test for prompt templates."""

    def __init__(
            self,
            test_name: str,
            variants: List[Variant],
            analytics: TemplateAnalytics,
            strategy: TrafficSplitStrategy = TrafficSplitStrategy.UNIFORM,
            min_sample_size: int = 30,
            significance_level: Optional[float] = 0.05
            ):
        self.test_name = test_name
        self.variants = variants
        self.analytics = analytics
        self.strategy = strategy
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level

    def select_variant(self) -> Variant:
        """Select a variant based on the traffic split strategy."""
        if self.strategy == TrafficSplitStrategy.UNIFORM:
            return random.choice(self.variants)
        elif self.strategy == TrafficSplitStrategy.WEIGHTED:
            total_weight = sum(v.weight for v in self.variants)
            pick = random.uniform(0, total_weight)
            current = 0
            for variant in self.variants:
                current += variant.weight
                if current >= pick:
                    return variant
            return self.variants[-1]  # Fallback
        elif self.strategy == TrafficSplitStrategy.EPSILON_GREEDY:
            epsilon = 0.1  # Exploration rate
            if random.random() < epsilon:
                return random.choice(self.variants)
            else:
                best_variant = None
                best_success_rate = -1.0
                for variant in self.variants:
                    metrics = self.analytics._metrics.get((variant.template_name, variant.version))
                    if metrics and metrics.total_calls > 0:
                        success_rate = metrics.successful_calls / metrics.total_calls
                        if success_rate > best_success_rate:
                            best_success_rate = success_rate
                            best_variant = variant
                return best_variant if best_variant else random.choice(self.variants)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
    def get_results(self) -> ABTestResult:
        """Analyze results and determine the winning variant."""
        metrics_by_variant = {}
        for variant in self.variants:
            key = (variant.template_name, variant.version)
            metrics = self.analytics._metrics.get(key)
            if metrics:
                metrics_by_variant[f"{variant.template_name}:{variant.version}"] = metrics

        winner = None
        confidence = 0.0
        is_significant = False
        sample_size = sum(m.total_calls for m in metrics_by_variant.values())
        
        if sample_size >= self.min_sample_size and len(metrics_by_variant) >= 2:
            # Find best performing variant
            best_variant = None
            best_success_rate = -1.0
            
            for variant in self.variants:
                key = f"{variant.template_name}:{variant.version}"
                if key in metrics_by_variant:
                    metrics = metrics_by_variant[key]
                    if metrics.success_rate > best_success_rate:
                        best_success_rate = metrics.success_rate
                        best_variant = variant
            
            # Perform statistical test between best and second best
            if best_variant and len(self.variants) >= 2:
                best_key = f"{best_variant.template_name}:{best_variant.version}"
                best_metrics = metrics_by_variant[best_key]
                
                # Compare with other variants
                min_p_value = 1.0
                
                for variant in self.variants:
                    variant_key = f"{variant.template_name}:{variant.version}"
                    if variant_key != best_key and variant_key in metrics_by_variant:
                        other_metrics = metrics_by_variant[variant_key]
                        
                        # Calculate z-score
                        z_score = self._calculate_z_score(
                            best_metrics.success_rate,
                            best_metrics.total_calls,
                            other_metrics.success_rate,
                            other_metrics.total_calls
                        )
                        
                        # Convert to p-value
                        p_value = self._z_to_p_value(z_score)
                        min_p_value = min(min_p_value, p_value)
                
                # Check if statistically significant
                is_significant = min_p_value < self.significance_level
                confidence = 1.0 - min_p_value if is_significant else 0.0
                winner = best_variant if is_significant else None

        return ABTestResult(
            winner=winner,
            confidence=confidence,
            metrics_by_variant=metrics_by_variant,
            is_significant=is_significant,
            sample_size=sample_size
        )
    
    def _calculate_z_score(self, p1: float, n1: int, p2: float, n2: int) -> float:
        """Calculate the Z-score for two proportions."""
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se == 0:
            return 0.0
        return abs((p1 - p2) / se)
        
    def _z_to_p_value(self, z: float) -> float:
        """Convert Z-score to p-value."""
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return p_value
    
