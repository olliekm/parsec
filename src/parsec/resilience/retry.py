from dataclasses import dataclass
from enum import Enum
from typing import Optional

class OperationType(str, Enum):
    GENERATION = "generation"
    VALIDATION = "validation"
    REPAIR = "repair"

@dataclass
class RetryPolicy:
    max_attempts: int
    base_delay: float  # in seconds
    max_delay: float  # in seconds
    retryable_exceptions: tuple = (Exception,)
    timeout: Optional[float] = None  # in seconds

    def is_retryable(self, exception: Exception) -> bool:
        """Check if the exception is retryable."""
        return isinstance(exception, self.retryable_exceptions)

DEFAULT_POLICIES = {
    OperationType.GENERATION: RetryPolicy(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=(TimeoutError, ConnectionError, OSError),
        timeout=120.0  # 2 minutes for LLM calls
    ),
    OperationType.VALIDATION: RetryPolicy(
        max_attempts=1,  # Don't retry validation
        base_delay=0.1,
        max_delay=1.0,
        retryable_exceptions=(),
        timeout=5.0
    )
}

def get_retry_policy(operation: OperationType) -> RetryPolicy:
    """Retrieve the retry policy for a given operation type."""
    return DEFAULT_POLICIES.get(
        operation, 
        RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            retryable_exceptions=(Exception,),
            timeout=60.0
        )
    )