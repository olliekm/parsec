import asyncio
import time
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass

from parsec.logging import get_logger

class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # in seconds

class CircuitBreakerError(Exception):
    """Raised when the circuit is open and calls are not allowed."""
    pass

class CircuitBreaker:
    """
    Docstring for CircuitBreaker
    """

    def __init__(
            self,
            name: str,
            config: Optional[CircuitBreakerConfig] = None,
        ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.logger = get_logger(__name__)
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.logger.info(f"CircuitBreaker {self.name} entering HALF_OPEN")
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try HALF_OPEN."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.timeout
    
    def _reset(self):
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    async def _on_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.logger.info(f"CircuitBreaker {self.name} reopening from HALF_OPEN")
                self.state = CircuitBreakerState.OPEN
            elif self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.logger.warning(f"CircuitBreaker {self.name} opening from CLOSED")
                    self.state = CircuitBreakerState.OPEN

    async def _on_success(self):
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.logger.info(f"CircuitBreaker {self.name} closing from HALF_OPEN")
                    self._reset()
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # reset failure count on success

    async def reset(self):
        """Manually reset the circuit breaker."""
        async with self._lock:
            self.logger.info(f"CircuitBreaker {self.name} manually reset")
            self._reset()

    def get_state(self) -> dict:
        """Get current state for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }