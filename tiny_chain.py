"""
tiny-chain — Zero-dependency structured chain-of-thought for AI agents.

Records, structures, and exports the "reasoning steps" of an AI agent
as machine-readable JSON traces with timing, metadata, and evaluation hooks.

MIT License | https://github.com/hussain-alsaibai/tiny-chain
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, List, Optional, TypeVar

__version__ = "0.2.0"
__all__ = ["TinyChain", "ChainStep", "ChainSnapshot"]

T = TypeVar("T")


@dataclass
class ChainStep:
    """A single reasoning step in a chain-of-thought."""

    name: str
    thought: str
    output: Any = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChainSnapshot:
    """Point-in-time snapshot of a chain's progress for evaluation."""

    chain_id: str
    chain_name: str
    steps_completed: int
    total_duration_ms: float
    step_names: list[str]
    step_thoughts: list[str]
    step_durations_ms: list[float]
    tags: list[str]
    timestamp: float

    @classmethod
    def from_chain(cls, chain: "TinyChain") -> "ChainSnapshot":
        return cls(
            chain_id=chain.id,
            chain_name=chain.name,
            steps_completed=len(chain.steps),
            total_duration_ms=chain.total_duration_ms,
            step_names=[s.name for s in chain.steps],
            step_thoughts=[s.thought for s in chain.steps],
            step_durations_ms=[s.duration_ms for s in chain.steps],
            tags=chain.tags,
            timestamp=time.time(),
        )


class TinyChain:
    """
    Zero-dependency structured chain-of-thought for AI agents.

    Transforms messy internal monologues into clean, machine-readable JSON
    traces with timing, metadata, and evaluation hooks.

    Example:
        chain = TinyChain("market-analysis")

        @chain.step("Fetch Data", "Fetching BTC price from API")
        def get_price():
            return 65000

        price = get_price()
        print(chain.to_json())
    """

    def __init__(
        self,
        name: str = "default",
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.tags: list[str] = tags or []
        self.metadata: dict[str, Any] = metadata or {}
        self.steps: list[ChainStep] = []
        self._start_time = time.time()
        self._current_step_start: Optional[float] = None

    def step(
        self,
        name: str,
        thought: str,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator to mark a function as a reasoning step.

        Args:
            name: Human-readable label for this step.
            thought: The reasoning behind this step (shown in traces/logs).
            tags: Optional classification tags.
            metadata: Optional structured data attached to the step.

        Returns:
            A decorator that wraps the function.
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                step_start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - step_start) * 1000
                    self.steps.append(
                        ChainStep(
                            name=name,
                            thought=thought,
                            output=result,
                            duration_ms=duration,
                            metadata=metadata or {},
                            tags=tags or [],
                        )
                    )
                    return result
                except Exception as exc:
                    duration = (time.time() - step_start) * 1000
                    self.steps.append(
                        ChainStep(
                            name=name,
                            thought=thought,
                            duration_ms=duration,
                            metadata=metadata or {},
                            tags=tags or [],
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    raise

            return wrapper  # type: ignore[return-type]
        return decorator

    @contextmanager
    def span(
        self,
        name: str,
        thought: str,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator[Any]:
        """
        Context manager for a reasoning step (when you can't use a decorator).

        Example:
            with chain.span("Analyze", "Computing trend") as result:
                result = compute_trend(prices)
        """
        step_start = time.time()
        result = None
        error: Optional[str] = None
        try:
            yield result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration = (time.time() - step_start) * 1000
            self.steps.append(
                ChainStep(
                    name=name,
                    thought=thought,
                    output=result,
                    duration_ms=duration,
                    metadata=metadata or {},
                    tags=tags or [],
                    error=error,
                )
            )

    def add_step(
        self,
        name: str,
        thought: str,
        output: Any = None,
        *,
        duration_ms: float = 0.0,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Manually record a reasoning step.

        Use this when the step logic is spread across multiple function
        calls or can't be decorated cleanly.
        """
        self.steps.append(
            ChainStep(
                name=name,
                thought=thought,
                output=output,
                duration_ms=duration_ms,
                metadata=metadata or {},
                tags=tags or [],
                error=error,
            )
        )

    # ── Evaluation / grading ──────────────────────────────────────────────

    def score(
        self,
        criteria: dict[str, Callable[[list[ChainStep]], float]],
    ) -> dict[str, float]:
        """
        Score the chain against named criteria.

        Args:
            criteria: Map of name -> scoring function. Function receives the
                     list of steps and returns a float in [0, 1].

        Returns:
            Dict of criterion name -> score.

        Example:
            chain.score({
                "has_fetch": lambda s: 1.0 if any("fetch" in x.name.lower() for x in s) else 0.0,
                "no_errors":  lambda s: 0.0 if any(x.error for x in s) else 1.0,
            })
        """
        return {name: fn(self.steps) for name, fn in criteria.items()}

    def grade(self, rubric: dict[str, float], threshold: float = 0.7) -> tuple[bool, float]:
        """
        Grade the chain against a weighted rubric.

        Args:
            rubric: Dict of criterion name -> max points (float) or scoring
                    function. Functions receive list[ChainStep] and return
                    float in [0, 1]. Float values are treated as max points
                    with a score of 1.0 (binary pass/fail on presence).
            threshold: Minimum average score (0–1) to pass.

        Returns:
            (passed, average_score)
        """
        score_args = {}
        for k, v in rubric.items():
            if callable(v):
                score_args[k] = v
            else:
                # float = max points, binary presence check (1.0 if present)
                score_args[k] = lambda s, _max=v: 1.0

        raw_scores = self.score(score_args)
        if not raw_scores:
            return True, 1.0

        total = 0.0
        max_possible = 0.0
        for k, v in rubric.items():
            s = raw_scores.get(k, 0.0)
            if callable(v):
                total += s
                max_possible += 1.0
            else:
                total += s * v
                max_possible += v

        avg = total / max_possible if max_possible else 1.0
        return avg >= threshold, avg

    def snapshot(self) -> ChainSnapshot:
        """Return a point-in-time snapshot for async/eval workflows."""
        return ChainSnapshot.from_chain(self)

    # ── Serialization ─────────────────────────────────────────────────────

    @property
    def total_duration_ms(self) -> float:
        return (time.time() - self._start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": __version__,
            "tags": self.tags,
            "metadata": self.metadata,
            "total_duration_ms": self.total_duration_ms,
            "steps": [s.to_dict() for s in self.steps],
            "errors": [s.error for s in self.steps if s.error],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_compact_dict(self) -> dict[str, Any]:
        """Minimal dict — good for logging."""
        return {
            "id": self.id[:8],
            "name": self.name,
            "steps": [
                {"n": s.name, "t": s.thought, "ms": round(s.duration_ms, 1)}
                for s in self.steps
            ],
            "total_ms": round(self.total_duration_ms, 1),
        }

    def to_compact_json(self) -> str:
        return json.dumps(self.to_compact_dict())

    def export(self, path: str) -> None:
        """Write the full trace to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyChain":
        """Reconstruct a chain from a serialized dict."""
        chain = cls(name=data.get("name", "reconstructed"), tags=data.get("tags", []))
        chain.id = data.get("id", chain.id)
        chain.metadata = data.get("metadata", {})
        # steps are not replayed; chain is returned in a "completed" state
        return chain

    # ── Introspection ──────────────────────────────────────────────────────

    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    def errors(self) -> list[str]:
        return [s.error for s in self.steps if s.error]

    def has_errors(self) -> bool:
        return any(s.error for s in self.steps)

    def slowest_step(self) -> Optional[ChainStep]:
        if not self.steps:
            return None
        return max(self.steps, key=lambda s: s.duration_ms)

    def summary(self) -> str:
        """One-line summary: name, steps, duration, errors."""
        err = f" ({len(self.errors())} errors)" if self.has_errors() else ""
        return (
            f"TinyChain({self.name}): {len(self.steps)} steps, "
            f"{self.total_duration_ms:.0f}ms{err}"
        )

    def __repr__(self) -> str:
        return self.summary()
