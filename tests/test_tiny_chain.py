"""Tests for tiny-chain."""
import time
import pytest
from tiny_chain import TinyChain, ChainStep, ChainSnapshot


class TestChainStep:
    def test_step_to_dict(self):
        step = ChainStep(name="test", thought="doing test", output="result", duration_ms=5.0)
        d = step.to_dict()
        assert d["name"] == "test"
        assert d["thought"] == "doing test"
        assert d["output"] == "result"
        assert d["duration_ms"] == 5.0
        assert d["error"] is None
        assert d["tags"] == []

    def test_step_with_error(self):
        step = ChainStep(name="fail", thought="boom", error="RuntimeError: oops", duration_ms=1.0)
        assert step.error == "RuntimeError: oops"
        assert "RuntimeError" in step.error


class TestTinyChain:
    def test_basic_usage(self):
        chain = TinyChain("test-run")

        @chain.step("Fetch", "Getting data from API")
        def fetch():
            return {"price": 65000}

        @chain.step("Analyze", "Computing trend")
        def analyze(data):
            return "bullish"

        data = fetch()
        result = analyze(data)

        assert len(chain.steps) == 2
        assert chain.steps[0].name == "Fetch"
        assert chain.steps[0].output == {"price": 65000}
        assert chain.steps[1].name == "Analyze"
        assert chain.steps[1].output == "bullish"
        assert not chain.has_errors()

    def test_decorator_preserves_return_value(self):
        chain = TinyChain()

        @chain.step("Step1", "step one")
        def step1():
            return 42

        @chain.step("Step2", "step two")
        def step2(x):
            return x * 2

        assert step1() == 42
        assert step2(21) == 42

    def test_decorator_records_error(self):
        chain = TinyChain()

        @chain.step("Risky", "attempting risky thing")
        def risky():
            raise ValueError("something went wrong")

        with pytest.raises(ValueError, match="something went wrong"):
            risky()

        assert len(chain.steps) == 1
        assert chain.steps[0].error is not None
        assert "ValueError" in chain.steps[0].error
        assert chain.has_errors()

    def test_span_context_manager(self):
        chain = TinyChain()

        with chain.span("Compute", "doing math"):
            pass

        assert len(chain.steps) == 1
        assert chain.steps[0].name == "Compute"
        assert chain.steps[0].duration_ms > 0

    def test_span_captures_exception(self):
        chain = TinyChain()

        with pytest.raises(RuntimeError):
            with chain.span("Boom", "testing error handling"):
                raise RuntimeError("kaboom")

        assert len(chain.steps) == 1
        assert chain.steps[0].error is not None
        assert "kaboom" in chain.steps[0].error

    def test_add_step_manual(self):
        chain = TinyChain()
        chain.add_step("Manual", "I did this myself", output="done", duration_ms=2.5, tags=["manual"])
        assert len(chain.steps) == 1
        assert chain.steps[0].output == "done"
        assert chain.steps[0].duration_ms == 2.5
        assert chain.steps[0].tags == ["manual"]

    def test_tags_and_metadata(self):
        chain = TinyChain("tagged", tags=["production", "v2"])

        @chain.step("Task", "thinking", tags=["reasoning"], metadata={"model": "gpt-4o"})
        def task():
            return "done"

        task()

        assert chain.tags == ["production", "v2"]
        assert chain.steps[0].tags == ["reasoning"]
        assert chain.steps[0].metadata == {"model": "gpt-4o"}

    def test_serialization_full(self):
        chain = TinyChain("serialize-me", tags=["test"])

        @chain.step("A", "alpha")
        def a():
            return 1

        a()
        d = chain.to_dict()

        assert d["name"] == "serialize-me"
        assert d["version"] == "0.2.0"
        assert d["tags"] == ["test"]
        assert len(d["steps"]) == 1
        assert d["errors"] == []

    def test_compact_dict(self):
        chain = TinyChain("compact")
        chain.add_step("X", "quick", output="y")
        d = chain.to_compact_dict()
        assert "id" in d
        assert len(d["id"]) == 8
        assert d["steps"][0]["n"] == "X"
        assert "total_ms" in d

    def test_score_basic(self):
        chain = TinyChain()
        chain.add_step("fetch", "getting data")
        chain.add_step("analyze", "thinking hard")

        scores = chain.score({
            "has_steps": lambda s: min(1.0, len(s) / 2),
            "named_fetch": lambda s: 1.0 if any("fetch" in x.name for x in s) else 0.0,
        })
        assert scores["has_steps"] == 1.0
        assert scores["named_fetch"] == 1.0

    def test_grade_pass(self):
        chain = TinyChain()
        chain.add_step("a", "b")
        passed, avg = chain.grade({"quality": 1.0, "completeness": 1.0}, threshold=0.5)
        assert passed is True
        assert avg == 1.0

    def test_grade_fail(self):
        chain = TinyChain()
        chain.add_step("a", "b")
        # Rubric requires step named 'fetch' but chain has no such step -> score 0 -> fails
        passed, avg = chain.grade({
            "has_fetch": lambda s: 1.0 if any("fetch" in x.name for x in s) else 0.0
        }, threshold=0.9)
        assert passed is False
        assert avg == 0.0

    def test_snapshot(self):
        chain = TinyChain("snap-test")
        chain.add_step("s1", "first", duration_ms=10.0)
        chain.add_step("s2", "second", duration_ms=20.0)
        snap = chain.snapshot()
        assert snap.chain_name == "snap-test"
        assert snap.steps_completed == 2
        assert len(snap.step_names) == 2

    def test_slowest_step(self):
        chain = TinyChain()
        chain.add_step("fast", "quick", duration_ms=1.0)
        chain.add_step("slow", "takes time", duration_ms=100.0)
        slowest = chain.slowest_step()
        assert slowest is not None
        assert slowest.name == "slow"

    def test_summary(self):
        chain = TinyChain("my-chain")
        chain.add_step("x", "y")
        s = chain.summary()
        assert "my-chain" in s
        assert "1 steps" in s

    def test_errors_method(self):
        chain = TinyChain()
        chain.add_step("ok", "fine")
        chain.add_step("bad", "oops", error="ValueError: bad")
        # errors() filters out None entries
        assert chain.errors() == ["ValueError: bad"]

    def test_repr(self):
        chain = TinyChain("repr-test", tags=["t1"])
        assert repr(chain) == chain.summary()

    def test_export_and_from_dict(self):
        chain = TinyChain("export-me", tags=["exported"])
        chain.add_step("Step1", "first")
        d = chain.to_dict()
        reconstructed = TinyChain.from_dict(d)
        assert reconstructed.name == "export-me"
        assert reconstructed.tags == ["exported"]
