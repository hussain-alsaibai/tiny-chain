"""Tests for tiny-chain. Stdlib only."""

import io
import json
import math
import re
import time
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

import tiny_chain as tc
from tiny_chain import (
    Chain,
    Function,
    FunctionCall,
    LLMResponse,
    TokenUsage,
    _cost_per_1k_tokens,
    json_extractor,
    structured_output,
)


# =============================================================================
# Helpers: fake HTTP body builders
# =============================================================================

def make_sse_response(chunks):
    """Build a fake streaming HTTP response (urllib style) from a list of (role, content) tuples."""
    body = io.BytesIO()
    for role, content in chunks:
        if content is None:
            payload = {"choices": [{"delta": {}, "finish_reason": role}]}
        else:
            payload = {
                "choices": [
                    {"delta": {"role": role, "content": content}, "finish_reason": None}
                ]
            }
        body.write(f"data: {json.dumps(payload)}\n\n".encode())
    body.write(b"data: [DONE]\n\n")
    body.seek(0)
    return body


def make_full_response(content, model="gpt-4o-mini", prompt_tokens=5, completion_tokens=5,
                       tool_calls=None, finish_reason="stop"):
    payload = {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return json.dumps(payload).encode()


class FakeHTTPResponse:
    """Mimics urllib's addinfourl-like object."""

    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self._body = body

    def getcode(self):
        return self.status

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._body.splitlines(keepends=True))


# Local re-export to keep this file's imports tidy
import urllib.error as urllib_error_module  # noqa: E402
urllib_error_HTTPError = urllib_error_module.HTTPError



# =============================================================================
# Orchestration: Pipeline, Parallel, If, retry
# =============================================================================

class TestOrchestration(unittest.TestCase):

    def test_pipeline_sequential(self):
        def step1(ctx): return {"a": ctx.get("input", 0) + 1}
        def step2(ctx): return {"b": ctx["a"] * 2}
        
        pipe = tc.Pipeline([step1, step2])
        ctx = pipe.run({"input": 10})
        self.assertEqual(ctx["a"], 11)
        self.assertEqual(ctx["b"], 22)

    def test_pipeline_hooks(self):
        events = []
        def hook(ctx): events.append("hook")
        def step(ctx): return {"done": True}
        
        pipe = tc.Pipeline([step])
        pipe.on("before", hook)
        pipe.on("after", hook)
        pipe.run({})
        self.assertEqual(events, ["hook", "hook"])

    def test_pipeline_error_hook(self):
        errors = []
        def err_hook(e, ctx): errors.append(e)
        def step(ctx): raise ValueError("fail")
        
        pipe = tc.Pipeline([step])
        pipe.on("error", err_hook)
        with self.assertRaises(ValueError):
            pipe.run({})
        self.assertEqual(len(errors), 1)
        self.assertEqual(str(errors[0]), "fail")

    def test_parallel_execution(self):
        def step1(ctx): 
            time.sleep(0.01)
            return {"p1": 1}
        def step2(ctx): 
            time.sleep(0.01)
            return {"p2": 2}
        
        par = tc.Parallel([step1, step2])
        ctx = par.run({})
        self.assertEqual(ctx["p1"], 1)
        self.assertEqual(ctx["p2"], 2)

    def test_if_branching(self):
        def step_true(ctx): return {"val": "true"}
        def step_false(ctx): return {"val": "false"}
        
        cond_true = tc.If(lambda c: c["x"] > 0, step_true, step_false)
        cond_false = tc.If(lambda c: c["x"] <= 0, step_true, step_false)
        
        self.assertEqual(cond_true.run({"x": 10})["val"], "true")
        self.assertEqual(cond_true.run({"x": -1})["val"], "false")

    def test_retry_decorator(self):
        calls = []
        @tc.retry(max_attempts=3, delay=0.001)
        def fail_twice(ctx):
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return {"ok": True}
            
        res = fail_twice({})
        self.assertEqual(len(calls), 3)
        self.assertTrue(res["ok"])

    def test_retry_exhaustion(self):
        calls = []
        @tc.retry(max_attempts=2, delay=0.001)
        def always_fail(ctx):
            calls.append(1)
            raise ValueError("perm")
            
        with self.assertRaises(ValueError):
            always_fail({})
        self.assertEqual(len(calls), 2)

    def test_pipeline_step_objects(self):
        class MyStep:
            def run(self, ctx):
                return {"obj": True}
        
        pipe = tc.Pipeline([MyStep()])
        ctx = pipe.run({})
        self.assertTrue(ctx["obj"])

# =============================================================================
# JSON extraction tests
# =============================================================================

class TestJsonExtraction(unittest.TestCase):

    def test_extract_plain_json(self):
        data, err = json_extractor.extract_json('{"a": 1, "b": "hello"}')
        self.assertIsNone(err)
        self.assertEqual(data, {"a": 1, "b": "hello"})

    def test_extract_from_markdown_fence(self):
        text = 'Sure, here:\n```json\n{"city": "Tokyo", "pop": 14000000}\n```\nDone.'
        data, err = json_extractor.extract_json(text)
        self.assertIsNone(err)
        self.assertEqual(data["city"], "Tokyo")
        self.assertEqual(data["pop"], 14000000)

    def test_extract_from_bare_fence(self):
        text = '```\n{"x": 42}\n```'
        data, err = json_extractor.extract_json(text)
        self.assertIsNone(err)
        self.assertEqual(data, {"x": 42})

    def test_extract_partial_truncated_object(self):
        # Truncated mid-key — should still recover something usable
        text = '{"a": 1, "b": "hello'
        data, err = json_extractor.extract_json(text)
        self.assertIsNotNone(data)
        self.assertEqual(data.get("a"), 1)

    def test_extract_partial_missing_close(self):
        text = '{"a": 1, "b": "hello"'
        data, err = json_extractor.extract_json(text)
        self.assertIsNotNone(data)
        self.assertEqual(data.get("a"), 1)
        self.assertEqual(data.get("b"), "hello")

    def test_extract_partial_with_array(self):
        text = '[1, 2, 3'
        data, err = json_extractor.extract_json(text)
        self.assertIsNotNone(data)
        self.assertEqual(data, [1, 2, 3])

    def test_extract_first_object_in_paragraph(self):
        text = 'The answer is {"x": 42}. End of story.'
        data, err = json_extractor.extract_json(text)
        self.assertEqual(data, {"x": 42})

    def test_extract_no_json(self):
        data, err = json_extractor.extract_json("just words")
        self.assertIsNone(data)
        self.assertIsNotNone(err)

    def test_extract_empty_input(self):
        data, err = json_extractor.extract_json("")
        self.assertIsNone(data)
        self.assertIsNotNone(err)

    def test_extract_ignores_extra_closers(self):
        # More closes than opens should not crash
        text = '}}}{{{"a":1}'
        data, err = json_extractor.extract_json(text)
        # Should either recover or return None with error
        if data is not None:
            self.assertIsInstance(data, dict)


class TestSchemaValidation(unittest.TestCase):

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age"],
    }

    def test_valid_schema(self):
        text = '{"name": "Ada", "age": 36, "tags": ["math"]}'
        data, err = json_extractor.extract_schema(text, self.schema)
        self.assertIsNone(err)
        self.assertEqual(data["name"], "Ada")

    def test_missing_required(self):
        text = '{"age": 36}'
        data, err = json_extractor.extract_schema(text, self.schema)
        self.assertIsNone(data)
        self.assertIn("name", err)

    def test_wrong_type(self):
        text = '{"name": "Ada", "age": "thirty-six"}'
        data, err = json_extractor.extract_schema(text, self.schema)
        self.assertIsNone(data)

    def test_age_below_minimum(self):
        text = '{"name": "Ada", "age": -1}'
        data, err = json_extractor.extract_schema(text, self.schema)
        self.assertIsNone(data)
        self.assertIn("minimum", err)

    def test_age_above_maximum(self):
        text = '{"name": "Methuselah", "age": 1000}'
        data, err = json_extractor.extract_schema(text, self.schema)
        self.assertIsNone(data)
        self.assertIn("maximum", err)

    def test_array_items_wrong_type(self):
        text = '{"name": "Ada", "age": 36, "tags": [1, 2, 3]}'
        data, err = json_extractor.extract_schema(text, self.schema)
        self.assertIsNone(data)

    def test_enum_constraint(self):
        s = {"type": "object", "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}}}
        ok, err = json_extractor.extract_schema('{"color": "red"}', s)
        bad, err2 = json_extractor.extract_schema('{"color": "purple"}', s)
        self.assertIsNone(err)
        self.assertIsNone(bad)
        self.assertIsNotNone(err2)
        self.assertIn("enum", err2)

    def test_pattern_string(self):
        s = {"type": "object", "properties": {"id": {"type": "string", "pattern": "^[a-z]+$"}}}
        data, err = json_extractor.extract_schema('{"id": "abc"}', s)
        self.assertIsNone(err)
        bad, err2 = json_extractor.extract_schema('{"id": "AB1"}', s)
        self.assertIsNone(bad)


# =============================================================================
# Cost estimation
# =============================================================================

class TestCostEstimation(unittest.TestCase):

    def test_gpt4o_cost(self):
        chain = Chain(api_key="sk-test")
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = chain._estimate_cost("gpt-4o", usage)
        # 1M input @ $2.50 + 500K output @ $10.00 = 2.50 + 5.00 = 7.50
        self.assertAlmostEqual(cost, 7.50, places=4)

    def test_gpt4o_mini_cost(self):
        chain = Chain(api_key="sk-test")
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        cost = chain._estimate_cost("gpt-4o-mini", usage)
        # 0.15 + 0.60 = 0.75
        self.assertAlmostEqual(cost, 0.75, places=4)

    def test_unknown_model_uses_default(self):
        chain = Chain(api_key="sk-test")
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        cost = chain._estimate_cost("some-future-model-9000", usage)
        # default = (1.00, 2.00), so 1M * 1.00 + 1M * 2.00 = 3.00
        self.assertAlmostEqual(cost, 3.00, places=4)

    def test_cost_table_known_models(self):
        # Sanity: our cost table covers the headline models
        for name in ("gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "gpt-3.5-turbo"):
            self.assertIn(name, _cost_per_1k_tokens)
        self.assertIn("default", _cost_per_1k_tokens)


# =============================================================================
# Function call parsing
# =============================================================================

class TestFunctionCallParsing(unittest.TestCase):

    def setUp(self):
        self.chain = Chain(api_key="sk-test")

    def test_parse_explicit_json_tool_call_response(self):
        # OpenAI emits function_call via tool_calls — but our content-based fallback
        # should recover from a JSON tool-call-shaped payload inside content.
        content = json.dumps({"name": "get_weather", "arguments": {"location": "Tokyo"}})
        fc = self.chain._parse_function_call(content)
        self.assertIsNotNone(fc)
        self.assertEqual(fc.name, "get_weather")
        self.assertEqual(fc.arguments, {"location": "Tokyo"})

    def test_parse_function_call_in_fence(self):
        content = "I'll call a function.\n```json\n" + json.dumps({
            "name": "lookup",
            "arguments": {"q": "weather"},
        }) + "\n```\n"
        fc = self.chain._parse_function_call(content)
        self.assertIsNotNone(fc)
        self.assertEqual(fc.name, "lookup")
        self.assertEqual(fc.arguments, {"q": "weather"})

    def test_no_function_call(self):
        fc = self.chain._parse_function_call("Just a regular reply with no tools.")
        self.assertIsNone(fc)


class TestCallFunction(unittest.TestCase):

    def test_call_function_executes(self):
        def add(a, b):
            return a + b

        response = LLMResponse(
            content="",
            function_call=FunctionCall(name="add", arguments={"a": 2, "b": 3}, parsed=True),
            usage=TokenUsage(),
            model="gpt-4o-mini",
            is_complete=True,
            is_function_call=True,
        )
        result = Chain(api_key="sk-test").call_function(response, {"add": add})
        self.assertEqual(result, 5)

    def test_call_function_missing_raises(self):
        response = LLMResponse(
            function_call=FunctionCall(name="nope", arguments={}),
            is_complete=True,
            is_function_call=True,
        )
        with self.assertRaises(KeyError):
            Chain(api_key="sk-test").call_function(response, {})

    def test_call_function_no_call_raises(self):
        response = LLMResponse(content="hi", is_complete=True)
        with self.assertRaises(ValueError):
            Chain(api_key="sk-test").call_function(response, {})


# =============================================================================
# Chain HTTP behavior — mocked responses
# =============================================================================

class TestChainHTTP(unittest.TestCase):

    def setUp(self):
        self.chain = Chain(api_key="sk-test", model="gpt-4o-mini", max_retries=2, base_delay=0.01)

    def test_complete_non_streaming_ok(self):
        body = make_full_response("Hello, world.")
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(200, body=body)):
            resp = self.chain.complete("Say hi")
        self.assertIsNone(resp.error)
        self.assertEqual(resp.content, "Hello, world.")
        self.assertTrue(resp.is_complete)
        self.assertEqual(resp.usage.total_tokens, 10)

    def test_complete_handles_400_raises(self):
        err_body = json.dumps({"error": {"message": "bad request"}}).encode()
        http_err = urllib_error_HTTPError(
            url="http://x", code=400, msg="Bad Request", hdrs={}, fp=io.BytesIO(err_body),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with self.assertRaises(tc.BadRequestError):
                self.chain.complete("test")

    def test_complete_handles_401_raises(self):
        err_body = json.dumps({"error": {"message": "invalid api key"}}).encode()
        http_err = urllib_error_HTTPError(
            url="http://x", code=401, msg="Unauthorized", hdrs={}, fp=io.BytesIO(err_body),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with self.assertRaises(tc.AuthError):
                self.chain.complete("test")

    def test_complete_retries_429_then_succeeds(self):
        # First call: 429 with Retry-After; second: 200
        body = make_full_response("after backoff")
        responses = [
            urllib_error_HTTPError(
                url="http://x", code=429, msg="Too Many Requests",
                hdrs={"Retry-After": "0"}, fp=io.BytesIO(b'{"error":{"message":"slow down"}}'),
            ),
            FakeHTTPResponse(200, body=body),
        ]
        with patch("urllib.request.urlopen", side_effect=responses), \
             patch("time.sleep") as mock_sleep:
            resp = self.chain.complete("test")
        self.assertIsNone(resp.error)
        self.assertEqual(resp.content, "after backoff")
        # Should have slept at least once for the 429
        self.assertGreater(mock_sleep.call_count, 0)

    def test_complete_retries_500_then_succeeds(self):
        body = make_full_response("recovered")
        responses = [
            urllib_error_HTTPError(
                url="http://x", code=500, msg="Internal Server Error", hdrs={},
                fp=io.BytesIO(b'{"error":{"message":"oops"}}'),
            ),
            FakeHTTPResponse(200, body=body),
        ]
        with patch("urllib.request.urlopen", side_effect=responses), \
             patch("time.sleep") as mock_sleep:
            resp = self.chain.complete("test")
        self.assertEqual(resp.content, "recovered")
        self.assertGreater(mock_sleep.call_count, 0)

    def test_complete_exhausts_retries(self):
        err_body = json.dumps({"error": {"message": "down"}}).encode()
        http_err = urllib_error_HTTPError(
            url="http://x", code=500, msg="Internal Server Error", hdrs={}, fp=io.BytesIO(err_body),
        )
        # Bump max_retries via a fresh chain
        chain = Chain(api_key="sk-test", max_retries=2, base_delay=0.01)
        with patch("urllib.request.urlopen", side_effect=http_err), \
             patch("time.sleep"):
            resp = chain.complete("test")
        self.assertIsNotNone(resp.error)

    def test_stream_yields_tokens(self):
        stream = make_sse_response([
            ("assistant", "Hel"),
            ("assistant", "lo"),
            ("assistant", "!"),
            ("stop", None),
        ])
        fake = FakeHTTPResponse(200, body=stream.read())
        with patch("urllib.request.urlopen", return_value=fake):
            chunks = list(self.chain.stream("say hi"))
        # 3 token chunks + final
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(chunks[-1].is_complete)
        self.assertEqual(chunks[-1].content, "Hello!")

    def test_complete_with_fallback_uses_first_working(self):
        body = make_full_response("via fallback")
        # First two models fail, third succeeds
        responses = [
            urllib_error_HTTPError(url="http://x", code=500, msg="fail", hdrs={}, fp=io.BytesIO(b'{"error":{"message":"nope"}}')),
            urllib_error_HTTPError(url="http://x", code=500, msg="fail", hdrs={}, fp=io.BytesIO(b'{"error":{"message":"nope"}}')),
            FakeHTTPResponse(200, body=body),
        ]  
        chain = Chain(api_key="sk-test", max_retries=0)
        with patch("urllib.request.urlopen", side_effect=responses), \
             patch("time.sleep"):
            resp = chain.complete_with_fallback(
                primary_model="gpt-4o",
                fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"],
                prompt="hi",
            )
        self.assertIsNone(resp.error)
        self.assertEqual(resp.content, "via fallback")

    def test_function_call_via_complete(self):
        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "add",
                "arguments": json.dumps({"a": 1, "b": 2}),
            },
        }]
        body = make_full_response("", tool_calls=tool_calls)
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(200, body=body)):
            resp = self.chain.complete("call add(1, 2)", functions=[
                Function(
                    name="add",
                    description="add two numbers",
                    parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
                )
            ])
        self.assertTrue(resp.is_function_call)
        self.assertEqual(resp.function_call.name, "add")
        self.assertEqual(resp.function_call.arguments, {"a": 1, "b": 2})

    def test_cost_is_populated_in_response(self):
        body = make_full_response("ok", model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(200, body=body)):
            resp = self.chain.complete("test")
        # 1K input @ 2.50 + 500 output @ 10.00 = 0.00250 + 0.00500 = 0.00750
        self.assertAlmostEqual(resp.usage.cost_usd, 0.00750, places=6)


# =============================================================================
# Function dataclass
# =============================================================================

class TestFunctionDataclass(unittest.TestCase):

    def test_function_to_openai_tool(self):
        f = Function(
            name="get_time",
            description="Get the time",
            parameters={"type": "object", "properties": {"zone": {"type": "string"}}},
            required_params=["zone"],
        )
        tool = f.to_openai_tool()
        self.assertEqual(tool["name"], "get_time")
        self.assertEqual(tool["description"], "Get the time")
        self.assertEqual(tool["parameters"]["properties"]["zone"]["type"], "string")


# =============================================================================
# Backoff & sleep — fast tests
# =============================================================================

class TestBackoff(unittest.TestCase):

    def test_sleep_for_retry_uses_retry_after(self):
        chain = Chain(api_key="sk-test", base_delay=10.0)
        with patch("time.sleep") as mock_sleep:
            chain._sleep_for_retry(0, {"Retry-After": "3.5"}, 429)
        # Should have slept approximately 3.5 (not 10)
        args, _ = mock_sleep.call_args
        self.assertAlmostEqual(args[0], 3.5, places=1)

    def test_sleep_for_retry_uses_backoff_when_no_header(self):
        chain = Chain(api_key="sk-test", base_delay=0.1)
        with patch("time.sleep") as mock_sleep:
            chain._sleep_for_retry(2, {}, 500)
        args, _ = mock_sleep.call_args
        # Should be > base_delay and <= 60
        self.assertGreater(args[0], 0)
        self.assertLessEqual(args[0], 60.0)

    def test_backoff_increases(self):
        chain = Chain(api_key="sk-test", base_delay=0.1)
        sleeps = []
        for attempt in range(4):
            with patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
                chain._sleep_for_retry(attempt, {}, 500)
        # Each subsequent attempt should generally sleep longer (with jitter caveat)
        # Just check the max is greater than the min
        self.assertGreater(max(sleeps), min(sleeps))


# =============================================================================
# Public API smoke tests (without HTTP)
# =============================================================================

class TestPublicAPI(unittest.TestCase):

    def test_chain_requires_api_key(self):
        with self.assertRaises(ValueError):
            Chain(api_key="")

    def test_function_usage_defaults(self):
        u = TokenUsage()
        self.assertEqual(u.total_tokens, 0)
        self.assertEqual(u.cost_usd, 0.0)

    def test_response_defaults(self):
        r = LLMResponse()
        self.assertEqual(r.content, "")
        self.assertFalse(r.is_function_call)
        self.assertFalse(r.is_complete)

    def test_structured_output_smoke(self):
        # Smoke test: structured_output issues a chat completion, extracts JSON,
        # and validates against a schema. We patch the network layer to return
        # a known JSON payload.
        body = make_full_response('{"title": "Inception", "rating": 5, "tags": ["sci-fi"]}')
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "rating": {"type": "integer", "minimum": 0, "maximum": 5},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "rating"],
        }
        # Build a custom chain to avoid polluting env
        chain = Chain(api_key="sk-test", max_retries=0)
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(200, body=body)):
            data, err = structured_output("rate Inception", schema, chain=chain)
        self.assertIsNotNone(data)
        self.assertIsNone(err)
        self.assertEqual(data["title"], "Inception")
        self.assertEqual(data["rating"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
