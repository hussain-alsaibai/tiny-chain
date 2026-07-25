"""tiny-chain: Streaming LLM processor with retries, fallbacks, function calling, and JSON extraction.

Zero dependencies — stdlib only.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union


# =============================================================================
# Cost table (USD per 1M tokens: input, output)
# =============================================================================

_cost_per_1k_tokens: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-05-13": (5.00, 15.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic (via OpenAI-compatible API)
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Mistral / Mixtral (commonly routed)
    "mistral-large": (2.00, 6.00),
    "mixtral-8x7b": (0.27, 0.27),
    # Fallback
    "default": (1.00, 2.00),
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class FunctionCall:
    name: str
    arguments: Dict[str, Any]
    parsed: bool = True


@dataclass
class LLMResponse:
    content: str = ""
    function_call: Optional[FunctionCall] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw: str = ""
    is_complete: bool = False
    is_function_call: bool = False
    extracted_json: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Function:
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Render as an OpenAI `tools[0].function` dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# =============================================================================
# Exception types
# =============================================================================

class ChainError(Exception):
    """Base exception raised by Chain. f"""


class AuthError(ChainError):
    """401: invalid API key or missing permissions."""


class BadRequestError(ChainError):
    """400: malformed request."""


class RateLimitError(ChainError):
    """429: rate limit hit; honoring Retry-After is mandatory upstream."""


class ServerError(ChainError):
    """5xx: server error, retry recommended."""


# =============================================================================
# Chain — primary class
# =============================================================================

class Chain:
    """A single LLM client with streaming, retries, fallbacks, function calling, and JSON extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.base_delay = float(base_delay)

        # Public knobs
        self.fallback_models: List[str] = []

    # ------------------------------------------------------------------ HTTP

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Function]],
        stream: bool,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if functions:
            payload["tools"] = [{"type": "function", "function": f.to_openai_tool()} for f in functions]
        return payload

    def _do_request(self, payload: Dict[str, Any], stream: bool) -> Tuple[int, Dict[str, str], Any]:
        """Make a single HTTP request. Returns (status_code, headers, body).

        For streaming, `body` is the raw response object (must be read iteratively).
        For non-streaming, `body` is the decoded JSON dict (or empty dict on error).
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "User-Agent": "tiny-chain/0.1.0",
            },
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            status = resp.getcode()
            headers = dict(resp.headers.items())
            if stream:
                return status, headers, resp
            body_text = resp.read().decode("utf-8", errors="replace")
            try:
                return status, headers, json.loads(body_text)
            except json.JSONDecodeError:
                return status, headers, {"_raw": body_text}
        except urllib.error.HTTPError as e:
            status = e.code
            headers = dict(e.headers.items()) if e.headers else {}
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
            except Exception:
                err_body = ""
            try:
                err_json = json.loads(err_body) if err_body else {}
            except json.JSONDecodeError:
                err_json = {"_raw": err_body}
            err_json["_status"] = status
            err_json["_headers"] = headers
            return status, headers, err_json
        except urllib.error.URLError as e:
            # Treat network errors as 503 for retry purposes
            return 503, {}, {"_raw": str(e), "error": {"message": str(e)}}

    def _classify_error(self, status: int, body: Any) -> Exception:
        msg: str = ""
        if isinstance(body, dict):
            err = body.get("error") or {}
            if isinstance(err, dict):
                msg = err.get("message") or body.get("_raw") or json.dumps(body)[:200]
            else:
                msg = str(err)
            if not msg:
                raw = body.get("_raw")
                if isinstance(raw, str):
                    msg = raw
                else:
                    msg = json.dumps(body)[:200]
        else:
            msg = str(body)
        if status == 400:
            return BadRequestError(msg)
        if status == 401:
            return AuthError(msg)
        if status == 403:
            return AuthError(msg)
        if status == 429:
            return RateLimitError(msg)
        if 500 <= status < 600:
            return ServerError(msg)
        if status == 503:
            return ServerError(msg or "network error")
        return ChainError(f"HTTP {status}: {msg}")

    # --------------------------------------------------------------- Stream

    def _stream_events(self, stream_response: Any) -> Iterator[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
        """Parse SSE lines from a streaming HTTP response.

        Yields (delta_content, final_chunk_dict). `delta_content` is None for non-content events.
        The final chunk has `final_chunk_dict` populated with the full assistant message metadata
        (finish_reason, usage, model) when the stream signals completion.
        """
        try:
            for raw_line in stream_response:
                try:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                except AttributeError:
                    line = str(raw_line).rstrip("\n").rstrip("\r")
                if not line:
                    continue
                if line.startswith(":"):
                    # SSE comment / keep-alive
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    # Treat as terminator
                    yield None, None
                    return
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = parsed.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    content_piece = delta.get("content")
                    if content_piece:
                        yield content_piece, None
                    finish = choice.get("finish_reason")
                    if finish:
                        yield None, {
                            "finish_reason": finish,
                            "usage": parsed.get("usage"),
                            "model": parsed.get("model"),
                        }
                else:
                    usage = parsed.get("usage")
                    if usage:
                        yield None, {"usage": usage, "model": parsed.get("model")}
        finally:
            try:
                stream_response.close()
            except Exception:
                pass

    # --------------------------------------------------------- Cost / Parse

    def _estimate_cost(self, model: str, usage: TokenUsage) -> float:
        input_cost_per_1m, output_cost_per_1m = _cost_per_1k_tokens.get(
            model, _cost_per_1k_tokens["default"]
        )
        if model not in _cost_per_1k_tokens:
            input_cost_per_1m, output_cost_per_1m = _cost_per_1k_tokens["default"]
        input_cost = (usage.prompt_tokens / 1_000_000.0) * input_cost_per_1m
        output_cost = (usage.completion_tokens / 1_000_000.0) * output_cost_per_1m
        return round(input_cost + output_cost, 10)

    def _parse_function_call(self, content: str) -> Optional[FunctionCall]:
        """Try to extract a function call from the response content.

        Looks for either:
        - A JSON top-level object with a "name" and "arguments"/"parameters" field
        - A ``function_call`` block embedded in the text
        """
        if not content:
            return None
        # Try ```json fences first
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        if fenced:
            candidate = fenced.group(1)
        else:
            candidate = content
        data, _ = json_extractor.extract_json(candidate)
        if not isinstance(data, dict):
            return None
        name = data.get("name") or data.get("function") or data.get("function_name")
        if not name or not isinstance(name, str):
            return None
        args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        return FunctionCall(name=name, arguments=args, parsed=True)

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        data, _ = json_extractor.extract_json(content)
        return data if isinstance(data, dict) else None

    # ----------------------------------------------------------- Public API

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        functions: Optional[List[Function]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a non-streaming chat completion request and return a populated LLMResponse."""
        messages = self._build_messages(prompt, system)
        return self._run_with_retries(
            messages=messages,
            functions=functions,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        functions: Optional[List[Function]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[LLMResponse, None, None]:
        """Stream the chat completion. Yields partial LLMResponse objects as tokens arrive.

        The final yielded response has ``is_complete=True``.
        """
        messages = self._build_messages(prompt, system)
        payload = self._build_payload(messages, functions, stream=True,
                                      temperature=temperature, max_tokens=max_tokens)
        accumulated = ""
        usage = TokenUsage()
        finish_reason: Optional[str] = None
        model_name = payload["model"]
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                status, headers, body = self._do_request(payload, stream=True)
                if status != 200:
                    err = self._classify_error(status, body)
                    if isinstance(err, AuthError) or isinstance(err, BadRequestError):
                        raise err
                    last_error = err
                    self._sleep_for_retry(attempt, headers, status)
                    continue
                # 200 path: stream and accumulate
                for delta, terminal in self._stream_events(body):
                    if delta:
                        accumulated += delta
                        yield LLMResponse(
                            content=accumulated,
                            function_call=None,
                            usage=TokenUsage(),
                            model=model_name,
                            raw=accumulated,
                            is_complete=False,
                            is_function_call=False,
                            extracted_json=None,
                            finish_reason=None,
                            error=None,
                        )
                    if terminal:
                        finish_reason = terminal.get("finish_reason") or finish_reason
                        if terminal.get("model"):
                            model_name = terminal["model"]
                        u = terminal.get("usage")
                        if u:
                            usage.prompt_tokens = u.get("prompt_tokens", usage.prompt_tokens) or 0
                            usage.completion_tokens = u.get("completion_tokens", usage.completion_tokens) or 0
                            usage.total_tokens = u.get("total_tokens", usage.total_tokens) or 0
                # Final accumulated
                usage.cost_usd = self._estimate_cost(model_name, usage)
                function_call = self._parse_function_call(accumulated)
                response = LLMResponse(
                    content=accumulated,
                    function_call=function_call,
                    usage=usage,
                    model=model_name,
                    raw=accumulated,
                    is_complete=True,
                    is_function_call=function_call is not None,
                    extracted_json=self._extract_json(accumulated),
                    finish_reason=finish_reason,
                    error=None,
                )
                yield response
                return
            except (AuthError, BadRequestError):
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    self._sleep_for_retry(attempt, {}, 500)
                    continue
        # exhausted retries
        yield LLMResponse(
            content="",
            usage=TokenUsage(),
            model=payload["model"],
            raw=str(last_error) if last_error else "",
            is_complete=True,
            is_function_call=False,
            error=str(last_error) if last_error else "unknown error",
        )
        return

    def call_function(
        self,
        response: LLMResponse,
        function_map: Dict[str, Callable[..., Any]],
    ) -> Any:
        """Execute a function call from a response. Returns the function's return value."""
        if not response.function_call:
            raise ValueError("response has no function_call to execute")
        name = response.function_call.name
        if name not in function_map:
            raise KeyError(f"function '{name}' not found in provided function_map")
        fn = function_map[name]
        try:
            return fn(**response.function_call.arguments)
        except TypeError:
            # allow single positional arg
            if len(response.function_call.arguments) == 1:
                only_value = next(iter(response.function_call.arguments.values()))
                return fn(only_value)
            raise

    def complete_with_fallback(
        self,
        primary_model: str,
        fallback_models: List[str],
        prompt: str,
        system: Optional[str] = None,
        functions: Optional[List[Function]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Try `primary_model`, then `fallback_models` in order."""
        models: List[str] = [primary_model] + list(fallback_models)
        last_response: Optional[LLMResponse] = None
        for model_name in models:
            messages = self._build_messages(prompt, system)
            payload = self._build_payload(
                messages, functions, stream=False,
                temperature=temperature, max_tokens=max_tokens, model=model_name,
            )
            attempt_response = self._do_with_retries_payload(payload, model_name=model_name)
            last_response = attempt_response
            # Treat "error" content as failure if no content produced
            content = (attempt_response.content or "").strip()
            if content and not attempt_response.error:
                return attempt_response
        # All failed; return the last response (may have error populated)
        return last_response or LLMResponse(error="no model in fallback chain succeeded")

    # --------------------------------------------------------- Internals

    def _build_messages(self, prompt: str, system: Optional[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _sleep_for_retry(self, attempt: int, headers: Dict[str, str], status: int) -> None:
        """Sleep with exponential backoff, honoring Retry-After for 429."""
        retry_after: Optional[float] = None
        if status == 429 and isinstance(headers, dict):
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra is not None:
                try:
                    retry_after = float(ra)
                except ValueError:
                    # Could be HTTP-date
                    try:
                        from email.utils import parsedate_to_datetime
                        target = parsedate_to_datetime(ra)
                        now = time.time()
                        retry_after = max(0.0, target.timestamp() - now)
                    except Exception:
                        retry_after = None
        backoff = self.base_delay * (2 ** attempt) + random.uniform(0, self.base_delay * 0.25)
        delay = retry_after if retry_after is not None else backoff
        time.sleep(min(delay, 60.0))

    def _run_with_retries(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Function]],
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        payload = self._build_payload(messages, functions, stream=stream,
                                      temperature=temperature, max_tokens=max_tokens)
        return self._do_with_retries_payload(payload)

    def _do_with_retries_payload(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
    ) -> LLMResponse:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            stream = bool(payload.get("stream"))
            try:
                status, headers, body = self._do_request(payload, stream=stream)
                if status != 200:
                    err = self._classify_error(status, body)
                    if isinstance(err, (AuthError, BadRequestError)):
                        raise err
                    last_error = err
                    self._sleep_for_retry(attempt, headers, status)
                    continue
                # OK
                if stream:
                    # Shouldn't hit this branch from complete(), but handle anyway
                    accumulated = ""
                    for delta, terminal in self._stream_events(body):
                        if delta:
                            accumulated += delta
                        if terminal:
                            break
                    return LLMResponse(content=accumulated, model=payload["model"], is_complete=True)
                # Non-stream JSON
                choices = body.get("choices") or []
                if not choices:
                    return LLMResponse(
                        model=payload.get("model", model_name or self.model),
                        raw=json.dumps(body)[:500],
                        is_complete=True,
                        error="no_choices",
                    )
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                function_call = None
                if tool_calls:
                    tc = tool_calls[0].get("function") or {}
                    raw_args = tc.get("arguments") or "{}"
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": raw_args}
                    function_call = FunctionCall(name=tc.get("name", ""), arguments=parsed_args, parsed=True)
                usage_data = body.get("usage") or {}
                usage = TokenUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0) or 0,
                    completion_tokens=usage_data.get("completion_tokens", 0) or 0,
                    total_tokens=usage_data.get("total_tokens", 0) or 0,
                )
                usage.cost_usd = self._estimate_cost(body.get("model") or payload["model"], usage)
                resp_model = body.get("model") or payload["model"]
                return LLMResponse(
                    content=content,
                    function_call=function_call,
                    usage=usage,
                    model=resp_model,
                    raw=json.dumps(body)[:500],
                    is_complete=True,
                    is_function_call=function_call is not None,
                    extracted_json=self._extract_json(content) if not function_call else None,
                    finish_reason=choices[0].get("finish_reason"),
                    error=None,
                )
            except (AuthError, BadRequestError):
                raise
            except RateLimitError as e:
                last_error = e
                self._sleep_for_retry(attempt, {}, 429)
            except ServerError as e:
                last_error = e
                self._sleep_for_retry(attempt, {}, 500)
            except urllib.error.URLError as e:
                last_error = e
                self._sleep_for_retry(attempt, {}, 503)
            except Exception as e:
                last_error = e
                self._sleep_for_retry(attempt, {}, 500)
        # Out of retries
        return LLMResponse(
            model=payload.get("model", model_name or self.model),
            raw=str(last_error) if last_error else "",
            is_complete=True,
            error=str(last_error) if last_error else "retries exhausted",
        )


# =============================================================================
# Orchestration: Pipeline, Parallel, If, retry
# =============================================================================

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """Simple retry decorator for steps. Usage: @retry() or step = retry()(step)"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_err = None
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    if i < max_attempts - 1:
                        time.sleep(delay * (2**i))
            raise last_err
        return wrapper
    return decorator


class Pipeline:
    """Sequential step execution with hooks and context management."""
    def __init__(self, steps: List[Callable] = None):
        self.steps = steps or []
        self.hooks: Dict[str, List[Callable]] = {"before": [], "after": [], "error": []}

    def add(self, step: Callable) -> Pipeline:
        """Add a step to the pipeline. Steps can be callables or objects with a run() method."""
        self.steps.append(step)
        return self

    def on(self, event: str, hook: Callable) -> Pipeline:
        """Register a hook: 'before'(ctx), 'after'(ctx), or 'error'(err, ctx)."""
        if event in self.hooks:
            self.hooks[event].append(hook)
        return self

    def run(self, ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run all steps sequentially. Results that are dicts are merged into the context."""
        ctx = ctx if ctx is not None else {}
        for hook in self.hooks["before"]:
            hook(ctx)
        try:
            for step in self.steps:
                res = step.run(ctx) if hasattr(step, "run") else step(ctx)
                if isinstance(res, dict):
                    ctx.update(res)
            for hook in self.hooks["after"]:
                hook(ctx)
        except Exception as e:
            for hook in self.hooks["error"]:
                hook(e, ctx)
            raise
        return ctx


class Parallel:
    """Parallel execution of multiple steps. Merges results into context."""
    def __init__(self, steps: List[Callable]):
        self.steps = steps

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        threads = []
        # Use a lock if steps might modify the same keys, but we assume
        # functional steps or independent keys for "tiny" implementation.
        lock = threading.Lock()
        def _task(s, c):
            res = s.run(c) if hasattr(s, "run") else s(c)
            if isinstance(res, dict):
                with lock:
                    ctx.update(res)

        for step in self.steps:
            t = threading.Thread(target=_task, args=(step, ctx))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return ctx


class If:
    """Conditional branching: run then_step if cond(ctx) is true, else run else_step."""
    def __init__(self, cond: Callable[[Dict[str, Any]], bool], then_step: Callable, else_step: Callable = None):
        self.cond = cond
        self.then_step = then_step
        self.else_step = else_step

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        step = self.then_step if self.cond(ctx) else self.else_step
        if step:
            return step.run(ctx) if hasattr(step, "run") else step(ctx)
        return ctx


# =============================================================================
# json_extractor — JSON utilities
# =============================================================================

class _JsonExtractor:
    """Pull JSON out of messy text. Returns (data, error)."""

    _FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")
    _OBJECT_RE = re.compile(r"\{[\s\S]*\}")
    _ARRAY_RE = re.compile(r"\[[\s\S]*\]")
    _PARTIAL_OPEN = re.compile(r"\{")
    _PARTIAL_CLOSE = re.compile(r"\}")
    _PARTIAL_OPEN_SQ = re.compile(r"\[")
    _PARTIAL_CLOSE_SQ = re.compile(r"\]")

    def extract_json(self, text: str) -> Tuple[Optional[Any], Optional[str]]:
        if text is None:
            return None, "no input text"
        if not isinstance(text, str):
            return None, f"input must be str, got {type(text).__name__}"
        raw = text.strip()
        if not raw:
            return None, "empty input"

        # 1. Try the full string as JSON
        parsed, err = self._loads_safe(raw)
        if err is None:
            return parsed, None

        # 2. Try fenced ``` blocks
        for fence in self._FENCE_RE.findall(raw):
            parsed, err = self._loads_safe(fence)
            if err is None:
                return parsed, None

        # 3. Try first balanced object
        obj = self._first_balanced(raw, "{", "}")
        if obj is not None:
            parsed, err = self._loads_safe(obj)
            if err is None:
                return parsed, None

        # 4. Try first balanced array
        arr = self._first_balanced(raw, "[", "]")
        if arr is not None:
            parsed, err = self._loads_safe(arr)
            if err is None:
                return parsed, None

        # 5. Try partial / repaired JSON
        repaired = self._repair_partial(raw)
        if repaired:
            parsed, err = self._loads_safe(repaired)
            if err is None:
                return parsed, None

        return None, "no JSON object/array found"

    def extract_schema(
        self,
        text: str,
        schema: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Extract JSON and validate against `schema` (subset of JSON Schema)."""
        data, err = self.extract_json(text)
        if err:
            return None, err
        if not isinstance(data, dict) and not isinstance(data, list):
            return None, "extracted value is not a JSON object or array"
        valid, validation_error = _validate_schema(data, schema)
        if not valid:
            return None, validation_error
        return data, None

    # ------------------------------------------------------- internal

    def _loads_safe(self, text: str) -> Tuple[Optional[Any], Optional[str]]:
        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            return None, f"json decode error at pos {e.pos}: {e.msg}"

    def _first_balanced(self, text: str, open_ch: str, close_ch: str) -> Optional[str]:
        """Return the substring of the first balanced pair of `open_ch`/`close_ch`, or None."""
        depth = 0
        in_str = False
        escape = False
        start: Optional[int] = None
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if in_str:
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == close_ch:
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return text[start:i + 1]
        return None

    def _repair_partial(self, text: str) -> Optional[str]:
        """Try to repair common partial-JSON cases (truncated, missing closing braces)."""
        # Find first opening object/array while respecting strings
        first_obj, first_arr = self._scan_first_opener(text)
        if first_obj is None and first_arr is None:
            return None
        # Choose whichever comes first; if tied, prefer object
        if first_obj is not None and (first_arr is None or first_obj <= first_arr):
            start = first_obj
            kind = "{"
            close = "}"
            other_open, other_close = "[", "]"
        else:
            start = first_arr  # type: ignore[assignment]
            kind = "["
            close = "]"
            other_open, other_close = "{", "}"
        candidate = text[start:]
        # Strip any trailing closing braces before the first complete close
        # (we don't have a complete close, by definition of partial)
        # Work only on candidate from `start`
        # Strip trailing whitespace/commas
        candidate = candidate.rstrip()
        candidate = re.sub(r",\s*$", "", candidate)
        # Count balances (in the substring from start)
        # Walk string-aware to do a proper balance
        ob = candidate.count("{") - candidate.count("}")
        osq = candidate.count("[") - candidate.count("]")
        # Auto-complete keys whose value is unterminated: replace last colon with ''
        # e.g. {"a":  -> {"a": null
        # We detect: ends with ':' or ': ' at the end
        if candidate.endswith(":"):
            candidate += " null"
        # Detect trailing partial value (number/true/false/etc) — skip repair
        # Count what we need to close
        # We need to close in reverse order of opening
        # Build reverse list by scanning
        stack: List[str] = []
        in_str = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if in_str:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{" or ch == "[":
                stack.append(ch)
            elif ch == "}" or ch == "]":
                if stack:
                    stack.pop()
        # If still in_string at the end, close it
        if in_str:
            candidate += '"'
            in_str = False
        # If we ended mid-value (no closing), append null
        # Detect this by trailing tokens like true/false/none/numerals
        tail = candidate.rstrip()
        # Heuristic: if last non-space char is a letter (id) or number, append null
        if tail and tail[-1].isalpha() and not tail.endswith("null"):
            # could be "true", "false", "none" but they're terminal already
            pass
        # Append closing tokens in reverse order
        closing = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
        candidate += closing
        # Validate parse
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            # If still invalid, try replacing any trailing colon/incomplete value with null
            # Try a less precise completion: append placeholder for any open string/int
            return None

    def _scan_first_opener(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Find the first '{' and '[' positions while respecting strings."""
        first_obj: Optional[int] = None
        first_arr: Optional[int] = None
        in_str = False
        escape = False
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if in_str:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{" and first_obj is None:
                first_obj = i
            elif ch == "[" and first_arr is None:
                first_arr = i
            if first_obj is not None and first_arr is not None:
                break
        return first_obj, first_arr


# Module-level instance for the API
json_extractor = _JsonExtractor()


def structured_output(
    prompt: str,
    schema: Dict[str, Any],
    chain: Optional[Chain] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Convenience: use a Chain (or the supplied one) to produce JSON validated against `schema`.

    Returns (data, error). On validation failure, returns the extracted data anyway,
    but with an error string for diagnostics.
    """
    if chain is None:
        chain = Chain.__new__(Chain)
        # best-effort: chain requires api_key; only construct if env var is set
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "sk-fake-for-structured-output")
        chain.api_key = api_key
        chain.model = "gpt-4o-mini"
        chain.base_url = "https://api.openai.com/v1"
        chain.timeout = 60
        chain.max_retries = 1
        chain.base_delay = 1.0
        chain.fallback_models = []
    response = chain.complete(prompt, temperature=0.0, max_tokens=2048)
    if response.error:
        return None, response.error
    return json_extractor.extract_schema(response.content or "", schema)


# =============================================================================
# Minimal JSON Schema validator (subset, stdlib only)
# =============================================================================

def _validate_schema(value: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate `value` against a JSON schema (subset). Returns (ok, error_or_none)."""
    if not isinstance(schema, dict):
        return True, None  # No-op for non-dict schemas
    # type
    expected_type = schema.get("type")
    if expected_type:
        if not _check_type(value, expected_type):
            return False, f"expected type {expected_type!r}, got {type(value).__name__}"
    # enum
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return False, f"value {value!r} not in enum {enum!r}"
    # const
    const = schema.get("const")
    if const is not None and value != const:
        return False, f"value {value!r} != const {const!r}"
    # anyOf / oneOf / allOf — minimal support
    if "anyOf" in schema:
        if not any(_validate_schema(value, s)[0] for s in schema["anyOf"] if isinstance(s, dict)):
            return False, "value does not match anyOf"
    if "allOf" in schema:
        for s in schema["allOf"]:
            ok, err = _validate_schema(value, s)
            if not ok:
                return False, f"allOf failed: {err}"
    # object
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        additional = schema.get("additionalProperties", True)
        for key in required:
            if key not in value:
                return False, f"missing required property {key!r}"
        for k, v in value.items():
            if k in properties:
                ok, err = _validate_schema(v, properties[k])
                if not ok:
                    return False, f"property {k!r}: {err}"
            else:
                if additional is False:
                    return False, f"unexpected property {k!r}"
                elif isinstance(additional, dict):
                    ok, err = _validate_schema(v, additional)
                    if not ok:
                        return False, f"property {k!r}: {err}"
        # patternProperties: very basic support
        for pattern, subschema in (schema.get("patternProperties") or {}).items():
            try:
                rgx = re.compile(pattern)
            except re.error:
                continue
            for k, v in value.items():
                if rgx.match(k):
                    ok, err = _validate_schema(v, subschema)
                    if not ok:
                        return False, f"patternProperty {pattern!r} on {k!r}: {err}"
    # array
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(value):
                ok, err = _validate_schema(v, items)
                if not ok:
                    return False, f"item {i}: {err}"
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return False, f"array length {len(value)} < minItems {min_items}"
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            return False, f"array length {len(value)} > maxItems {max_items}"
        unique = schema.get("uniqueItems")
        if unique and len(set(_hashable(v) for v in value)) != len(value):
            return False, "array items not unique"
    # string
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False, f"string length < minLength {schema['minLength']}"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False, f"string length > maxLength {schema['maxLength']}"
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    return False, f"string does not match pattern {schema['pattern']!r}"
            except re.error:
                pass
        if "format" in schema:
            # very light format support
            fmt = schema["format"]
            if fmt == "email" and "@" not in value:
                return False, "string is not a valid email"
            if fmt == "uri" and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
                return False, "string is not a valid URI"
    # number/integer
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False, f"value < minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return False, f"value > maximum {schema['maximum']}"
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False, f"value <= exclusiveMinimum {schema['exclusiveMinimum']}"
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return False, f"value >= exclusiveMaximum {schema['exclusiveMaximum']}"
        if schema.get("type") == "integer" and not isinstance(value, int):
            return False, "value is not an integer"
        if "multipleOf" in schema:
            try:
                if value % schema["multipleOf"] != 0:
                    return False, f"value not a multiple of {schema['multipleOf']}"
            except Exception:
                pass
    return True, None


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _check_type(value: Any, expected: str) -> bool:
    py_type = _TYPE_MAP.get(expected)
    if py_type is None:
        return True
    # JSON boolean edge-case: in JSON, bools are not ints
    if expected == "integer" and isinstance(value, bool):
        return False
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, py_type)


def _hashable(v: Any):
    try:
        hash(v)
        return v
    except TypeError:
        return json.dumps(v, sort_keys=True, default=str)


# =============================================================================
# main / CLI entry
# =============================================================================

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="tiny-chain", description=__doc__)
    parser.add_argument("prompt", nargs="*", help="Prompt(s). Joined with spaces.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    if not args.prompt:
        parser.print_help()
        return 1
    chain = Chain(api_key=_env_api_key(), model=args.model)
    for chunk in chain.stream(" ".join(args.prompt), temperature=args.temperature,
                              max_tokens=args.max_tokens):
        if chunk.is_complete:
            print(f"\n[tiny-chain: usage={chunk.usage.total_tokens} cost=${chunk.usage.cost_usd:.6f}]")
        else:
            print(chunk.content, end="", flush=True)
    return 0


def _env_api_key() -> str:
    import os
    for var in ("OPENAI_API_KEY", "TINY_CHAIN_API_KEY", "ANTHROPIC_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    raise SystemExit("no API key found in OPENAI_API_KEY / TINY_CHAIN_API_KEY / ANTHROPIC_API_KEY")


if __name__ == "__main__":
    raise SystemExit(main())
