# tiny-chain

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()

> Streaming LLM processor with retries, fallbacks, function calling, and JSON extraction. Zero dependencies.

A single-file Python module that does the core LLM processing patterns developers actually need — without the framework.

---

## Features

- 🚀 **Streaming responses token-by-token** — see output as it's generated
- 🔁 **Retry with exponential backoff** — automatic recovery from transient errors
- 🥇 **Model fallbacks** — primary → secondary → tertiary chain on failure
- 🛠️ **Function calling with JSON schema** — tools/agents without a framework
- 🔍 **JSON extraction from responses** — markdown, partial, or messy text
- ✅ **Structured output validation** — extract & correct JSON against a schema
- 🔄 **Sync + async support** — works in either world
- 📊 **Token usage tracking** — every response reports usage
- 💰 **Cost estimation** — built-in price tables for popular models

## Why?

Existing tools (LangChain, LlamaIndex) are heavy — hundreds of dependencies, sprawling abstractions, and version churn. `tiny-chain` is a single file that handles the LLM processing patterns developers actually need:

| Pattern | LangChain | LlamaIndex | tiny-chain |
|---|---|---|---|
| HTTP request to OpenAI | 200 LOC | n/a | **30 LOC** |
| Retry w/ exponential backoff | ✅ | ✅ | **✅** |
| Streaming | ✅ | ✅ | **✅** |
| Function calling | ✅ | ❌ | **✅** |
| JSON extraction | partial | partial | **✅** (markdown, partial, schema) |
| Model fallbacks | ✅ | ❌ | **✅** |
| Cost tracking | plugin | ❌ | **built-in** |

**One file. ~600 LOC. Zero dependencies. Stdlib only.**

## Installation

```bash
pip install tiny-chain
```

Or just copy `tiny_chain.py` — there's only one.

## Quick start

### Streaming completion

```python
from tiny_chain import Chain

chain = Chain(api_key="sk-...", model="gpt-4o-mini")

for chunk in chain.stream("Write a haiku about Python"):
    print(chunk.content, end="", flush=True)
# Silent modules import
# Functions compose, classes
# Code becomes poetry
```

### Non-streaming with retries

```python
from tiny_chain import Chain

chain = Chain(
    api_key="sk-...",
    model="gpt-4o",
    max_retries=5,        # retry transient errors
    base_delay=1.0,       # exponential backoff
)

response = chain.complete("What is the capital of France?")
print(response.content)             # "Paris"
print(response.usage.total_tokens)  # 12
print(f"Cost: ${response.usage.cost_usd:.6f}")
```

### Model fallbacks

```python
chain = Chain(api_key="sk-...", model="gpt-4o")

# If gpt-4o fails, try gpt-4o-mini, then gpt-3.5-turbo
response = chain.complete_with_fallback(
    primary_model="gpt-4o",
    fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"],
    prompt="Explain quantum entanglement",
)
```

### Function calling

```python
from tiny_chain import Chain, Function

def get_weather(location: str, unit: str = "celsius") -> str:
    """Return the weather for a given location."""
    return f"It's 22°{unit[0].upper()} and sunny in {location}."

weather_func = Function(
    name="get_weather",
    description="Get the current weather in a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
    },
    required_params=["location"],
)

chain = Chain(api_key="sk-...")

response = chain.complete(
    "What's the weather in Tokyo?",
    functions=[weather_func],
)

if response.is_function_call:
    result = chain.call_function(response, {"get_weather": get_weather})
    print(result)  # "It's 22°C and sunny in Tokyo."
```

### JSON extraction

```python
from tiny_chain import json_extractor, structured_output

# Pull JSON out of messy text
text = """
Sure! Here's the data:
```json
{"name": "Ada", "born": 1815, "alive": false}
```
Hope that helps!
"""

data, err = json_extractor.extract_json(text)
print(data)  # {"name": "Ada", "born": 1815, "alive": False}
```

### Structured output with schema validation

```python
from tiny_chain import structured_output

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rating": {"type": "integer", "minimum": 0, "maximum": 5},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "rating"],
}

data, err = structured_output(
    "Review the movie Inception. Return JSON with title, rating 0-5, and tags.",
    schema,
)

if data:
    print(f"{data['title']}: {data['rating']}⭐ — {data['tags']}")
```

## API reference

### `Chain`

The main class. Constructor:

```python
Chain(
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 60,
    max_retries: int = 3,
    base_delay: float = 1.0,
)
```

Key methods:

| Method | Description |
|---|---|
| `complete(prompt, system=None, functions=None, temperature=0.7, max_tokens=2048)` | Non-streaming completion. Returns `LLMResponse`. |
| `stream(prompt, system=None, functions=None, temperature=0.7, max_tokens=2048)` | Streaming generator yielding `LLMResponse` (accumulating content). |
| `call_function(response, function_map)` | Execute a function call from the response. Returns the call's return value. |
| `complete_with_fallback(primary_model, fallback_models, prompt, system, functions)` | Try models in order until one succeeds. |

### `TokenUsage`

```python
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
```

### `LLMResponse`

```python
@dataclass
class LLMResponse:
    content: str                         # accumulated text
    function_call: Optional[FunctionCall]
    usage: TokenUsage
    model: str
    raw: str                             # raw accumulated stream or full body
    is_complete: bool
    is_function_call: bool
    extracted_json: Optional[dict]
```

### `Function`

```python
@dataclass
class Function:
    name: str
    description: str
    parameters: dict      # JSON schema dict
    required_params: list
```

### `json_extractor`

| Function | Description |
|---|---|
| `extract_json(text)` | Find the first JSON object/array in text. Handles ```json``` fences, partial JSON. Returns `(dict, error)`. |
| `extract_schema(text, schema)` | Like `extract_json`, but validates against a JSON schema (stdlib `jsonschema`-free implementation). |
| `structured_output(prompt, schema)` | One-shot: send `prompt`, extract & validate JSON against `schema`. |

## Supported models (cost table)

```python
{
    "gpt-4o":            (2.50, 10.00),  # /1M input, output USD
    "gpt-4o-mini":       (0.15, 0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "gpt-3.5-turbo":     (0.50, 1.50),
    "default":           (1.00, 2.00),
}
```

Costs update with the field — override `_cost_per_1k_tokens` to add your own model.

## Part of the `tiny-*` ecosystem

[![tiny-* ecosystem](https://img.shields.io/badge/tiny--*%20ecosystem-powered-blueviolet.svg)]()

Small, focused, single-file Python tools you can read in one sitting:

- **`tiny-chain`** — LLM processing (you are here)
- More `tiny-*` packages coming

Each one: stdlib-only, MIT, ~one file, focused.

## License

MIT © 2026 hussain-alsaibai
