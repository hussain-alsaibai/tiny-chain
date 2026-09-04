# tiny-chain

> **Zero-dependency structured chain-of-thought for AI agents. MIT.**

`tiny-chain` provides a lightweight way to record, structure, and export the "reasoning steps" of an AI agent. It transforms messy internal monologues into clean, machine-readable JSON traces with timing and metadata.

## Why?

Most agents print thoughts to the console or hide them in logs. `tiny-chain` makes reasoning a first-class data structure that can be:
1. Debugged in real-time.
2. Evaluated for quality.
3. Shown to users as "working..." progress.
4. Saved for long-term memory or fine-tuning.

## Install

Single file. No dependencies.

```bash
pip install tiny-chain
```

## Quickstart

```python
from tiny_chain import TinyChain

chain = TinyChain("market-analysis")

@chain.step("Fetch Data", "I need to get the latest BTC price from the API.")
def get_price():
    return 65000

@chain.step("Analyze Trend", "Price is above the 200-day EMA, suggesting a bullish trend.")
def analyze(price):
    return "bullish"

price = get_price()
trend = analyze(price)

print(chain.to_json())
```

## Features

- **Context-aware:** Wrap any function in a step.
- **Timed:** Automatic duration tracking for each reasoning hop.
- **Exportable:** Full JSON serialization for telemetry or UI.
- **Zero-dep:** Standard library only.

## License

MIT
