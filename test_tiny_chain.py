import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from tiny_chain import TinyChain

def test_step_decorator():
    chain = TinyChain("test")
    @chain.step("math", "adding numbers")
    def add(a, b):
        return a + b
    assert add(2, 3) == 5
    assert len(chain.steps) == 1
    step = chain.steps[0]
    assert step.name == "math"
    assert step.thought == "adding numbers"
    assert step.output == 5

def test_add_step_manual():
    chain = TinyChain("manual")
    chain.add_step("observe", "reading input", {"x": 1})
    assert len(chain.steps) == 1
    assert chain.steps[0].output == {"x": 1}

def test_export_json():
    chain = TinyChain("export")
    chain.add_step("a", "first", 1)
    chain.add_step("b", "second", 2)
    data = json.loads(chain.to_json())
    assert data["name"] == "export"
    assert len(data["steps"]) == 2
    assert data["steps"][0]["output"] == 1

def test_metadata():
    chain = TinyChain("meta")
    chain.add_step("s", "t", metadata={"source": "test"})
    assert chain.steps[0].metadata == {"source": "test"}

if __name__ == "__main__":
    test_step_decorator()
    test_add_step_manual()
    test_export_json()
    test_metadata()
    print("tiny-chain: all tests passed")
