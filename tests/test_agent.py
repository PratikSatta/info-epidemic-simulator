import pytest
from src.agent import Agent

def test_agent_created_correctly():
    a = Agent(node_id=0, trust=0.7, bias=0.2, influence=0.5)
    assert a.state == "S"
    assert a.trust == 0.7

def test_trust_out_of_range_raises_valueerror():
    with pytest.raises(ValueError):
        Agent(node_id=0, trust=1.5, bias=0.0, influence=0.5)

def test_trust_wrong_type_raises_typeerror():
    with pytest.raises(TypeError):
        Agent(node_id=0, trust="high", bias=0.0, influence=0.5)

def test_invalid_state_raises_valueerror():
    a = Agent(node_id=0, trust=0.5, bias=0.0, influence=0.5)
    with pytest.raises(ValueError):
        a.state = "X"

def test_bias_affinity_identical():
    a = Agent(node_id=0, trust=0.5, bias=0.5, influence=0.5)
    assert a.bias_affinity(0.5) == 1.0

def test_bias_affinity_opposite():
    a = Agent(node_id=0, trust=0.5, bias=1.0, influence=0.5)
    assert a.bias_affinity(-1.0) == 0.5

def test_node_id_is_readonly():
    a = Agent(node_id=5, trust=0.5, bias=0.0, influence=0.5)
    with pytest.raises(AttributeError):
        a.node_id = 99