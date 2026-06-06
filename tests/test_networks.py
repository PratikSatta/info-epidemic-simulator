import pytest
from src.networks import (
    BarabasiAlbertNetwork, WattsStrogatzNetwork, ErdosRenyiNetwork, create_network
)
from src.base import SocialNetwork
import networkx as nx

def test_ba_is_subclass_of_socialnetwork():
    assert issubclass(BarabasiAlbertNetwork, SocialNetwork)

def test_all_subclasses_inherit_socialnetwork():
    for cls in [BarabasiAlbertNetwork, WattsStrogatzNetwork, ErdosRenyiNetwork]:
        assert issubclass(cls, SocialNetwork)

def test_ba_builds_connected_graph():
    net = BarabasiAlbertNetwork(n=100)
    net.build()
    assert nx.is_connected(net.G)

def test_graph_type_names():
    assert BarabasiAlbertNetwork(n=50).graph_type_name == "Barabasi-Albert"
    assert WattsStrogatzNetwork(n=50).graph_type_name  == "Watts-Strogatz"
    assert ErdosRenyiNetwork(n=50).graph_type_name     == "Erdos-Renyi"

def test_unbuilt_graph_raises_runtime_error():
    net = BarabasiAlbertNetwork(n=50)
    with pytest.raises(RuntimeError):
        _ = net.G

def test_factory_returns_correct_class():
    assert isinstance(create_network("watts_strogatz", n=100), WattsStrogatzNetwork)

def test_polymorphism_same_interface():
    """All three subclasses expose the same interface -- polymorphism."""
    for cls in [BarabasiAlbertNetwork, WattsStrogatzNetwork, ErdosRenyiNetwork]:
        net = cls(n=50)
        net.build()
        assert "num_nodes" in net.summary()