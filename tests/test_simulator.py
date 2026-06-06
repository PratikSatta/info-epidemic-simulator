from src.networks import BarabasiAlbertNetwork
from src.simulator import Simulator, SimulationResult

def test_run_returns_simulation_result():
    net = BarabasiAlbertNetwork(n=50)
    net.build()
    sim = Simulator(net, max_steps=20)
    assert isinstance(sim.run(), SimulationResult)

def test_state_counts_always_sum_to_n():
    net = BarabasiAlbertNetwork(n=80)
    net.build()
    sim = Simulator(net, max_steps=30)
    result = sim.run()
    n = result.final_graph.number_of_nodes()
    for record in result.history:
        total = sum(record.get(s, 0) for s in ["S", "E", "I", "R"])
        assert total == n

def test_reset_allows_clean_rerun():
    net = BarabasiAlbertNetwork(n=50)
    net.build()
    sim = Simulator(net, max_steps=10)
    sim.run()
    sim.reset()
    assert isinstance(sim.run(), SimulationResult)

def test_simulator_accepts_any_network_subclass():
    """Polymorphism: Simulator works with any SocialNetwork subclass."""
    from src.networks import WattsStrogatzNetwork, ErdosRenyiNetwork
    for cls in [BarabasiAlbertNetwork, WattsStrogatzNetwork, ErdosRenyiNetwork]:
        net = cls(n=50)
        net.build()
        result = Simulator(net, max_steps=10).run()
        assert isinstance(result, SimulationResult)