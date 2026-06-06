import networkx as nx
from src.base import SocialNetwork


class BarabasiAlbertNetwork(SocialNetwork):
    """
    Scale-free network via preferential attachment.

    INHERITANCE: Inherits from SocialNetwork.
    POLYMORPHISM: Overrides build() and graph_type_name.

    Models platforms like Twitter where a few hubs have enormous reach
    and most users have very few connections.
    """

    def __init__(self, n: int = 300, m: int = 3, seed: int = 42):
        super().__init__(n=n, seed=seed)  # call parent __init__
        self._m = m  # edges each new node attaches — higher = denser hubs

    @property
    def graph_type_name(self) -> str:
        return "Barabasi-Albert"

    def build(self) -> None:
        """
        Grows graph by adding nodes one at a time. Each new node attaches
        to m existing nodes with probability proportional to their current
        degree — 'preferential attachment' or 'the rich get richer'.
        """
        self._G = nx.barabasi_albert_graph(n=self._n, m=self._m, seed=self._seed)
        self._extract_largest_component()  # inherited from SocialNetwork

    def __repr__(self) -> str:
        status = f"{self._G.number_of_nodes()} nodes, m={self._m}" if self._G else "not built"
        return f"BarabasiAlbertNetwork({status})"


class WattsStrogatzNetwork(SocialNetwork):
    """
    Small-world network via random rewiring.

    INHERITANCE: Inherits from SocialNetwork.
    POLYMORPHISM: Overrides build() and graph_type_name.

    Models tight friend groups still loosely connected to the wider world.
    High clustering coefficient + short average path length.
    """

    def __init__(self, n: int = 300, k: int = 6, p: float = 0.1, seed: int = 42):
        super().__init__(n=n, seed=seed)
        self._k = k    # each node connects to k nearest neighbours in ring
        self._p = p    # probability of rewiring each edge

    @property
    def graph_type_name(self) -> str:
        return "Watts-Strogatz"

    def build(self) -> None:
        """
        Starts with a regular ring lattice where everyone connects to their
        k nearest neighbours on each side. Then randomly rewires each edge
        with probability p. p=0 is a pure lattice; p=1 is effectively random.
        """
        self._G = nx.watts_strogatz_graph(
            n=self._n, k=self._k, p=self._p, seed=self._seed
        )
        self._extract_largest_component()

    def __repr__(self) -> str:
        status = (
            f"{self._G.number_of_nodes()} nodes, k={self._k}, p={self._p}"
            if self._G else "not built"
        )
        return f"WattsStrogatzNetwork({status})"


class ErdosRenyiNetwork(SocialNetwork):
    """
    Random graph — each pair connected independently with probability p.

    INHERITANCE: Inherits from SocialNetwork.
    POLYMORPHISM: Overrides build() and graph_type_name.

    Used as a research baseline. No hubs, no clustering. If findings hold
    on ER graphs, they are topology-independent.
    """

    def __init__(self, n: int = 300, p: float = 0.01, seed: int = 42):
        super().__init__(n=n, seed=seed)
        self._p = p  # edge probability — 0.01 gives ~3 edges per node on avg

    @property
    def graph_type_name(self) -> str:
        return "Erdos-Renyi"

    def build(self) -> None:
        """Each of the n*(n-1)/2 possible edges is included with probability p."""
        self._G = nx.erdos_renyi_graph(n=self._n, p=self._p, seed=self._seed)
        self._extract_largest_component()

    def __repr__(self) -> str:
        status = (
            f"{self._G.number_of_nodes()} nodes, p={self._p}"
            if self._G else "not built"
        )
        return f"ErdosRenyiNetwork({status})"


def create_network(graph_type: str, **kwargs) -> SocialNetwork:
    """
    Factory function demonstrating POLYMORPHISM.

    Returns the correct SocialNetwork subclass by name.
    The caller only knows it has a SocialNetwork — not which specific subclass.

    Usage:
        net = create_network("barabasi_albert", n=300, m=3)
        net.build()
        summary = net.summary()  # works regardless of which subclass

    This is polymorphism in practice: one function name, many possible
    behaviours depending on what object is returned.
    """
    registry = {
        "barabasi_albert": BarabasiAlbertNetwork,
        "watts_strogatz":  WattsStrogatzNetwork,
        "erdos_renyi":     ErdosRenyiNetwork,
    }
    if graph_type not in registry:
        raise ValueError(
            f"Unknown graph_type '{graph_type}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[graph_type](**kwargs)