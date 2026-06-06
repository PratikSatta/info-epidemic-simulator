from abc import ABC, abstractmethod
import networkx as nx
from typing import Optional
import numpy as np


class SocialNetwork(ABC):
    """
    Abstract Base Class for all social network topologies.

    ABSTRACTION: This class cannot be instantiated directly.
    Attempting SocialNetwork() raises TypeError.

    Subclasses MUST implement:
        build()           -- construct the nx.Graph
        graph_type_name   -- return a string name for the topology

    Subclasses INHERIT (no need to reimplement):
        summary()         -- compute graph statistics
        G (property)      -- access the underlying networkx graph
        _extract_largest_component()  -- internal utility
    """

    def __init__(self, n: int = 300, seed: int = 42):
        self._n    = n
        self._seed = seed
        self._G: Optional[nx.Graph] = None  # private, only set after build()

    # ── Abstract interface (MUST be overridden by every subclass) ─────────────

    @abstractmethod
    def build(self) -> None:
        """
        Build the networkx graph and store in self._G.
        Must be called before using the network.
        """
        pass

    @property
    @abstractmethod
    def graph_type_name(self) -> str:
        """Return human-readable topology name, e.g. 'Barabasi-Albert'."""
        pass

    # ── Shared concrete behaviour (INHERITED by all subclasses) ──────────────

    @property
    def G(self) -> nx.Graph:
        """Provide read access to the graph. Raises if build() not called."""
        if self._G is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: graph not built yet. Call build() first."
            )
        return self._G

    @property
    def n(self) -> int:
        return self._n

    def summary(self) -> dict:
        """Compute key structural properties. Available after build()."""
        G       = self.G
        degrees = [d for _, d in G.degree()]
        return {
            "graph_type":     self.graph_type_name,
            "num_nodes":      G.number_of_nodes(),
            "num_edges":      G.number_of_edges(),
            "avg_degree":     round(float(np.mean(degrees)), 2),
            "max_degree":     max(degrees),
            "density":        round(nx.density(G), 4),
            "avg_clustering": round(nx.average_clustering(G), 4),
        }

    def _extract_largest_component(self) -> None:
        """
        Keep only the largest connected component.
        Called by subclass build() methods to ensure a connected graph.
        Protected (single underscore) — used by subclasses but not external code.
        """
        if not nx.is_connected(self._G):
            largest   = max(nx.connected_components(self._G), key=len)
            self._G   = self._G.subgraph(largest).copy()
            self._G   = nx.convert_node_labels_to_integers(self._G)

    def __repr__(self) -> str:
        status = f"{self._G.number_of_nodes()} nodes" if self._G else "not built"
        return f"{self.__class__.__name__}({status})"


class BaseAnalyzer(ABC):
    """
    Abstract Base Class for all analysis components.

    ABSTRACTION: Forces every analyzer subclass to implement analyze().
    Provides shared validation utility methods.
    """

    @abstractmethod
    def analyze(self) -> dict:
        """
        Compute metrics from simulation results.
        Every subclass must implement this and return a metrics dict.
        """
        pass

    def _validate_history(self, history: list) -> None:
        """Shared validation — raises if history is empty or None."""
        if not history:
            raise ValueError(
                "History is empty. The simulation must be run before analysis."
            )