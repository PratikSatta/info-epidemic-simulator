import copy
import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.base import SocialNetwork
from src.agent import Agent


@dataclass
class SimulationResult:
    """
    Data class holding the complete output of one simulation run.

    Using a dataclass makes the return value self-documenting and easy
    to pass between objects. All fields have clear types.
    """
    history:     List[dict]         # per-timestep state counts
    final_graph: nx.Graph           # networkx graph annotated with final states
    agents:      Dict[int, Agent]   # agent objects at end of simulation
    steps_run:   int
    converged:   bool               # True if stopped early (no spreaders left)
    params:      dict = field(default_factory=dict)


class Simulator:
    """
    Runs the discrete-time SEIR / SIR simulation on a SocialNetwork.

    ENCAPSULATION: _history, _agents, _rng are private.
    Public interface is only run() and reset().

    POLYMORPHISM: Accepts any SocialNetwork subclass (BA, WS, ER).
    The Simulator does not know or care which specific network it received.

    Usage:
        network = BarabasiAlbertNetwork(n=300)
        network.build()
        sim = Simulator(network, beta=0.3, gamma=0.05)
        result = sim.run()       # returns SimulationResult object
    """

    def __init__(
        self,
        network:       SocialNetwork,
        beta:          float = 0.3,
        gamma:         float = 0.05,
        sigma:         float = 0.2,
        content_bias:  float = 0.0,
        n_seeds:       int   = 3,
        seed_strategy: str   = "random",
        use_seir:      bool  = True,
        max_steps:     int   = 200,
        seed:          int   = 42,
    ):
        _ = network.G  # raises RuntimeError if network not built yet

        self._network       = network
        self._beta          = beta
        self._gamma         = gamma
        self._sigma         = sigma
        self._content_bias  = content_bias
        self._n_seeds       = n_seeds
        self._seed_strategy = seed_strategy
        self._use_seir      = use_seir
        self._max_steps     = max_steps
        self._rng           = np.random.default_rng(seed)

        # Private simulation state — not accessible from outside
        self._agents:  Dict[int, Agent] = {}
        self._history: List[dict]       = []

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self) -> SimulationResult:
        """
        Run the full simulation and return a SimulationResult object.
        Calling run() again resets state and starts fresh.
        """
        self._initialise_agents()
        self._history = []
        converged     = False

        for step in range(self._max_steps):
            counts = self._count_states()
            self._history.append({"step": step, **counts})

            if counts["I"] == 0 and counts.get("E", 0) == 0:
                converged = True
                break

            # Collect transitions BEFORE applying any — synchronous update
            transitions = self._compute_transitions(self._network.G)

            for node_id, new_state in transitions.items():
                self._agents[node_id].state = new_state

        self._history.append({"step": len(self._history), **self._count_states()})

        return SimulationResult(
            history     = list(self._history),
            final_graph = self._build_final_graph(),
            agents      = dict(self._agents),
            steps_run   = len(self._history),
            converged   = converged,
            params      = self._params_dict(),
        )

    def reset(self) -> None:
        """Clear all simulation state so run() can be called again fresh."""
        self._agents  = {}
        self._history = []

    # ── Private methods ────────────────────────────────────────────────────────

    def _initialise_agents(self) -> None:
        """Create Agent objects for every node and assign attributes."""
        G        = self._network.G
        pr       = nx.pagerank(G, alpha=0.85)
        pr_vals  = np.array(list(pr.values()))
        pr_min, pr_max = pr_vals.min(), pr_vals.max()

        for node in G.nodes():
            norm_inf = (
                (pr[node] - pr_min) / (pr_max - pr_min)
                if pr_max > pr_min else 0.5
            )
            self._agents[node] = Agent(
                node_id   = node,
                trust     = float(self._rng.beta(3.0, 2.0)),
                bias      = float(np.clip(self._rng.normal(0.0, 0.4), -1, 1)),
                influence = float(norm_inf),
                state     = "S",
            )

        seed_nodes  = self._select_seed_nodes()
        init_state  = "E" if self._use_seir else "I"
        for node in seed_nodes:
            self._agents[node].state = init_state

    def _select_seed_nodes(self) -> List[int]:
        """Select which nodes start as infected based on seed strategy."""
        nodes   = list(self._agents.keys())
        n       = min(self._n_seeds, len(nodes))

        if self._seed_strategy == "random":
            return list(self._rng.choice(nodes, size=n, replace=False))
        elif self._seed_strategy == "high_degree":
            return sorted(
                nodes, key=lambda v: self._network.G.degree(v), reverse=True
            )[:n]
        elif self._seed_strategy == "high_pagerank":
            return sorted(
                nodes, key=lambda v: self._agents[v].influence, reverse=True
            )[:n]
        raise ValueError(f"Unknown seed_strategy: {self._seed_strategy!r}")

    def _compute_transitions(self, G: nx.Graph) -> Dict[int, str]:
        """
        Determine all state transitions for this timestep.
        Must be computed BEFORE applying any to ensure synchronous update.
        """
        transitions = {}
        for node_id, agent in self._agents.items():
            if agent.is_susceptible():
                if self._receives_transmission(node_id, G):
                    transitions[node_id] = "E" if self._use_seir else "I"
            elif agent.state == "E":
                if self._rng.random() < self._sigma:
                    transitions[node_id] = "I"
            elif agent.is_infectious():
                if self._rng.random() < self._gamma:
                    transitions[node_id] = "R"
        return transitions

    def _receives_transmission(self, node_id: int, G: nx.Graph) -> bool:
        """
        Return True if this node receives a successful transmission
        from any infected neighbour this timestep.
        """
        receiver = self._agents[node_id]
        for nb_id in G.neighbors(node_id):
            sender = self._agents[nb_id]
            if not sender.is_infectious():
                continue
            sender_inf = 0.2 + 0.8 * sender.influence
            p = (self._beta * sender_inf
                 * receiver.trust
                 * receiver.bias_affinity(self._content_bias))
            if self._rng.random() < min(p, 1.0):
                return True
        return False

    def _count_states(self) -> dict:
        counts = {"S": 0, "E": 0, "I": 0, "R": 0}
        for agent in self._agents.values():
            counts[agent.state] += 1
        return counts

    def _build_final_graph(self) -> nx.Graph:
        G_out = self._network.G.copy()
        for node, agent in self._agents.items():
            G_out.nodes[node]["state"]     = agent.state
            G_out.nodes[node]["trust"]     = agent.trust
            G_out.nodes[node]["bias"]      = agent.bias
            G_out.nodes[node]["influence"] = agent.influence
        return G_out

    def _params_dict(self) -> dict:
        return {
            "graph_type":    self._network.graph_type_name,
            "beta":          self._beta,
            "gamma":         self._gamma,
            "sigma":         self._sigma,
            "use_seir":      self._use_seir,
            "n_seeds":       self._n_seeds,
            "seed_strategy": self._seed_strategy,
        }

    def __repr__(self) -> str:
        return (
            f"Simulator(network={self._network.graph_type_name}, "
            f"beta={self._beta}, gamma={self._gamma})"
        )