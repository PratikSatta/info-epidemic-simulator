import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pyvis.network import Network
from typing import Optional

from src.base import BaseAnalyzer
from src.simulator import SimulationResult


class SimulationAnalyzer(BaseAnalyzer):
    """
    Computes metrics and produces visualisations from a SimulationResult.

    INHERITANCE: Inherits from BaseAnalyzer (ABC).
    ABSTRACTION: Implements the required analyze() method.

    Usage:
        result   = sim.run()
        analyzer = SimulationAnalyzer(result)
        metrics  = analyzer.analyze()
        fig      = analyzer.plot_curves()
        path     = analyzer.export_pyvis()
    """

    def __init__(self, result: SimulationResult):
        self._result  = result
        self._metrics: Optional[dict] = None  # cached after first call

    def analyze(self) -> dict:
        """
        Compute all summary metrics. Results are cached after the first call.

        Returns dict with keys:
            peak_infected_pct, time_to_peak, attack_rate_pct,
            r0_estimate, spread_velocity, superspreaders
        """
        self._validate_history(self._result.history)  # inherited from BaseAnalyzer

        if self._metrics is not None:
            return self._metrics  # return cached, avoid re-computing

        df = self._to_dataframe()
        n  = self._result.final_graph.number_of_nodes()

        peak_row          = df.loc[df["I"].idxmax()]
        peak_infected_pct = round(100 * peak_row["I"] / n, 2)
        time_to_peak      = int(peak_row["step"])

        final             = df.iloc[-1]
        attack_rate_pct   = round(100 * (final["I"] + final["R"]) / n, 2)

        avg_deg     = np.mean([d for _, d in self._result.final_graph.degree()])
        r0_estimate = round(
            (self._result.params.get("beta", 0.3) /
             self._result.params.get("gamma", 0.05)) * avg_deg, 2
        )

        i_before = df[df["step"] <= time_to_peak]["I"].values
        spread_velocity = (
            round(float(np.mean(np.diff(i_before))), 3)
            if len(i_before) > 1 else 0.0
        )

        self._metrics = {
            "peak_infected_pct": peak_infected_pct,
            "time_to_peak":      time_to_peak,
            "attack_rate_pct":   attack_rate_pct,
            "r0_estimate":       r0_estimate,
            "spread_velocity":   spread_velocity,
            "superspreaders":    self._find_superspreaders(),
        }
        return self._metrics

    def plot_curves(self, title: str = "Information Spread Over Time") -> plt.Figure:
        """Plot S, E, I, R curves over time. Returns matplotlib Figure."""
        df = self._to_dataframe().set_index("step")
        colors = {"S": "#378ADD", "E": "#EF9F27", "I": "#E24B4A", "R": "#1D9E75"}
        labels = {"S": "Susceptible", "E": "Exposed", "I": "Infected", "R": "Recovered"}

        fig, ax = plt.subplots(figsize=(9, 4))
        for col in ["S", "E", "I", "R"]:
            if col in df.columns and df[col].sum() > 0:
                ax.plot(df.index, df[col], label=labels[col],
                        color=colors[col], linewidth=2)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Number of agents")
        ax.set_title(title, fontweight="normal")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    def export_pyvis(
        self,
        output_path: str = "outputs/graphs/network.html",
        max_nodes:   int = 300,
    ) -> str:
        """Export interactive Pyvis HTML graph. Returns path to saved file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        G = self._result.final_graph
        colors = {"S": "#378ADD", "E": "#EF9F27", "I": "#E24B4A", "R": "#1D9E75"}

        all_nodes = list(G.nodes())
        if len(all_nodes) > max_nodes:
            all_nodes = list(np.random.choice(all_nodes, size=max_nodes, replace=False))
            G = G.subgraph(all_nodes).copy()

        net = Network(height="550px", width="100%", bgcolor="#ffffff")
        net.barnes_hut()
        for node in G.nodes():
            a = G.nodes[node]
            net.add_node(
                node,
                color=colors.get(a.get("state", "S"), "#888"),
                size=8 + 20 * a.get("influence", 0.0),
                title=(
                    f"Node {node}<br>State: {a.get('state','?')}<br>"
                    f"Trust: {a.get('trust', 0):.2f}<br>"
                    f"Influence: {a.get('influence', 0):.3f}"
                ),
            )
        for u, v in G.edges():
            net.add_edge(u, v, color="#cccccc", width=0.5)

        net.save_graph(output_path)
        return output_path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._result.history)

    def _find_superspreaders(self, top_k: int = 5) -> list:
        infected = [
            a for a in self._result.agents.values()
            if a.state in ("I", "R")
        ]
        infected.sort(key=lambda a: a.influence, reverse=True)
        return [(a.node_id, round(a.influence, 4)) for a in infected[:top_k]]

    def __repr__(self) -> str:
        return f"SimulationAnalyzer(steps={self._result.steps_run})"