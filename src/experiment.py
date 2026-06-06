import itertools
import pandas as pd
import mlflow
import os
from typing import List, Dict, Any

from src.networks import create_network
from src.simulator import Simulator, SimulationResult
from src.analysis import SimulationAnalyzer


class ExperimentRunner:
    """
    Manages a factorial grid of simulation experiments.

    Usage:
        runner = ExperimentRunner(param_grid={
            "graph_type":    ["barabasi_albert", "watts_strogatz", "erdos_renyi"],
            "beta":          [0.15, 0.30, 0.50],
            "seed_strategy": ["random", "high_pagerank"],
        })
        runner.run_all()
        runner.save_csv("data/results/comparison.csv")
        df = runner.to_dataframe()
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        n_nodes:    int = 300,
        max_steps:  int = 150,
        n_repeats:  int = 3,
    ):
        self._param_grid = param_grid
        self._n_nodes    = n_nodes
        self._max_steps  = max_steps
        self._n_repeats  = n_repeats
        self._results:   List[dict] = []

    def run_all(self, verbose: bool = True) -> None:
        """Run every parameter combination, n_repeats times each."""
        self._results = []
        keys   = list(self._param_grid.keys())
        combos = list(itertools.product(*self._param_grid.values()))
        total  = len(combos) * self._n_repeats

        mlflow.set_experiment("info-epidemic-simulator")

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            for rep in range(self._n_repeats):
                seed   = 42 + rep * 100
                run_n  = i * self._n_repeats + rep + 1
                if verbose:
                    print(f"  Run {run_n}/{total}: {params} | seed={seed}")
                record = self._run_one(params, seed=seed)
                record["repeat_seed"] = seed
                self._results.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._results:
            raise RuntimeError("No results yet. Call run_all() first.")
        return pd.DataFrame(self._results)

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        print(f"Results saved to {path}")

    def _run_one(self, params: dict, seed: int) -> dict:
        """Run a single parameter combination and return metrics dict."""
        graph_type = params.get("graph_type", "barabasi_albert")

        with mlflow.start_run():
            mlflow.log_params({**params, "seed": seed})

            network = create_network(graph_type, n=self._n_nodes, seed=seed)
            network.build()

            sim = Simulator(
                network       = network,
                beta          = params.get("beta", 0.3),
                gamma         = params.get("gamma", 0.05),
                n_seeds       = params.get("n_seeds", 3),
                seed_strategy = params.get("seed_strategy", "random"),
                max_steps     = self._max_steps,
                seed          = seed,
            )
            result: SimulationResult = sim.run()

            analyzer = SimulationAnalyzer(result)
            metrics  = analyzer.analyze()

            mlflow.log_metrics({
                "attack_rate_pct":   metrics["attack_rate_pct"],
                "peak_infected_pct": metrics["peak_infected_pct"],
                "time_to_peak":      metrics["time_to_peak"],
                "r0_estimate":       metrics["r0_estimate"],
            })

        return {**params, **metrics, "graph_type": network.graph_type_name}

    def __repr__(self) -> str:
        n = 1
        for v in self._param_grid.values():
            n *= len(v)
        return f"ExperimentRunner({n} combos x {self._n_repeats} repeats)"