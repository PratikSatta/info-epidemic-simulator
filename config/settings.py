from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    """
    Centralised configuration dataclass.
    All default values live here — never scattered as magic numbers in code.

    Usage:
        from config.settings import SimulatorConfig
        cfg = SimulatorConfig()
        sim = Simulator(network, beta=cfg.beta, gamma=cfg.gamma)
    """
    graph_type:    str   = "barabasi_albert"
    n_nodes:       int   = 300
    ba_m:          int   = 3
    ws_k:          int   = 6
    ws_p:          float = 0.1
    er_p:          float = 0.01
    beta:          float = 0.3
    gamma:         float = 0.05
    sigma:         float = 0.2
    use_seir:      bool  = True
    content_bias:  float = 0.0
    max_steps:     int   = 200
    n_seeds:       int   = 3
    seed_strategy: str   = "random"
    results_dir:   str   = "data/results"
    graphs_dir:    str   = "outputs/graphs"