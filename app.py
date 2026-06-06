import streamlit as st
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from src.networks import create_network
from src.simulator import Simulator, SimulationResult
from src.analysis import SimulationAnalyzer
from src.experiment import ExperimentRunner

st.set_page_config(page_title="Information Epidemic Simulator", layout="wide")
st.title("Information Epidemic Simulator")
st.caption("Simulate how rumors and fake news spread through social networks.")

with st.sidebar:
    st.header("Network")
    graph_type = st.selectbox("Topology", ["barabasi_albert", "watts_strogatz", "erdos_renyi"])
    n_nodes    = st.slider("Nodes", 100, 1000, 300, step=50)

    st.subheader("Graph parameters")
    if graph_type == "barabasi_albert":
        graph_kwargs = {"m": st.slider("Edges per node (m)", 1, 10, 3)}
    elif graph_type == "watts_strogatz":
        graph_kwargs = {
            "k": st.slider("Neighbours in ring (k)", 2, 20, 6, step=2),
            "p": st.slider("Rewiring probability", 0.0, 1.0, 0.1, step=0.05),
        }
    else:
        graph_kwargs = {"p": st.slider("Edge probability (p)", 0.001, 0.05, 0.01)}

    st.divider()
    st.header("Simulation")
    beta          = st.slider("Transmission rate (beta)", 0.01, 1.0, 0.3, step=0.01)
    gamma         = st.slider("Recovery rate (gamma)", 0.01, 0.5, 0.05, step=0.01)
    n_seeds       = st.slider("Initial spreaders", 1, 20, 3)
    seed_strategy = st.selectbox("Seed strategy", ["random", "high_degree", "high_pagerank"])
    use_seir      = st.checkbox("Use SEIR model", value=True)
    content_bias  = st.slider("Content bias", -1.0, 1.0, 0.0, step=0.1)
    max_steps     = st.slider("Max timesteps", 50, 500, 200, step=50)

    st.divider()
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Simulation", "Network Analysis", "Compare Experiments"])

if run_btn:
    with st.spinner("Building network..."):
        # POLYMORPHISM: create_network returns whichever SocialNetwork subclass matches
        network = create_network(graph_type, n=n_nodes, **graph_kwargs)
        network.build()
        summary = network.summary()

    with st.spinner("Running simulation..."):
        sim = Simulator(
            network=network, beta=beta, gamma=gamma,
            n_seeds=n_seeds, seed_strategy=seed_strategy,
            use_seir=use_seir, content_bias=content_bias, max_steps=max_steps,
        )
        result: SimulationResult = sim.run()

    with st.spinner("Analysing..."):
        analyzer = SimulationAnalyzer(result)
        metrics  = analyzer.analyze()

    st.session_state.update({
        "result": result, "analyzer": analyzer,
        "metrics": metrics, "summary": summary, "ran": True,
    })

if st.session_state.get("ran"):
    result   = st.session_state["result"]
    analyzer = st.session_state["analyzer"]
    metrics  = st.session_state["metrics"]
    summary  = st.session_state["summary"]

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak infected",  f"{metrics['peak_infected_pct']}%")
        c2.metric("Attack rate",    f"{metrics['attack_rate_pct']}%")
        c3.metric("Time to peak",   f"{metrics['time_to_peak']} steps")
        c4.metric("Estimated R0",   metrics["r0_estimate"])

        st.subheader("Spread over time")
        st.pyplot(analyzer.plot_curves(title=f"Spread -- {summary['graph_type']}"))

        with st.expander("Network structure"):
            cols = st.columns(4)
            for i, (k, v) in enumerate(list(summary.items())[1:5]):
                cols[i].metric(k.replace("_", " ").title(), v)

        st.subheader("Network visualisation")
        path = analyzer.export_pyvis("outputs/graphs/current.html")
        st.components.v1.html(open(path, encoding="utf-8").read(), height=500)

    with tab2:
        st.subheader("Superspreaders")
        st.dataframe(
            pd.DataFrame([{"Node ID": n, "Influence": s}
                          for n, s in metrics["superspreaders"]]),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Run experiment grid")
        if st.button("Run comparison (~60 sec)"):
            runner = ExperimentRunner(
                param_grid={
                    "graph_type":    ["barabasi_albert", "watts_strogatz", "erdos_renyi"],
                    "seed_strategy": ["random", "high_degree", "high_pagerank"],
                    "beta":          [0.15, 0.30, 0.50],
                },
                n_nodes=200, max_steps=100, n_repeats=2,
            )
            with st.spinner("Running..."):
                runner.run_all(verbose=False)
            runner.save_csv("data/results/comparison.csv")
            st.dataframe(runner.to_dataframe(), use_container_width=True)

else:
    with tab1:
        st.info("Configure parameters in the sidebar and click Run Simulation.")