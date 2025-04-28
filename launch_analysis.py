import concurrent.futures
from functools import partial
import os
from pathlib import Path
import pickle

from graph_tool import Graph  # noqa
from graph_tool import load_graph
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm.auto import tqdm
import wandb

from flow.graphs import build_graph_from_contributions
from utils.misc import set_random_seed
from utils.setup import create_metrics


@hydra.main(config_path="configs", config_name="analysis_config", version_base=None)
def run(cfg: DictConfig):
    import torch
    set_random_seed(cfg.seed)

    if cfg.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        print(OmegaConf.to_yaml(cfg))

    base_dir = Path(cfg.results_dir) / cfg.run_dir
    results_table = pd.read_parquet(base_dir / "results_table.parquet")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    simple_graph_dir = base_dir / "simple_graphs"
    simple_graph_dir.mkdir(parents=True, exist_ok=True)
    full_graph_dir = base_dir / "full_graphs"
    full_graph_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.run_dir, group="misc",
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    vertex_metrics, graph_metrics = create_metrics(cfg)

    simple_graph_dict, node_layers_dict, node_pos_dict = {}, {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        if cfg.construct_graphs:
            graph_metrics_results: dict[str, dict[int, list]] = {}
            for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                                 desc="Building graphs"):

                attn, attn_res, ffn, ffn_res = torch.load(base_dir / "graphs" / f"{row.idx}_graph_tensors.pkl",
                                                          weights_only=True)

                img_begin, img_end, generated_len = row.img_begin, row.img_end, row.num_generated_tokens

                build_graph_for = partial(build_graph_from_contributions, attn=attn, attn_res=attn_res, ffn=ffn,
                                          ffn_res=ffn_res, img_begin=img_begin, img_end=img_end)

                results = list(
                    executor.map(
                        build_graph_for,
                        [cfg.graph_threshold] * generated_len,
                        range(-generated_len, 0)
                    )
                )

                simple_graphs, node_layers, node_pos = [], [], []

                for i, (full_graph, simple_graph, node_layer) in enumerate(results):
                    full_graph.save(str((full_graph_dir / f"{idx}_{i}.gt").resolve()))
                    simple_graphs.append(simple_graph)
                    node_layers.append(node_layer)
                    node_pos.append((simple_graph.vp.layer_num.a, simple_graph.vp.token_num.a))

                simple_graph_dict[idx] = simple_graphs
                node_layers_dict[idx] = node_layers
                node_pos_dict[idx] = node_pos
        else:
            graph_metrics_results = pickle.load(open(base_dir / "graph_metrics_results.pkl", "rb"))
            node_layers_dict = pickle.load(open(base_dir / "node_layers_dict.pkl", "rb"))
            node_pos_dict = pickle.load(open(base_dir / "node_pos_dict.pkl", "rb"))
            for idx in tqdm(results_table.idx, desc="Loading graphs"):
                simple_graphs, node_layers = [], []
                for i in range(len(node_layers_dict[idx])):
                    simple_graphs.append(load_graph(str(base_dir / "simple_graphs" / f"{idx}_{i}.gt")))

                simple_graph_dict[idx] = simple_graphs

        for metric in vertex_metrics:
            vertex_metric_results: dict[str, dict[int, list[np.ndarray]]] = {lab: {} for lab in metric.labels}
            for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                                 desc=f"Computing {metric.name}"):  # type: int, pd.Series

                for lab in metric.labels:
                    vertex_metric_results[lab][idx] = []

                simple_graphs: list[Graph] = simple_graph_dict[idx]
                node_layers = node_layers_dict[idx]

                results = list(
                    executor.map(
                        metric,
                        simple_graphs,
                        node_layers
                    )
                )
                for i, (graph, result_arrays) in enumerate(results):
                    simple_graphs[i] = graph
                    for lab, arr in zip(metric.labels, result_arrays):
                        vertex_metric_results[lab][idx].append(arr)

            pickle.dump(vertex_metric_results, open(base_dir / f"{metric.name}_results.pkl", "wb"))

        for metric in graph_metrics:
            for lab in metric.labels:
                graph_metrics_results[lab] = {}
            for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                                 desc=f"Computing {metric.name}"):  # type: int, pd.Series
                simple_graphs: list[Graph] = simple_graph_dict[idx]

                for lab in metric.labels:
                    graph_metrics_results[lab][idx] = []

                results = list(
                    map(
                        metric,
                        simple_graphs
                    )
                )
                for result_arrays in results:
                    for lab, val in zip(metric.labels, result_arrays):
                        graph_metrics_results[lab][idx].append(val)

        pickle.dump(graph_metrics_results, open(base_dir / "graph_metrics_results.pkl", "wb"))

    # Save full graphs and simple graphs using the gt format

    for idx, simple_graphs in tqdm(simple_graph_dict.items(), desc="Saving simple graphs"):
        for i, graph in enumerate(simple_graphs):
            graph.save(str((simple_graph_dir / f"{idx}_{i}.gt").resolve()))

    # for idx, full_graphs in tqdm(full_graph_dict.items(), desc="Saving full graphs"):
    #     for i, graph in enumerate(full_graphs):
    #         graph.save(str((full_graph_dir / f"{idx}_{i}.gt").resolve()))

    # pickle.dump(full_graph_dict, open(base_dir / "full_graph_dict.pkl", "wb"))
    # pickle.dump(simple_graph_dict, open(base_dir / "simple_graph_dict.pkl", "wb"))
    pickle.dump(node_layers_dict, open(base_dir / "node_layers_dict.pkl", "wb"))
    pickle.dump(node_pos_dict, open(base_dir / "node_pos_dict.pkl", "wb"))


if __name__ == '__main__':
    run()
