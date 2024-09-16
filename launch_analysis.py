import concurrent.futures
from functools import partial
from numbers import Number
import os
from pathlib import Path
import pickle
import json

from graph_tool import Graph  # noqa
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoProcessor

from flow.graphs import build_graph_from_contributions
from utils.misc import set_random_seed, get_image_token_boundaries
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
    processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    vertex_metrics, graph_metrics = create_metrics(cfg)

    full_graph_dict, simple_graph_dict, node_layers_dict = {}, {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                             desc="Building graphs"):

            attn, attn_res, ffn, ffn_res = torch.load(base_dir / "graphs" / f"{row.idx}_graph_tensors.pkl",
                                                      weights_only=True)

            img_begin, img_end = get_image_token_boundaries(processor.tokenizer, row.prompt, cfg.model.img_token_id,
                                                            cfg.model.img_dims)

            tokenized_prompt = processor.tokenizer(row.prompt)["input_ids"]
            generated_token_ids = row.generated_ids[len(tokenized_prompt):]
            generated_len = len(generated_token_ids)

            build_graph_for = partial(build_graph_from_contributions, attn=attn, attn_res=attn_res, ffn=ffn,
                                      ffn_res=ffn_res,
                                      img_begin=img_begin, img_end=img_end)

            results = list(
                executor.map(
                    build_graph_for,
                    [cfg.graph_threshold] * generated_len,
                    range(-generated_len, 0)
                )
            )

            full_graphs, simple_graphs, node_layers = [], [], []

            for full_graph, simple_graph, node_layer in results:
                full_graphs.append(full_graph)
                simple_graphs.append(simple_graph)
                node_layers.append(node_layer)

            full_graph_dict[idx] = full_graphs
            simple_graph_dict[idx] = simple_graphs
            node_layers_dict[idx] = node_layers


        for metric in vertex_metrics:
            vertex_metric_results: dict[str, dict[int, list[np.ndarray]]] = {lab: {} for lab in metric.labels}
            for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                                 desc=f"Computing {metric.name}"): #type: int, pd.Series

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

        graph_metrics_results: dict[str, dict[int, list[Number]]] = {}
        for metric in graph_metrics:
            for lab in metric.labels:
                graph_metrics_results[lab] = {}
            for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                                 desc=f"Computing {metric.name}"):  # type: int, pd.Series
                simple_graphs: list[Graph] = simple_graph_dict[idx]

                graph_metrics_results[lab][idx] = []

                results = list(
                    executor.map(
                        metric,
                        simple_graphs
                    )
                )
                for result_arrays in results:
                    for lab, val in zip(metric.labels, result_arrays):
                        graph_metrics_results[lab][idx].append(val)

        pickle.dump(graph_metrics_results, open(base_dir / "graph_metrics_results.pkl", "wb"))

    # Save full graphs and simple graphs using the gt format
    simple_graph_dir = base_dir / "simple_graphs"
    simple_graph_dir.mkdir(parents=True, exist_ok=True)
    full_graph_dir = base_dir / "full_graphs"
    full_graph_dir.mkdir(parents=True, exist_ok=True)
    for idx, simple_graphs in tqdm(simple_graph_dict.items(), desc="Saving simple graphs"):
        for i, graph in enumerate(simple_graphs):
            graph.save(str((simple_graph_dir / f"{idx}_{i}.gt").resolve()))

    for idx, full_graphs in tqdm(full_graph_dict.items(), desc="Saving full graphs"):
        for i, graph in enumerate(full_graphs):
            graph.save(str((full_graph_dir / f"{idx}_{i}.gt").resolve()))

    # pickle.dump(full_graph_dict, open(base_dir / "full_graph_dict.pkl", "wb"))
    # pickle.dump(simple_graph_dict, open(base_dir / "simple_graph_dict.pkl", "wb"))
    pickle.dump(node_layers_dict, open(base_dir / "node_layers_dict.json", "wb"))


if __name__ == '__main__':
    run()
