import concurrent.futures
import os
from pathlib import Path
import pickle
from functools import partial

from graph_tool import Graph
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoProcessor

from flow.analysis import modality_ratio
from flow.graphs import build_graph_from_contributions
from utils.misc import set_random_seed, get_image_token_boundaries


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

    full_graph_dict, simple_graph_dict, node_layers_dict = {}, {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                             desc="Building graphs"):

            attn, attn_res, ffn, ffn_res = torch.load(base_dir / "graphs" / f"{row.idx}_graph_tensors.pkl",
                                                      weights_only=True)

            img_begin, img_end = get_image_token_boundaries(processor.tokenizer, row.prompt, cfg.model.img_token_id,
                                                            cfg.model.img_dims)

            tokenized_prompt = processor.tokenizer(row.prompt)["input_ids"]
            generated_token_ids = row.generated_ids[len(tokenized_prompt):]
            generated_len = len(generated_token_ids)

            build_graph_for = partial(build_graph_from_contributions, attn=attn, attn_res=attn_res, ffn=ffn, ffn_res=ffn_res,
                                      img_begin=img_begin, img_end=img_end)


            results = list(executor.map(build_graph_for,
                                        [cfg.graph_threshold] * generated_len,
                                        range(-generated_len, 0)
                           ))

            full_graphs, simple_graphs, node_layers = [], [], []


            for full_graph, simple_graph, node_layer in results:
                full_graphs.append(full_graph)
                simple_graphs.append(simple_graph)
                node_layers.append(node_layer)

            full_graph_dict[idx] = full_graphs
            simple_graph_dict[idx] = simple_graphs
            node_layers_dict[idx] = node_layers

    simple_graph_metrics = [modality_ratio]  # TODO make configurable
    for metric in simple_graph_metrics:
        metric_results = []
        for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                             desc=f"Computing {metric.__name__}"):
            simple_graphs: list[Graph] = simple_graph_dict[idx]
            node_layers = node_layers_dict[idx]
            metric_results.append([metric(graph, node_layer) for graph, node_layer in zip(simple_graphs, node_layers)])
        pickle.dump(metric_results, open(base_dir / f"{metric.__name__}_results.pkl", "wb"))

    pickle.dump(full_graph_dict, open(base_dir / "full_graph_dict.pkl", "wb"))
    pickle.dump(simple_graph_dict, open(base_dir / "simple_graph_dict.pkl", "wb"))
    pickle.dump(node_layers_dict, open(base_dir / "node_layers_dict.pkl", "wb"))


if __name__ == '__main__':
    run()
