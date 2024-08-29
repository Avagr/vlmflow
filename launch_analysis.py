import os
from pathlib import Path
import pickle

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
    results_table = pd.read_pickle(base_dir / "results_table.pkl")
    processor = AutoProcessor.from_pretrained(cfg.model.processor_path)

    full_graph_dict, simple_graph_dict, node_layers_dict = {}, {}, {}

    for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                         desc="Building graphs"):

        attn, attn_res, ffn, ffn_res = torch.load(base_dir / "graphs" / f"{row.idx}_graph_tensors.pkl",
                                                  weights_only=True)
        if cfg.graph_for_last_token:
            top_token_num = -1
        else:
            raise NotImplementedError("Only last token is supported for now")

        img_begin, img_end = get_image_token_boundaries(processor.tokenizer, row.prompt, cfg.model.img_token_id,
                                                        cfg.model.img_dims)

        full_graph, simple_graph, node_layers = build_graph_from_contributions(cfg.graph_threshold, top_token_num, attn,
                                                                               attn_res, ffn, ffn_res,
                                                                               img_begin, img_end)

        full_graph_dict[idx] = full_graph
        simple_graph_dict[idx] = simple_graph
        node_layers_dict[idx] = node_layers

    simple_graph_metrics = [modality_ratio]  # TODO make configurable
    for metric in simple_graph_metrics:
        metric_results = []
        for idx, row in tqdm(results_table.iterrows(), total=len(results_table), disable=cfg.disable_tqdm,
                             desc=f"Computing {metric.__name__}"):
            simple_graph = simple_graph_dict[idx]
            node_layers = node_layers_dict[idx]
            metric_results.append(metric(simple_graph, node_layers))
        pickle.dump(metric_results, open(base_dir / f"{metric.__name__}_results.pkl", "wb"))

    pickle.dump(full_graph_dict, open(base_dir / "full_graph_dict.pkl", "wb"))
    pickle.dump(simple_graph_dict, open(base_dir / "simple_graph_dict.pkl", "wb"))
    pickle.dump(node_layers_dict, open(base_dir / "node_layers_dict.pkl", "wb"))

if __name__ == '__main__':
    run()
