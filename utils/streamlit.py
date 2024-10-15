import pickle
from typing import Literal

from graph_tool import Graph, PropertyMap, load_graph  # noqa
import numpy as np
import pandas as pd
import streamlit as st

from transformers import AutoProcessor

from models.transparent_models import TransparentLlava, TransparentMolmo, TransparentPixtral

# An array of simple and distinct colors (names) for the nodes

colors = [
    "red", "blue", "green", "yellow", "purple", "orange", "brown", "pink", "cyan", "magenta", "lime", "indigo",
    "teal", "lavender", "maroon", "navy", "olive", "gray", "white", "black", "silver", "gold", "crimson", "coral",
    "chocolate", "chartreuse", "cadetblue", "burlywood", "blueviolet", "bisque", "beige", "azure", "aquamarine",
    "antiquewhite", "aliceblue", "aqua", "aquamarine"
]


@st.cache_resource
def read_metric_results(path: str):
    return (
        pd.read_parquet(f"{path}/results_table.parquet"),
        pickle.load(open(f"{path}/node_layers_dict.pkl", "rb")),
        pickle.load(open(f"{path}/modality_centrality_results.pkl", "rb")),
        pickle.load(open(f"{path}/graph_metrics_results.pkl", "rb")),
    )


@st.cache_resource
def read_graphs(path: str, idx, num_graphs) -> tuple[list[Graph], list[Graph]]:
    simple_graphs = []
    full_graphs = []
    for i in range(num_graphs):
        simple_graphs.append(load_graph(f"{path}/simple_graphs/{idx}_{i}.gt"))
        full_graphs.append(load_graph(f"{path}/full_graphs/{idx}_{i}.gt"))
    return simple_graphs, full_graphs


@st.cache_data
def get_image_dir(run_dir):
    image_paths = {
        "WhatsUp_": "/home/projects/shimon/agroskin/datasets/whatsup",
        "SEED": "/home/projects/shimon/Collaboration/seedbench/SEED-Bench-2-image",
        "COCO": "/home/projects/bagon/shared/coco/unlabeled2017"
    }
    for k, path in image_paths.items():
        if k in run_dir:
            return path
    raise ValueError(f"Dataset from run {run_dir} is not supported")


@st.cache_resource(hash_funcs={dict: id})
def process_centrality(centrality):
    txt_centrality = []
    img_centrality = []
    total_centrality = []
    intersection_centrality = []
    jaccard_similarity = []
    txt_only = []
    img_only = []

    for i in range(len(centrality['txt_centrality'])):
        txt_sample = centrality['txt_centrality'][i]
        img_sample = centrality['img_centrality'][i]
        txt_centrality.append([])
        img_centrality.append([])
        total_centrality.append([])
        intersection_centrality.append([])
        jaccard_similarity.append([])
        txt_only.append([])
        img_only.append([])
        for j, txt_tok, img_tok in zip(range(len(txt_sample)), txt_sample, img_sample):
            if (len(txt_tok) == 0) or (len(img_tok) == 0):
                jaccard_similarity[-1].append(np.nan)
                txt_centrality[-1].append(np.nan)
                img_centrality[-1].append(np.nan)
                total_centrality[-1].append(np.nan)
                intersection_centrality[-1].append(np.nan)
                txt_only[-1].append(np.nan)
                img_only[-1].append(np.nan)
                continue
            total_centrality[-1].append((txt_tok + img_tok))
            intersection_centrality[-1].append(np.where(txt_tok > img_tok, img_tok, txt_tok))

            txt_only[-1].append(txt_tok - intersection_centrality[-1][-1])
            img_only[-1].append(img_tok - intersection_centrality[-1][-1])

            jaccard_similarity[-1].append(
                (intersection_centrality[-1][-1].sum() / total_centrality[-1][-1].sum()).item())

            if intersection_centrality[-1][-1].max() != 0:
                intersection_centrality[-1][-1] = intersection_centrality[-1][-1] / min(txt_tok.max(), img_tok.max())

            total_centrality[-1][-1] = total_centrality[-1][-1] / total_centrality[-1][-1].max()

            txt_centrality[-1].append((txt_tok / txt_tok.max()) if txt_tok.max() != 0 else txt_tok)
            img_centrality[-1].append((img_tok / img_tok.max()) if img_tok.max() != 0 else img_tok)
            txt_only[-1][-1] = (txt_only[-1][-1] / txt_tok.max()) if txt_tok.max() != 0 else txt_only[-1][-1]
            img_only[-1][-1] = (img_only[-1][-1] / img_tok.max()) if img_tok.max() != 0 else img_only[-1][-1]

    return txt_centrality, img_centrality, total_centrality, intersection_centrality, txt_only, img_only, jaccard_similarity


@st.cache_resource
def load_processor(model_id):
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


@st.cache_data
def tokens_to_strings(token_ids, model_name, _processor):
    match model_name:
        case "llava":
            return TransparentLlava.tokens_to_strings(token_ids, _processor.tokenizer)
        case "molmo":
            return TransparentMolmo.tokens_to_strings(token_ids, _processor.tokenizer)
        case "pixtral":
            return TransparentPixtral.tokens_to_strings(token_ids, _processor.tokenizer)
        case _:
            raise ValueError(f"Unsupported model '{model_name}'")


@st.cache_resource(hash_funcs={Graph: id})
def get_edge_list(graph: Graph):
    res = []
    for s, t, w in graph.get_edges([graph.ep.weight]):
        source_id_beg, source_id_end = graph.vp.ids[int(s)].split("_")
        target_id_beg, target_id_end = graph.vp.ids[int(t)].split("_")
        if source_id_beg[0] != "X":
            res.append({"source": f"{source_id_beg[0]}{int(source_id_beg[1:]) - 1}_{source_id_end}",
                        "target": f"{target_id_beg[0]}{int(target_id_beg[1:]) - 1}_{target_id_end}",
                        "weight": w.item()})
        else:
            res.append({"source": f"{source_id_beg}_{source_id_end}",
                        "target": f"{target_id_beg[0]}{int(target_id_beg[1:]) - 1}_{target_id_end}",
                        "weight": w.item()})
    return res


@st.cache_resource(hash_funcs={list[Graph]: id, list[PropertyMap | np.ndarray]: id})
def create_node_style_map(graphs: list[Graph], metrics: list[PropertyMap | np.ndarray],
                          mode: Literal["value", "cluster"]):
    style_map: list[list] = []
    match mode:
        case "value":
            for simple_graph, metric in zip(graphs, metrics):
                style_map.append([])
                for v, layer_num, token_num in simple_graph.get_vertices(
                        vprops=[simple_graph.vp.layer_num, simple_graph.vp.token_num]):
                    if layer_num > 0:
                        metric_value = metric[v].item()
                        color = f"rgb({int(255 * (1 - metric_value))}, {int(255 * (1 - metric_value))}, 255)"
                        style_map[-1].append([f"{layer_num - 1}_{token_num}", [color, metric_value]])
        case "cluster":
            class_color_mapping: dict[int, str] = {}
            color_idx = 0
            for simple_graph, metric in zip(graphs, metrics):
                style_map.append([])
                for v, layer_num, token_num in simple_graph.get_vertices(
                        vprops=[simple_graph.vp.layer_num, simple_graph.vp.token_num]):
                    if layer_num > 0:
                        cluster = metric[v]
                        if cluster not in class_color_mapping:
                            class_color_mapping[cluster] = colors[color_idx % len(colors)]
                            color_idx += 1
                            if color_idx == len(colors):
                                print("WARNING: Too many clusters, some colors will be reused")
                        style_map[-1].append([f"{layer_num - 1}_{token_num}", [class_color_mapping[cluster], cluster]])
    return style_map


def plot_for_item(metrics, nodes):
    mean, m_len = [], []
    for n, metric in zip(nodes, metrics):
        mean.append([])
        m_len.append([])
        for layer in range(len(n)):
            mean[-1].append(np.mean(metric[n[layer]]))
            m_len[-1].append(len(n[layer]))

    mean = np.array(mean).transpose(1, 0)
    m_len = np.array(m_len).transpose(1, 0)

    return mean, m_len
