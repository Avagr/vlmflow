import pickle
from typing import LiteralString, Literal

from graph_tool import Graph, PropertyMap  # noqa
import numpy as np
import pandas as pd
import streamlit as st


# An array of simple and distinct colors (names) for the nodes

colors = [
    "red", "blue", "green", "yellow", "purple", "orange", "brown", "pink", "cyan", "magenta", "lime", "indigo",
    "teal", "lavender", "maroon", "navy", "olive", "gray", "white", "black", "silver", "gold", "crimson", "coral",
    "chocolate", "chartreuse", "cadetblue", "burlywood", "blueviolet", "bisque", "beige", "azure", "aquamarine",
    "antiquewhite", "aliceblue", "aqua", "aquamarine"
]



@st.cache_resource
def read_results(path: str):
    return (
        pd.read_parquet(f"{path}/results_table.parquet"),
        pickle.load(open(f"{path}/full_graph_dict.pkl", "rb")),
        pickle.load(open(f"{path}/simple_graph_dict.pkl", "rb")),
        pickle.load(open(f"{path}/node_layers_dict.pkl", "rb")),
        pickle.load(open(f"{path}/modality_ratio_results.pkl", "rb"))['img_contrib'],
        pickle.load(open(f"{path}/modality_centrality_results.pkl", "rb"))
    )


@st.cache_resource
def process_centrality(centrality):
    txt_centrality = []
    img_centrality = []
    total_centrality = []
    intersection_centrality = []
    jaccard_similarity = []
    txt_only = []
    img_only = []

    for i, txt_sample, img_sample in zip(range(len(centrality['txt_centrality'])), centrality['txt_centrality'],
                                         centrality['img_centrality']):
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

            txt_centrality[-1].append(txt_tok / txt_tok.max())
            img_centrality[-1].append(img_tok / img_tok.max())
            txt_only[-1][-1] = txt_only[-1][-1] / txt_tok.max()
            img_only[-1][-1] = img_only[-1][-1] / img_tok.max()

    return txt_centrality, img_centrality, total_centrality, intersection_centrality, txt_only, img_only, jaccard_similarity


@st.cache_resource
def load_processor(model_id):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)


@st.cache_data
def tokens_to_strings(token_ids, _processor):
    res = []
    img_beg, img_end = None, None
    for i, tok in enumerate(token_ids):
        if tok == 32000:
            res.extend([f"I_{img_count}" for img_count in range(576)])
            img_beg = i
            img_end = i + 576
        else:
            str_repr = _processor.tokenizer.decode(tok)
            match str_repr:
                case '':
                    str_repr = "_"
                case '\n':
                    str_repr = "\\n"
            res.append(str_repr)
    return res, img_beg, img_end


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
def create_node_style_map(graphs: list[Graph], metrics: list[PropertyMap | np.ndarray], mode: Literal["value", "cluster"]):
    style_map: list[list] = []
    match mode:
        case "value":
            for simple_graph, metric in zip(graphs, metrics):
                style_map.append([])
                for v, layer_num, token_num in simple_graph.get_vertices(
                        vprops=[simple_graph.vp.layer_num, simple_graph.vp.token_num]):
                    if layer_num > 0:
                        metric_value = metric[v]
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
        for layer in range(41):
            mean[-1].append(np.mean(metric[n[layer]]))
            m_len[-1].append(len(n[layer]))

    mean = np.array(mean).transpose(1, 0)
    m_len = np.array(m_len).transpose(1, 0)

    return mean, m_len
