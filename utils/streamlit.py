import pickle
from typing import Literal

from graph_tool import Graph, PropertyMap, load_graph, GraphView  # noqa
from graph_tool.centrality import closeness  # noqa
from graph_tool.topology import kcore_decomposition, shortest_distance  # noqa
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
        pickle.load(open(f"{path}/node_pos_dict.pkl", "rb")),
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
        "COCO": "/home/projects/bagon/shared/coco/unlabeled2017",
        "CLEVR_val": "/home/projects/shimon/agroskin/datasets/clevr/images/val",
        "AI2D": "/home/projects/shimon/agroskin/datasets/ai2d/images",
    }
    for k, path in image_paths.items():
        if k in run_dir:
            return path
    raise ValueError(f"Dataset from run {run_dir} is not supported")


@st.cache_resource(hash_funcs={list[Graph]: id})
def process_centrality(graphs: list[Graph], adaptive_normalization=True):
    txt_centrality, img_centrality, total_centrality, intersection_centrality, txt_difference, img_difference = [], [], [], [], [], []
    for graph in graphs:
        # if graph.num_vertices()
        txt_centr = graph.vp.txt_centrality.a
        img_centr = graph.vp.img_centrality.a
        if adaptive_normalization:
            img_tokens_mask = (graph.vp.img_contrib.a == 1) & (graph.vp.layer_num.a == 0)
            txt_tokens_mask = (graph.vp.txt_contrib.a == 1) & (graph.vp.layer_num.a == 0)
            img_norm = (graph.vp.token_num.a[img_tokens_mask, None] <= graph.vp.token_num.a[None]).sum(axis=0)
            txt_norm = (graph.vp.token_num.a[txt_tokens_mask, None] <= graph.vp.token_num.a[None]).sum(axis=0)
            total_norm = img_norm + txt_norm
            intersection_norm = np.minimum(txt_norm, img_norm)

        else:
            # Can do this because everything will converge at the final node
            txt_norm = txt_centr.max()
            img_norm = img_centr.max()
            total_norm = (txt_centr + img_centr).max()
            intersection_norm = min(txt_norm, img_norm)

        intersection = np.minimum(txt_centr, img_centr)

        txt_centrality.append(np.divide(txt_centr, txt_norm, out=np.zeros_like(txt_centr, dtype=float),
                                        where=txt_norm != 0))
        img_centrality.append(np.divide(img_centr, img_norm, out=np.zeros_like(txt_centr, dtype=float),
                                        where=img_norm != 0))
        total_centrality.append(np.divide(txt_centr + img_centr, total_norm, out=np.zeros_like(txt_centr, dtype=float),
                                          where=total_norm != 0))
        intersection_centrality.append(
            np.divide(intersection, intersection_norm, out=np.zeros_like(txt_centr, dtype=float),
                      where=intersection_norm != 0))
        txt_difference.append(np.divide(txt_centr - intersection, txt_norm, out=np.zeros_like(txt_centr, dtype=float),
                                        where=txt_norm != 0))
        img_difference.append(np.divide(img_centr - intersection, img_norm, out=np.zeros_like(txt_centr, dtype=float),
                                        where=img_norm != 0))
    return txt_centrality, img_centrality, total_centrality, intersection_centrality, txt_difference, img_difference


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


@st.cache_resource(hash_funcs={list[Graph]: id})
def get_kcore(graphs: list[Graph]):
    res = []
    for graph in graphs:
        res.append(kcore_decomposition(graph).a)
        res[-1] = res[-1] / res[-1].max()
    return res


@st.cache_resource(hash_funcs={list[Graph]: id})
def get_closeness_centrality(graphs: list[Graph]):
    res = []
    for graph in graphs:
        if graph.vp.closeness_centrality.a.max() == 0:
            res.append(np.zeros(graph.num_vertices()))
            continue
        res.append(graph.vp.closeness_centrality.a / graph.vp.closeness_centrality.a.max())
    return res


@st.cache_resource(hash_funcs={list[Graph]: id})
def get_token_closeness_centrality(graphs: list[Graph]):
    res = []
    for graph in graphs:
        vertices = graph.get_vertices(vprops=[graph.vp.token_num])
        all_img_vertices = vertices[(graph.vp.token_num.a >= graph.gp.img_begin) &
                                    (graph.vp.token_num.a < graph.gp.img_end) &
                                    (graph.vp.layer_num.a == 0)]
        if all_img_vertices.shape[0] == 0:
            res.append(np.zeros(graph.num_vertices()))
            continue
        distances = shortest_distance(GraphView(graph, reversed=True), directed=True).get_2d_array(
            pos=all_img_vertices[:, 0]
        )
        print(distances)
        normalized_inverse_distances = np.divide(distances.shape[0] - 1., distances,
                                                 where=np.isfinite(distances) & (distances != 0),
                                                 out=np.zeros_like(distances, dtype=float))

        res.append(normalized_inverse_distances.sum(axis=0))
        if res[-1].max() != 0:
            res[-1] = res[-1] / res[-1].max()
    return res


@st.cache_resource(hash_funcs={list[Graph]: id})
def get_closeness(graphs: list[Graph], dist_to_all_img_vertices=True):
    res = []
    for graph in graphs:
        vertices = graph.get_vertices(vprops=[graph.vp.token_num])
        if dist_to_all_img_vertices:
            all_img_vertices = vertices[
                (graph.vp.token_num.a >= graph.gp.img_begin) & (graph.vp.token_num.a < graph.gp.img_end)]
        else:
            all_img_vertices = vertices[
                (graph.vp.token_num.a >= graph.gp.img_begin) & (graph.vp.token_num.a < graph.gp.img_end) & (
                            graph.vp.layer_num == 0)]
        if all_img_vertices.shape[0] == 0:
            res.append(np.zeros(graph.num_vertices()))
            continue
        closeness_scores = shortest_distance(GraphView(graph, reversed=True), directed=True).get_2d_array(
            pos=all_img_vertices[:, 0]
        ).min(axis=0)
        reached = closeness_scores < 100000
        closeness_scores = np.where(reached, graph.gp.num_layers - closeness_scores, 0)
        res.append(closeness_scores)
        if res[-1].max() != 0:
            res[-1] = res[-1] / res[-1].max()
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
