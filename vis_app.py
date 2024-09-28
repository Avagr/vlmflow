from functools import partial
import pickle

from graph_tool import Graph, PropertyMap  # noqa
import graph_tool.all as gt  # noqa
import numpy as np
import numpy_indexed as npi
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from tqdm.auto import tqdm

from utils.streamlit import load_processor

if __name__ == "__main__":

    st.set_page_config(layout="wide")
    load_from_disk = st.checkbox("Load from disk", value=True)


    @st.cache_resource
    def load_data():
        dataset_metrics = {}
        for run_dir in run_dirs:
            dataset_metrics[run_dir] = (
                pd.read_parquet(f"{base_dir}/{run_dir}/results_table.parquet"),
                pickle.load(open(f"{base_dir}/{run_dir}/node_pos_dict.pkl", "rb")),
                pickle.load(open(f"{base_dir}/{run_dir}/modality_contribution_results.pkl", "rb")),
                pickle.load(open(f"{base_dir}/{run_dir}/modality_centrality_results.pkl", "rb")),
                pickle.load(open(f"{base_dir}/{run_dir}/local_clustering_coefficient_results.pkl", "rb")),
                pickle.load(open(f"{base_dir}/{run_dir}/graph_metrics_results.pkl", "rb")),
            )
        return dataset_metrics


    @st.cache_resource
    def process_metrics():
        jaccard_index = {run_dir: [] for run_dir in run_dirs}
        txt_difference = {run_dir: [] for run_dir in run_dirs}
        img_difference = {run_dir: [] for run_dir in run_dirs}
        densities = {run_dir: [] for run_dir in run_dirs}
        num_cross_modal_edges = {run_dir: [] for run_dir in run_dirs}
        num_cross_modal_edges_centrality = {run_dir: [] for run_dir in run_dirs}
        global_clustering_coefficient = {run_dir: [] for run_dir in run_dirs}
        metrics = {run_dir: [] for run_dir in run_dirs}

        num_layers = 41
        reduction = partial(np.mean, axis=0)

        for run_dir, (table, node_pos, contrib, centrality, clustering, graph_metrics) in dataset_metrics.items():
            for idx in tqdm(range(len(table)), disable=False):
                jaccard_index[run_dir].extend(graph_metrics["jaccard_index"][idx])
                txt_difference[run_dir].extend(graph_metrics["txt_difference"][idx])
                img_difference[run_dir].extend(graph_metrics["img_difference"][idx])
                densities[run_dir].extend(graph_metrics["density"][idx])
                num_cross_modal_edges[run_dir].extend(graph_metrics["cross_modal_edges"][idx])
                num_cross_modal_edges_centrality[run_dir].extend(graph_metrics["num_cross_modal_edges_centrality"][idx])
                global_clustering_coefficient[run_dir].extend(graph_metrics["global_clustering_coefficient"][idx])

                for (layer_num,
                     token_num), img_contrib, img_centrality, txt_centrality, local_clustering_coefficient in zip(
                    node_pos[idx], contrib["img_contrib"][idx], centrality["img_centrality"][idx],
                    centrality["txt_centrality"][idx], clustering["local_clustering_coefficient"][idx]):

                    starting_tokens = layer_num == 0
                    num_img = (starting_tokens & (img_contrib == 1)).sum()
                    num_txt = starting_tokens.sum() - num_img

                    if len(img_contrib) == 0:
                        metrics[run_dir].append(np.zeros((num_layers, 4)))
                        continue # Skip empty layers

                    node_props = np.stack((img_contrib, img_centrality / num_img, txt_centrality / num_txt,
                                           local_clustering_coefficient), axis=1)
                    grouped = npi.group_by(keys=layer_num, values=node_props, reduction=reduction)
                    if any((grouped[i][0] > i for i in range(num_layers))):
                        raise ValueError("Grouped layers are not in order, I hoped this wouldn't happen")
                    metrics[run_dir].append([grouped[i][1] for i in range(num_layers)])

        metrics = {run_dir: np.array(m) for run_dir, m in metrics.items()}
        img_contribs = {run_dir: np.array(m[:, :, 0]) for run_dir, m in metrics.items()}
        img_centralities = {run_dir: np.array(m[:, :, 1]) for run_dir, m in metrics.items()}
        txt_centralities = {run_dir: np.array(m[:, :, 2]) for run_dir, m in metrics.items()}
        local_clustering_coefficients = {run_dir: np.array(m[:, :, 3]) for run_dir, m in metrics.items()}

        return (jaccard_index, txt_difference, img_difference, densities, num_cross_modal_edges,
                num_cross_modal_edges_centrality, global_clustering_coefficient, img_contribs, img_centralities,
                txt_centralities, local_clustering_coefficients)


    processor = load_processor("llava-hf/llava-1.5-13b-hf")

    run_dirs = {
        "WhatsUp_A/llava_base_2024_09_14-22_49_27": "WhatsUp A",
        "WhatsUp_A/llava_gen_2024_09_22-23_12_59": "WhatsUp A Gen",
        "WhatsUp_B/llava_base_2024_09_14-22_49_28": "WhatsUp B",
        "WhatsUp_B/llava_gen_2024_09_22-23_12_59": "WhatsUp B Gen",
        # "SEED-Bench-2_Part_1/llava_base_2024_09_14-22_52_11": "SEED Scene Understanding",
        # "SEED-Bench-2_Part_2/llava_base_2024_09_14-22_52_11": "SEED Instance Identity",
        "SEED-Bench-2_Part_3/llava_base_2024_09_14-22_52_12": "SEED Instance Attributes",
        # "SEED-Bench-2_Part_4/llava_base_2024_09_14-22_52_18": "SEED Instance Location",
        # "SEED-Bench-2_Part_5/llava_base_2024_09_14-22_52_17": "SEED Instance Count",
        "SEED-Bench-2_Part_6/llava_base_2024_09_14-22_52_14": "SEED Spatial Relation",
        "SEED-Bench-2_Part_7/llava_base_2024_09_14-22_52_15": "SEED Instance Interaction",
        # "SEED-Bench-2_Part_8/llava_base_2024_09_14-22_52_15": "SEED Visual Reasoning",
        # "SEED-Bench-2_Part_9/llava_base_2024_09_14-22_52_17": "SEED Text Understanding",
        # "SEED-Bench-2_Part_10/llava_base_2024_09_14-22_52_17": "SEED Celebrity Recognition",
        # "SEED-Bench-2_Part_11/llava_base_2024_09_14-22_52_14": "SEED Landmark Recognition",
        # "SEED-Bench-2_Part_12/llava_base_2024_09_14-22_52_14": "SEED Chart Understanding",
        # "SEED-Bench-2_Part_13/llava_base_2024_09_14-22_52_14": "SEED Visual Referring Expression",
        # "SEED-Bench-2_Part_14/llava_base_2024_09_14-22_52_14": "SEED Science Knowledge",
        # "SEED-Bench-2_Part_15/llava_base_2024_09_14-22_52_15": "SEED Emotion Recognition",
        # "SEED-Bench-2_Part_16/llava_base_2024_09_14-22_52_21": "SEED Visual Mathematics",
        "Unlabeled_COCO/llava_2000_2024_09_15-01_57_15": "COCO Captions",
        # "Unlabeled_COCO/llava_2000_reverse_2024_09_14-22_41_57": "COCO Captions Reverse",
    }

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
        "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    ]

    base_dir = "/home/projects/shimon/agroskin/projects/vlmflow/results"




    @st.cache_resource
    def unpickle():
        return pickle.load(open("processed_metrics.pkl", "rb"))


    def aggregate_metric_by_layers(metric, node_layers_dict):
        layers = []
        for layer in range(len(node_layers_dict)):
            layers.append(metric[node_layers_dict[layer]].mean().item())
        return layers


    print(load_from_disk)

    if not load_from_disk:
        dataset_metrics = load_data()
        processed_metrics = process_metrics()
        pickle.dump(processed_metrics, open("processed_metrics.pkl", "wb"))
    else:
        processed_metrics = unpickle()
        for d in run_dirs:
            if d not in processed_metrics[0]:
                dataset_metrics = load_data()
                processed_metrics = process_metrics()
                pickle.dump(processed_metrics, open("processed_metrics.pkl", "wb"))


    (jaccard_index, txt_difference, img_difference, densities, num_cross_modal_edges, num_cross_modal_edges_centrality,
     global_clustering_coefficient, img_contribs, img_centralities, txt_centralities,
     local_clustering_coefficients) = processed_metrics

    st.header("Graph Metrics")

    dir_names = [run_dirs[run_dir] for run_dir in run_dirs]
    # Plot a stacked bar char of jaccard_index, txt_Difference, img_difference for each run_dir
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Bar(x=dir_names, y=[np.nanmean(txt_difference[run_dir]) for run_dir in run_dirs], name="Text Difference"))
    fig.add_trace(
        go.Bar(x=dir_names, y=[2 * np.nanmean(jaccard_index[run_dir]) for run_dir in run_dirs], name="Jaccard Index"))
    fig.add_trace(
        go.Bar(x=dir_names, y=[np.nanmean(img_difference[run_dir]) for run_dir in run_dirs], name="Image Difference"))
    fig.update_layout(barmode="stack")
    st.plotly_chart(fig)

    st.header("Node Metrics")

    # Plot histograms for each run_dir for densities, cross_modal_edges, num_cross_modal_edges_centrality, global_clustering_coefficient
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Density", "Num Cross Modal Edges", "Num Cross Modal Edges Centrality",
                                        "Global Clustering Coefficient"])
    for run_dir, color in zip(list(run_dirs.keys()), colors):
        hist_args = {"histnorm": "probability density", "legendgroup": run_dirs[run_dir], "name": run_dirs[run_dir],
                     "hovertemplate": "%{x}", "marker": {"color": color}}
        fig.add_trace(go.Histogram(x=densities[run_dir], **hist_args), row=1, col=1)
        hist_args["showlegend"] = False
        fig.add_trace(go.Histogram(x=num_cross_modal_edges[run_dir], **hist_args), row=1, col=2)
        fig.add_trace(go.Histogram(x=num_cross_modal_edges_centrality[run_dir], **hist_args), row=2, col=1)
        fig.add_trace(go.Histogram(x=global_clustering_coefficient[run_dir], **hist_args), row=2, col=2)

    fig.update_xaxes(range=[0, 0.4], row=1, col=1)
    fig.update_layout(height=1000, width=1300, barmode="overlay")
    fig.update_traces(opacity=0.5)
    st.plotly_chart(fig)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Image Contribution", "Image Centrality", "Text Centrality",
                                        "Local Clustering Coefficient"])

    for run_dir, color in zip(list(run_dirs.keys()), colors):
        hist_args = {"histnorm": "probability density", "legendgroup": run_dirs[run_dir], "name": run_dirs[run_dir],
                     "hovertemplate": "%{x}", "marker": {"color": color}, "xbins": {"start": 0.01}}

        if "COCO" in run_dir:
            img_contribs_data = np.random.choice(img_contribs[run_dir].flatten(), size=10000, replace=False)
            img_centralities_data = np.random.choice(img_centralities[run_dir].flatten(), size=10000, replace=False)
            txt_centralities_data = np.random.choice(txt_centralities[run_dir].flatten(), size=10000, replace=False)
            local_clustering_coefficients_data = np.random.choice(local_clustering_coefficients[run_dir].flatten(),
                                                                  size=10000, replace=False)
        else:
            img_contribs_data = img_contribs[run_dir].flatten()
            img_centralities_data = img_centralities[run_dir].flatten()
            txt_centralities_data = txt_centralities[run_dir].flatten()
            local_clustering_coefficients_data = local_clustering_coefficients[run_dir].flatten()

        fig.add_trace(go.Histogram(x=img_contribs_data, **hist_args), row=1, col=1)
        hist_args["showlegend"] = False
        fig.add_trace(go.Histogram(x=img_centralities_data, **hist_args), row=1, col=2)
        fig.add_trace(go.Histogram(x=txt_centralities_data, **hist_args), row=2, col=1)
        fig.add_trace(go.Histogram(x=local_clustering_coefficients_data, **hist_args), row=2, col=2)

    fig.update_layout(height=1000, width=1300, barmode="overlay")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_traces(opacity=0.5)


    st.plotly_chart(fig)

    # st.header("Node Metrics by Layer")
    # tabs = st.tabs([f"{layer}" for layer in range(0, 41)])
    #
    # for layer, tab in tqdm(zip(range(0, 41), tabs), total=41):
    #     fig = make_subplots(rows=2, cols=2,
    #                         subplot_titles=["Image Contribution", "Image Centrality", "Text Centrality",
    #                                         "Local Clustering Coefficient"])
    #
    #     for run_dir, color in zip(list(run_dirs.keys()), colors):
    #         hist_args = {"histnorm": "probability density", "legendgroup": run_dirs[run_dir], "name": run_dirs[run_dir],
    #                      "hovertemplate": "%{x}", "marker": {"color": color}}
    #         fig.add_trace(go.Histogram(x=img_contribs[run_dir][:, layer], **hist_args), row=1, col=1)
    #         hist_args["showlegend"] = False
    #         fig.add_trace(go.Histogram(x=img_centralities[run_dir][:, layer], **hist_args, ), row=1, col=2)
    #         fig.add_trace(go.Histogram(x=txt_centralities[run_dir][:, layer], **hist_args), row=2, col=1)
    #         fig.add_trace(go.Histogram(x=local_clustering_coefficients[run_dir][:, layer], **hist_args), row=2, col=2)
    #
    #         fig.update_layout(height=1000, width=1300, barmode="overlay")
    #         fig.update_traces(opacity=0.5)
    #
    #     tab.plotly_chart(fig)
