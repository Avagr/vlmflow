from functools import partial
import pickle

from graph_tool import Graph, PropertyMap  # noqa
import graph_tool.all as gt  # noqa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from tqdm.auto import tqdm

from utils.streamlit import load_processor

if __name__ == "__main__":

    st.set_page_config(layout="wide")
    load_from_disk = st.checkbox("Load from disk", value=False)


    @st.cache_resource
    def load_data():
        dataset_metrics = {}
        for run_dir in dirs_to_name:
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
        jaccard_index = {run_dir: [] for run_dir in dirs_to_name}
        txt_difference = {run_dir: [] for run_dir in dirs_to_name}
        img_difference = {run_dir: [] for run_dir in dirs_to_name}
        node_densities = {run_dir: [] for run_dir in dirs_to_name}
        edge_densities = {run_dir: [] for run_dir in dirs_to_name}
        num_cross_modal_edges = {run_dir: [] for run_dir in dirs_to_name}
        # num_cross_modal_edges_centrality = {run_dir: [] for run_dir in dirs_to_name}
        # global_clustering_coefficient = {run_dir: [] for run_dir in dirs_to_name}
        # metrics = {run_dir: [] for run_dir in dirs_to_name}

        reduction = partial(np.mean, axis=0)

        for run_dir, (table, node_pos, contrib, centrality, clustering, graph_metrics) in tqdm(dataset_metrics.items()):
            if "llava" in run_dir:
                num_layers = 41
            elif "molmo_72b" in run_dir:
                num_layers = 81
            elif "molmo" in run_dir:
                num_layers = 29
            elif "pixtral" in run_dir:
                num_layers = 41
            else:
                raise ValueError("Unknown model")

            for idx in tqdm(range(len(table)), disable=False):
                jaccard_index[run_dir].extend(graph_metrics["jaccard_index"][idx])
                txt_difference[run_dir].extend(graph_metrics["txt_difference"][idx])
                img_difference[run_dir].extend(graph_metrics["img_difference"][idx])
                # node_densities[run_dir].extend(graph_metrics["node_density"][idx])
                # edge_densities[run_dir].extend(graph_metrics["edge_density"][idx])
                # num_cross_modal_edges[run_dir].extend(graph_metrics["cross_modal_edges"][idx])
                # num_cross_modal_edges_centrality[run_dir].extend(graph_metrics["cross_modal_edges"][idx])
                # global_clustering_coefficient[run_dir].extend(graph_metrics["global_clustering_coefficient"][idx])

                # for (layer_num,
                #      token_num), img_contrib, img_centrality, txt_centrality, local_clustering_coefficient in zip(
                #     node_pos[idx], contrib["img_contrib"][idx], centrality["img_centrality"][idx],
                #     centrality["txt_centrality"][idx], clustering["local_clustering_coefficient"][idx]):
                #
                #     starting_tokens = layer_num == 0
                #     num_img = (starting_tokens & (img_contrib == 1)).sum()
                #     num_txt = starting_tokens.sum() - num_img
                #
                #     if len(img_contrib) == 0:
                #         metrics[run_dir].append(np.zeros((num_layers, 4)))
                #         continue # Skip empty layers
                #
                #     node_props = np.stack((img_contrib, img_centrality / (num_img if num_img > 0 else 1),
                #                            txt_centrality / (num_txt if num_txt > 0 else 1),
                #                            local_clustering_coefficient), axis=1)
                #     grouped = npi.group_by(keys=layer_num, values=node_props, reduction=reduction)
                #     if any((grouped[i][0] > i for i in range(num_layers))):
                #         raise ValueError("Grouped layers are not in order, I hoped this wouldn't happen")
                #     metrics[run_dir].append([grouped[i][1] for i in range(num_layers)])

        # metrics = {run_dir: np.array(m) for run_dir, m in metrics.items()}
        # img_contribs = {run_dir: np.array(m[:, :, 0]) for run_dir, m in metrics.items()}
        # img_centralities = {run_dir: np.array(m[:, :, 1]) for run_dir, m in metrics.items()}
        # txt_centralities = {run_dir: np.array(m[:, :, 2]) for run_dir, m in metrics.items()}
        # local_clustering_coefficients = {run_dir: np.array(m[:, :, 3]) for run_dir, m in metrics.items()}

        return (
            jaccard_index,
            txt_difference,
            img_difference,
            # node_densities,
            # edge_densities,
            # num_cross_modal_edges,
            #  global_clustering_coefficient,
            # img_contribs, img_centralities,
            # txt_centralities,
            # local_clustering_coefficients
        )


    processor = load_processor("llava-hf/llava-1.5-13b-hf")

    dirs_to_name = {

        # LLaVA 13B

        # "Unlabeled_COCO/llava_3000_2024_09_26-02_13_31": "LLaVA 13B COCO Captions",
        # "WhatsUp_A/llava_base_2024_09_14-22_49_27": "LLaVA 13B WhatsUp A",
        # "WhatsUp_B/llava_base_2024_09_14-22_49_28": "LLaVA 13B WhatsUp B",
        # "SEED-Bench-2_Part_1/llava_base_2024_09_14-22_52_11": "LLaVA 13B SEED Scene Understanding",
        # "SEED-Bench-2_Part_2/llava_base_2024_09_14-22_52_11": "LLaVA 13B SEED Instance Identity",
        # "SEED-Bench-2_Part_3/llava_base_2024_09_14-22_52_12": "LLaVA 13B SEED Instance Attributes",
        # "SEED-Bench-2_Part_4/llava_base_2024_09_14-22_52_18": "LLaVA 13B SEED Instance Location",
        # "SEED-Bench-2_Part_5/llava_base_2024_09_14-22_52_17": "LLaVA 13B SEED Instance Count",
        # "SEED-Bench-2_Part_6/llava_base_2024_09_14-22_52_14": "LLaVA 13B SEED Spatial Relation",
        # "SEED-Bench-2_Part_7/llava_base_2024_09_14-22_52_15": "LLaVA 13B SEED Instance Interaction",
        # "SEED-Bench-2_Part_8/llava_base_2024_09_14-22_52_15": "LLaVA 13B SEED Visual Reasoning",
        # "SEED-Bench-2_Part_9/llava_base_2024_09_14-22_52_17": "LLaVA 13B SEED Text Understanding",
        # "SEED-Bench-2_Part_10/llava_base_2024_09_14-22_52_17": "LLaVA 13B SEED Celebrity Recognition",
        # "SEED-Bench-2_Part_11/llava_base_2024_09_14-22_52_14": "LLaVA 13B SEED Landmark Recognition",
        # "SEED-Bench-2_Part_12/llava_base_2024_09_14-22_52_14": "LLaVA 13B EED Chart Understanding",
        # "SEED-Bench-2_Part_13/llava_base_2024_09_14-22_52_14": "LLaVA 13B SEED Visual Referring Expression",
        # "SEED-Bench-2_Part_14/llava_base_2024_09_14-22_52_14": "LLaVA 13B SEED Science Knowledge",
        # "SEED-Bench-2_Part_15/llava_base_2024_09_14-22_52_15": "LLaVA 13B SEED Emotion Recognition",
        # "SEED-Bench-2_Part_16/llava_base_2024_09_14-22_52_21": "LLaVA 13B SEED Visual Mathematics",
        # "Unlabeled_COCO/llava_2000_2024_09_15-01_57_15": "COCO Captions",
        # "WhatsUp_A/llava_gen_2024_09_22-23_12_59": "WhatsUp A Gen",
        # "Unlabeled_COCO/llava_2000_reverse_2024_09_14-22_41_57": "COCO Captions Reverse",
        # "WhatsUp_B/llava_gen_2024_09_22-23_12_59": "WhatsUp B Gen",

        # MOLMO 7B
        "Unlabeled_COCO/molmo_uninit_2024_12_18-00_14_19": "Molmo 7B Uninitialized",
        "Unlabeled_COCO/molmo_3000_2024_09_29-04_08_54": "Molmo 7B COCO Captions",
        "WhatsUp_A/molmo_fixed_abcd_2024_10_10-17_44_42": "Molmo 7B WhatsUp A",
        "WhatsUp_B/molmo_fixed_abcd_2024_10_10-17_44_42": "Molmo 7B WhatsUp B",
        "SEED-Bench-2_Part_1/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Scene Understanding",
        "SEED-Bench-2_Part_2/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Instance Identity",
        "SEED-Bench-2_Part_3/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Instance Attributes",
        "SEED-Bench-2_Part_4/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Instance Location",
        "SEED-Bench-2_Part_5/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Instance Count",
        "SEED-Bench-2_Part_6/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Spatial Relation",
        "SEED-Bench-2_Part_7/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Instance Interaction",
        "SEED-Bench-2_Part_8/molmo_fixed_abcd_2024_10_10-17_46_01": "Molmo 7B SEED Visual Reasoning",
        "SEED-Bench-2_Part_9/molmo_fixed_abcd_2024_10_10-17_46_02": "Molmo 7B SEED Text Understanding",
        "SEED-Bench-2_Part_10/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Celebrity Recognition",
        "SEED-Bench-2_Part_11/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Landmark Recognition",
        "SEED-Bench-2_Part_12/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Chart Understanding",
        "SEED-Bench-2_Part_13/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Visual Referring Expression",
        "SEED-Bench-2_Part_14/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Science Knowledge",
        "SEED-Bench-2_Part_15/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Emotion Recognition",
        "SEED-Bench-2_Part_16/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Visual Mathematics",

        # MOLMO 72B

        # "Unlabeled_COCO/molmo_72b_merged_600": "Molmo 72B",
        # "WhatsUp_A/molmo_72b_abcd_2024_10_12-15_27_10": "Molmo 72B WhatsUp A",
        # "WhatsUp_B/molmo_72b_abcd_2024_10_12-02_38_50": "Molmo 72B WhatsUp B",
        # "SEED-Bench-2_Part_1/": "Molmo 72B SEED Scene Understanding",
        # "SEED-Bench-2_Part_2/": "Molmo 72B SEED Instance Identity",
        # "SEED-Bench-2_Part_3/": "Molmo 72B SEED Instance Attributes",
        # "SEED-Bench-2_Part_4/": "Molmo 72B SEED Instance Location",
        # "SEED-Bench-2_Part_5/": "Molmo 72B SEED Instance Count",
        # "SEED-Bench-2_Part_6/molmo_72b_abcd_2024_10_12-09_42_51": "Molmo 72B SEED Spatial Relation",
        # "SEED-Bench-2_Part_7/molmo_72b_abcd_2024_10_12-15_04_10": "Molmo 72B SEED Instance Interaction",
        # "SEED-Bench-2_Part_8/molmo_72b_abcd_2024_10_14-16_46_49": "Molmo 72B SEED Visual Reasoning",
        # "SEED-Bench-2_Part_9/molmo_72b_abcd_2024_10_12-08_49_30": "Molmo 72B SEED Text Understanding",
        # "SEED-Bench-2_Part_10/molmo_72b_abcd_2024_10_12-06_05_53": "Molmo 72B SEED Celebrity Recognition",
        # "SEED-Bench-2_Part_11/molmo_72b_abcd_2024_10_12-05_11_16": "Molmo 72B SEED Landmark Recognition",
        # "SEED-Bench-2_Part_12/molmo_72b_abcd_2024_10_12-02_48_43": "Molmo 72B SEED Chart Understanding",
        # "SEED-Bench-2_Part_13/molmo_72b_abcd_2024_10_12-02_48_43": "Molmo 72B SEED Visual Referring Expression",
        # "SEED-Bench-2_Part_14/molmo_72b_abcd_2024_10_12-02_48_36": "Molmo 72B SEED Science Knowledge",
        # "SEED-Bench-2_Part_15/molmo_72b_abcd_2024_10_12-02_48_36": "Molmo 72B SEED Emotion Recognition",
        # "SEED-Bench-2_Part_16/molmo_72b_abcd_2024_10_12-02_48_34": "Molmo 72B SEED Visual Mathematics",

        # Pixtral

        "Unlabeled_COCO/pixtral_3000_merged": "Pixtral COCO Captions",
        "WhatsUp_A/pixtral_abcd_2024_10_13-18_14_52": "Pixtral WhatsUp A",
        "WhatsUp_B/pixtral_abcd_2024_10_13-18_14_52": "Pixtral WhatsUp B",
        "SEED-Bench-2_Part_1/pixtral_abcd_merged": "Pixtral SEED Scene Understanding",
        "SEED-Bench-2_Part_2/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Instance Identity",
        "SEED-Bench-2_Part_3/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Instance Attributes",
        "SEED-Bench-2_Part_4/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Instance Location",
        "SEED-Bench-2_Part_5/pixtral_abcd_2024_10_13-21_02_17": "Pixtral SEED Instance Count",
        "SEED-Bench-2_Part_6/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Spatial Relation",
        "SEED-Bench-2_Part_7/pixtral_abcd_2024_10_13-18_19_36": "Pixtral SEED Instance Interaction",
        "SEED-Bench-2_Part_8/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Visual Reasoning",
        "SEED-Bench-2_Part_9/pixtral_abcd_2024_10_13-21_02_05": "Pixtral SEED Text Understanding",
        "SEED-Bench-2_Part_10/pixtral_abcd_2024_10_13-21_02_04": "Pixtral SEED Celebrity Recognition",
        "SEED-Bench-2_Part_11/pixtral_abcd_2024_10_13-21_02_17": "Pixtral SEED Landmark Recognition",
        "SEED-Bench-2_Part_12/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Chart Understanding",
        "SEED-Bench-2_Part_13/pixtral_abcd_2024_10_13-18_18_43": "Pixtral SEED Visual Referring Expression",
        "SEED-Bench-2_Part_14/pixtral_abcd_2024_10_13-18_18_43": "Pixtral SEED Science Knowledge",
        "SEED-Bench-2_Part_15/pixtral_abcd_2024_10_13-18_18_43": "Pixtral SEED Emotion Recognition",
        "SEED-Bench-2_Part_16/pixtral_abcd_2024_10_13-21_02_04": "Pixtral SEED Visual Mathematics",
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
        for d in dirs_to_name:
            if d not in processed_metrics[0]:
                dataset_metrics = load_data()
                processed_metrics = process_metrics()
                pickle.dump(processed_metrics, open("processed_metrics.pkl", "wb"))

    (
        jaccard_index,
        txt_difference,
        img_difference,
        # node_densities,
        # edge_densities,
        # num_cross_modal_edges,
        # global_clustering_coefficient,
        # img_contribs,
        # img_centralities,
        # txt_centralities,
        # local_clustering_coefficients
    ) = processed_metrics

    st.header("Graph Metrics")

    llava_run_dirs = [run_dir for run_dir in dirs_to_name if dirs_to_name[run_dir].startswith("LLaVA 13B")]
    molmo_7b_run_dirs = [run_dir for run_dir in dirs_to_name if dirs_to_name[run_dir].startswith("Molmo 7B")]
    molmo_72b_run_dirs = [run_dir for run_dir in dirs_to_name if dirs_to_name[run_dir].startswith("Molmo 72B")]
    pixtral_run_dirs = [run_dir for run_dir in dirs_to_name if dirs_to_name[run_dir].startswith("Pixtral")]
    llava_dir_names = [dirs_to_name[run_dir] for run_dir in llava_run_dirs]
    molmo_7b_dir_names = [dirs_to_name[run_dir] for run_dir in molmo_7b_run_dirs]
    molmo_72b_dir_names = [dirs_to_name[run_dir] for run_dir in molmo_72b_run_dirs]
    pixtral_dir_names = [dirs_to_name[run_dir] for run_dir in pixtral_run_dirs]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Bar(x=llava_dir_names, y=[np.nanmean(txt_difference[run_dir]) for run_dir in llava_run_dirs],
               name="Text Difference"), row=1, col=1)
    fig.add_trace(
        go.Bar(x=llava_dir_names, y=[2 * np.nanmean(jaccard_index[run_dir]) for run_dir in llava_run_dirs],
               name="Jaccard Index"), row=1, col=1)
    fig.add_trace(
        go.Bar(x=llava_dir_names, y=[np.nanmean(img_difference[run_dir]) for run_dir in llava_run_dirs],
               name="Image Difference"), row=1, col=1)
    fig.add_trace(
        go.Bar(x=molmo_7b_dir_names, y=[np.nanmean(txt_difference[run_dir]) for run_dir in molmo_7b_run_dirs],
               name="Text Difference"), row=2, col=1)
    fig.add_trace(
        go.Bar(x=molmo_7b_dir_names, y=[2 * np.nanmean(jaccard_index[run_dir]) for run_dir in molmo_7b_run_dirs],
               name="Jaccard Index"), row=2, col=1)
    fig.add_trace(
        go.Bar(x=molmo_7b_dir_names, y=[np.nanmean(img_difference[run_dir]) for run_dir in molmo_7b_run_dirs],
               name="Image Difference"), row=2, col=1)
    # fig.add_trace(
    #     go.Bar(x=molmo_72b_dir_names, y=[np.nanmean(txt_difference[run_dir]) for run_dir in molmo_72b_run_dirs],
    #            name="Text Difference"), row=3, col=1)
    # fig.add_trace(
    #     go.Bar(x=molmo_72b_dir_names, y=[2 * np.nanmean(jaccard_index[run_dir]) for run_dir in molmo_72b_run_dirs],
    #            name="Jaccard Index"), row=3, col=1)
    # fig.add_trace(
    #     go.Bar(x=molmo_72b_dir_names, y=[np.nanmean(img_difference[run_dir]) for run_dir in molmo_72b_run_dirs],
    #            name="Image Difference"), row=3, col=1)
    fig.add_trace(
        go.Bar(x=pixtral_dir_names, y=[np.nanmean(txt_difference[run_dir]) for run_dir in pixtral_run_dirs],
                name="Text Difference"), row=4, col=1)
    fig.add_trace(
        go.Bar(x=pixtral_dir_names, y=[2 * np.nanmean(jaccard_index[run_dir]) for run_dir in pixtral_run_dirs],
                name="Jaccard Index"), row=4, col=1)
    fig.add_trace(
        go.Bar(x=pixtral_dir_names, y=[np.nanmean(img_difference[run_dir]) for run_dir in pixtral_run_dirs],
                name="Image Difference"), row=4, col=1)
    fig.update_layout(barmode="stack")
    st.plotly_chart(fig)

    # st.header("Node Metrics")
    #
    # # Plot histograms for each run_dir for densities, cross_modal_edges, num_cross_modal_edges_centrality, global_clustering_coefficient
    # fig = make_subplots(rows=2, cols=2,
    #                     subplot_titles=["Node Density", 'Edge Density', "Num Cross Modal Edges",
    #                                     "Global Clustering Coefficient"])
    # for run_dir, color in zip(list(dirs_to_name.keys()), colors):
    #     hist_args = {"histnorm": "probability density", "legendgroup": dirs_to_name[run_dir], "name": dirs_to_name[run_dir],
    #                  "hovertemplate": "%{x}", "marker": {"color": color}}
    #     fig.add_trace(go.Histogram(x=node_densities[run_dir], **hist_args), row=1, col=1)
    #     hist_args["showlegend"] = False
    #     fig.add_trace(go.Histogram(x=edge_densities[run_dir], **hist_args), row=1, col=2)
    #
    #     fig.add_trace(go.Histogram(x=num_cross_modal_edges[run_dir], **hist_args), row=2, col=1)
    #     fig.add_trace(go.Histogram(x=global_clustering_coefficient[run_dir], **hist_args), row=2, col=2)
    #
    # fig.update_xaxes(range=[0, 0.4], row=1, col=1)
    # fig.update_layout(height=1000, width=1300, barmode="overlay")
    # fig.update_traces(opacity=0.5)
    # st.plotly_chart(fig)
    #
    # fig = make_subplots(rows=2, cols=2,
    #                     subplot_titles=["Image Contribution", "Image Centrality", "Text Centrality",
    #                                     "Local Clustering Coefficient"])
    #
    # for run_dir, color in zip(list(dirs_to_name.keys()), colors):
    #     hist_args = {"histnorm": "probability density", "legendgroup": dirs_to_name[run_dir], "name": dirs_to_name[run_dir],
    #                  "hovertemplate": "%{x}", "marker": {"color": color}, "xbins": {"start": 0.01}}
    #
    #     if "COCO" in run_dir:
    #         img_contribs_data = np.random.choice(img_contribs[run_dir].flatten(), size=10000, replace=False)
    #         img_centralities_data = np.random.choice(img_centralities[run_dir].flatten(), size=10000, replace=False)
    #         txt_centralities_data = np.random.choice(txt_centralities[run_dir].flatten(), size=10000, replace=False)
    #         local_clustering_coefficients_data = np.random.choice(local_clustering_coefficients[run_dir].flatten(),
    #                                                               size=10000, replace=False)
    #     else:
    #         img_contribs_data = img_contribs[run_dir].flatten()
    #         img_centralities_data = img_centralities[run_dir].flatten()
    #         txt_centralities_data = txt_centralities[run_dir].flatten()
    #         local_clustering_coefficients_data = local_clustering_coefficients[run_dir].flatten()
    #
    #     fig.add_trace(go.Histogram(x=img_contribs_data, **hist_args), row=1, col=1)
    #     hist_args["showlegend"] = False
    #     fig.add_trace(go.Histogram(x=img_centralities_data, **hist_args), row=1, col=2)
    #     fig.add_trace(go.Histogram(x=txt_centralities_data, **hist_args), row=2, col=1)
    #     fig.add_trace(go.Histogram(x=local_clustering_coefficients_data, **hist_args), row=2, col=2)
    #
    # fig.update_layout(height=1000, width=1300, barmode="overlay")
    # fig.update_yaxes(type="log", row=1, col=1)
    # fig.update_yaxes(type="log", row=1, col=2)
    # fig.update_yaxes(type="log", row=2, col=1)
    # fig.update_traces(opacity=0.5)
    #
    #
    # st.plotly_chart(fig)

    # st.header("Node Metrics by Layer")
    # tabs = st.tabs([f"{layer}" for layer in range(0, 41)])
    #
    # for layer, tab in tqdm(zip(range(0, 41), tabs), total=41):
    #     fig = make_subplots(rows=2, cols=2,
    #                         subplot_titles=["Image Contribution", "Image Centrality", "Text Centrality",
    #                                         "Local Clustering Coefficient"])
    #
    #     for run_dir, color in zip(list(dirs_to_name.keys()), colors):
    #         hist_args = {"histnorm": "probability density", "legendgroup": dirs_to_name[run_dir], "name": dirs_to_name[run_dir],
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
