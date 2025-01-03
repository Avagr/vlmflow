import json

from PIL import Image
from graph_tool import Graph, PropertyMap  # noqa
import graph_tool.all as gt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui import contribution_graph
from utils.streamlit import *

if __name__ == "__main__":

    @st.cache_resource(hash_funcs={list[Graph]: id, Graph: id})
    def min_block(graphs: list[Graph], nested: bool = False):
        if nested:
            return [gt.minimize_nested_blockmodel_dl(g) for g in graphs]
        return [
            gt.minimize_blockmodel_dl(g, state=gt.ModularityState).get_blocks()
            for g in graphs]


    def reset_state():
        st.session_state.clear()


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
        # "MMVP/llava_abcd_2024_10_16-02_23_57": "LLaVA 13B MMVP",

        # "Unlabeled_COCO/llava_2000_2024_09_15-01_57_15": "COCO Captions",
        # "WhatsUp_A/llava_gen_2024_09_22-23_12_59": "WhatsUp A Gen",
        # "Unlabeled_COCO/llava_2000_reverse_2024_09_14-22_41_57": "COCO Captions Reverse",
        # "WhatsUp_B/llava_gen_2024_09_22-23_12_59": "WhatsUp B Gen",

        # MOLMO 7B

        "Unlabeled_COCO/molmo_3000_2024_09_29-04_08_54": "Molmo 7B COCO Captions",
        # "Unlabeled_COCO/molmo_uninit_2024_12_18-00_14_19": "Molmo 7B Uninitialized",
        "WhatsUp_A/molmo_grouped_2024_12_19-18_23_55": "Molmo 7B WhatsUp A Grouped",
        "WhatsUp_A/molmo_fixed_abcd_2024_10_10-17_44_42": "Molmo 7B WhatsUp A",
        "WhatsUp_B/molmo_fixed_abcd_2024_10_10-17_44_42": "Molmo 7B WhatsUp B",
        # "SEED-Bench-2_Part_1/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Scene Understanding",
        # "SEED-Bench-2_Part_2/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Instance Identity",
        # "SEED-Bench-2_Part_3/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Instance Attributes",
        # "SEED-Bench-2_Part_4/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Instance Location",
        # "SEED-Bench-2_Part_5/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Instance Count",
        # "SEED-Bench-2_Part_6/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Spatial Relation",
        # "SEED-Bench-2_Part_7/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Instance Interaction",
        # "SEED-Bench-2_Part_8/molmo_fixed_abcd_2024_10_10-17_46_01": "Molmo 7B SEED Visual Reasoning",
        # "SEED-Bench-2_Part_9/molmo_fixed_abcd_2024_10_10-17_46_02": "Molmo 7B SEED Text Understanding",
        # "SEED-Bench-2_Part_10/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Celebrity Recognition",
        # "SEED-Bench-2_Part_11/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Landmark Recognition",
        # "SEED-Bench-2_Part_12/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Chart Understanding",
        # "SEED-Bench-2_Part_13/molmo_fixed_abcd_2024_10_10-17_45_57": "Molmo 7B SEED Visual Referring Expression",
        # "SEED-Bench-2_Part_14/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Science Knowledge",
        # "SEED-Bench-2_Part_15/molmo_fixed_abcd_2024_10_10-17_45_59": "Molmo 7B SEED Emotion Recognition",
        # "SEED-Bench-2_Part_16/molmo_fixed_abcd_2024_10_10-17_45_58": "Molmo 7B SEED Visual Mathematics",
        # "MMVP/molmo_abcd_2024_10_16-02_20_52": "Molmo 7B MMVP",

        # MOLMO 72B

        # "Unlabeled_COCO/molmo_72b_merged_600": "Molmo 72B",
        # "WhatsUp_A/molmo_72b_abcd_2024_10_12-15_27_10": "Molmo 72B WhatsUp A",
        # "WhatsUp_B/molmo_72b_abcd_2024_10_12-02_38_50": "Molmo 72B WhatsUp B",
        # "SEED-Bench-2_Part_1/molmo_72b_abcd_merged": "Molmo 72B SEED Scene Understanding",
        # "SEED-Bench-2_Part_2/molmo_72b_abcd_merged": "Molmo 72B SEED Instance Identity",
        # "SEED-Bench-2_Part_3/molmo_72b_abcd_merged": "Molmo 72B SEED Instance Attributes",
        # "SEED-Bench-2_Part_4/molmo_72b_abcd_merged": "Molmo 72B SEED Instance Location",
        # "SEED-Bench-2_Part_5/molmo_72b_abcd_merged": "Molmo 72B SEED Instance Count",
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
        # "MMVP/molmo_72b_abcd_2024_10_16-02_20_19": "Molmo 72B MMVP",

        # Pixtral

        # "Unlabeled_COCO/pixtral_3000_merged": "Pixtral COCO Captions",
        # "WhatsUp_A/pixtral_abcd_2024_10_13-18_14_52": "Pixtral WhatsUp A",
        # "WhatsUp_B/pixtral_abcd_2024_10_13-18_14_52": "Pixtral WhatsUp B",
        # "SEED-Bench-2_Part_1/pixtral_abcd_merged": "Pixtral SEED Scene Understanding",
        # "SEED-Bench-2_Part_2/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Instance Identity",
        # "SEED-Bench-2_Part_3/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Instance Attributes",
        # "SEED-Bench-2_Part_4/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Instance Location",
        # "SEED-Bench-2_Part_5/pixtral_abcd_2024_10_13-21_02_17": "Pixtral SEED Instance Count",
        # "SEED-Bench-2_Part_6/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Spatial Relation",
        # "SEED-Bench-2_Part_7/pixtral_abcd_2024_10_13-18_19_36": "Pixtral SEED Instance Interaction",
        # "SEED-Bench-2_Part_8/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Visual Reasoning",
        # "SEED-Bench-2_Part_9/pixtral_abcd_2024_10_13-21_02_05": "Pixtral SEED Text Understanding",
        # "SEED-Bench-2_Part_10/pixtral_abcd_2024_10_13-21_02_04": "Pixtral SEED Celebrity Recognition",
        # "SEED-Bench-2_Part_11/pixtral_abcd_2024_10_13-21_02_17": "Pixtral SEED Landmark Recognition",
        # "SEED-Bench-2_Part_12/pixtral_abcd_2024_10_13-21_02_06": "Pixtral SEED Chart Understanding",
        # "SEED-Bench-2_Part_13/pixtral_abcd_2024_10_13-18_18_43": "Pixtral SEED Visual Referring Expression",
        # "SEED-Bench-2_Part_14/pixtral_abcd_2024_10_13-18_18_43": "Pixtral SEED Science Knowledge",
        # "SEED-Bench-2_Part_15/pixtral_abcd_2024_10_13-18_18_43": "Pixtral SEED Emotion Recognition",
        # "SEED-Bench-2_Part_16/pixtral_abcd_2024_10_13-21_02_04": "Pixtral SEED Visual Mathematics",
        # "MMVP/pixtral_abcd_2024_10_16-02_24_14": "Pixtral MMVP",
    }

    margins_css = """
        <style>
            .main > div {
                padding: 1rem;
                padding-top: 3rem;  # Still need this gap for the top bar
                gap: 0rem;
            }
    
            section[data-testid="stSidebar"] {
                width: 300px !important; # Set the width to your desired value
            }
        </style>
    """
    st.set_page_config(layout="wide")
    st.markdown(margins_css, unsafe_allow_html=True)

    base_dir = "/home/projects/shimon/agroskin/projects/vlmflow/results"
    run_dir = st.selectbox("Run Directory", list(dirs_to_name.keys()), index=0, on_change=st.session_state.clear,
                           format_func=lambda x: dirs_to_name[x])

    if 'llava' in run_dir:
        model_name = 'llava'
        processor = load_processor("llava-hf/llava-1.5-13b-hf")
    elif 'molmo' in run_dir:
        model_name = 'molmo'
        processor = load_processor("allenai/Molmo-72B-0924")
    elif "pixtral" in run_dir:
        model_name = 'pixtral'
        processor = load_processor("mistral-community/pixtral-12b")
    else:
        raise ValueError(f"Unsupported model made '{run_dir}'")
    print("Loaded processor")

    table, node_layers, centrality_results, graph_metrics = read_metric_results(f"{base_dir}/{run_dir}")
    print("Loaded metrics")
    image_dir = get_image_dir(run_dir)

    (txt_centrality, img_centrality, total_centrality, intersection_centrality, txt_difference, img_difference,
     jaccard_similarity) = process_centrality(centrality_results)
    print("Processed centrality")

    with st.sidebar:
        table_index = st.number_input("Table index", value=0, min_value=0, max_value=len(table) - 1)
        table_row = table.iloc[table_index]
        st.write(processor.tokenizer.decode(table_row.generated_ids, skip_special_tokens=True))
        if 'answer' in table_row:
            st.write("\nCORRECT:", table_row.answer)
        # Show the image
        if "home" not in table_row.image:
            img_path = f"{image_dir}/{table_row.image}"
        else:
            img_path = table_row.image
        st.image(Image.open(img_path), caption=table_row.image, use_container_width=True)


    simple_graphs, full_graphs = read_graphs(f"{base_dir}/{run_dir}", table_index, len(node_layers[table_index]))

    print("Read graphs")

    img_beg, img_end = table_row.img_begin, table_row.img_end

    tokens = tokens_to_strings(table_row.generated_ids, model_name, processor)

    if 'current_token' not in st.session_state:
        st.session_state.current_token = len(tokens) - 1

    num_before_graphs = len(tokens) - 1 - len(full_graphs)

    # local_clustering_coeffs = [np.nan_to_num(g.vp.local_clustering.a) for g in simple_graphs]
    # print(local_clustering_coeffs)

    if st.checkbox("Show Overgraph", value=False):
        overgraph = gt.load_graph(f"{base_dir}/{run_dir}/overgraphs/overgraph_{table_index}.gt")
        vertex_weights = json.load(open(f"{base_dir}/{run_dir}/overgraphs/vertex_weights_{table_index}.json", 'r'))
        node_style_map = []
        for v, weight in vertex_weights.items():
            layer_num, token_num = map(int, v.split('_'))
            if layer_num > 0:
                value = weight / len(simple_graphs)
                color = f"rgb({int(255 * (1 - value))}, {int(255 * (1 - value))}, 255)"
                node_style_map.append([f"{layer_num - 1}_{token_num}", [color, value]])
        graph_output = contribution_graph(
            simple_graphs[0].gp.num_layers,
            tokens,
            [[] for _ in range(len(tokens) - 2)] + [get_edge_list(overgraph)],
            key=f"overgraph_{run_dir}_{table_index}",
            node_style_map=[node_style_map]
        )

    else:
        match st.selectbox("Visualize metric",
                           ["Nothing", "Modality Ratio", "Local Clustering", "Text Centrality", "Image Centrality",
                            "Centrality Sum", "Centrality Intersection", "Text Centrality Difference",
                            "Image Centrality Difference", "SBM Clustering", "K-core decomposition", "Nested SBM Clustering"], index=1):
            case "Modality Ratio":
                node_style_map = create_node_style_map(simple_graphs,
                                                       [g.vp.img_contrib.a for g in simple_graphs], "value")
            case "Text Centrality":
                node_style_map = create_node_style_map(simple_graphs, txt_centrality[table_index], "value")
            case "Image Centrality":
                node_style_map = create_node_style_map(simple_graphs, img_centrality[table_index], "value")
            # case "Local Clustering":
            #     node_style_map = create_node_style_map(simple_graphs, local_clustering_coeffs, "value")
            case "Centrality Sum":
                node_style_map = create_node_style_map(simple_graphs, total_centrality[table_index], "value")
            case "Centrality Intersection":
                node_style_map = create_node_style_map(simple_graphs, intersection_centrality[table_index], "value")
            case "Text Centrality Difference":
                node_style_map = create_node_style_map(simple_graphs, txt_difference[table_index], "value")
            case "Image Centrality Difference":
                node_style_map = create_node_style_map(simple_graphs, img_difference[table_index], "value")
            case "K-core decomposition":
                # print(simple_graphs)
                kcore = get_kcore(simple_graphs)
                node_style_map = create_node_style_map(simple_graphs, kcore, "value")
            case "SBM Clustering":
                clustering = min_block(simple_graphs)
                node_style_map = create_node_style_map(simple_graphs, clustering, "cluster")
            case "Nested SBM Clustering":
                clustering = min_block(simple_graphs, nested=True)
                i = st.number_input("Nested level", value=0, min_value=0, max_value=3)
                clustering = [c.project_level(i).get_blocks() for c in clustering]
                node_style_map = create_node_style_map(simple_graphs, clustering, "cluster")
            case _:
                node_style_map = None

        with st.container():
            graph_output = contribution_graph(
                simple_graphs[0].gp.num_layers,
                tokens,
                [[] for _ in range(num_before_graphs)] + [get_edge_list(graph) for graph in full_graphs],
                key=f"graph_{run_dir}_{table_index}",
                node_style_map=node_style_map
            )

        # print(res, len(tokens), num_before_graphs, len(full_graphs))

        if graph_output is not None:
            st.session_state.current_token = graph_output

    # heatmap = np.zeros((24, 24))
    # graph_idx = st.session_state.current_token - num_before_graphs
    # print(graph_idx)
    # if graph_idx >= 0:
    #     graph = simple_graphs[graph_idx]
    #     starting_nodes = find_vertex(graph, graph.vp.layer_num, 0)
    #     for node in starting_nodes:
    #         node_pos = graph.vp.token_num[node]
    #         if img_beg <= node_pos < img_end:
    #             node_pos -= img_beg
    #             heatmap[node_pos // 24, node_pos % 24] = 1
    #
    # print(heatmap)

    # with st.sidebar:
    #     fig, ax = plt.subplots()
    #     img_path = f"{image_dir}/{table_row.image}"
    #     if "seedbench" in image_dir and not table_row.image.endswith(".png"):
    #         img_path = f"{image_dir}/cc3m-images/{table_row.image}"
    #     heatmap_overlay = plot_image_with_heatmap(Image.open(img_path), heatmap, ax=ax)
    #     st.pyplot(fig)

    stringified_tokens = tokens[-table_row.num_generated_tokens:]

    contrib_mean, contrib_len = plot_for_item([g.vp.img_contrib.a for g in simple_graphs], node_layers[table_index])
    txt_centrality_mean, _ = plot_for_item(txt_centrality[table_index], node_layers[table_index])
    img_centrality_mean, _ = plot_for_item(img_centrality[table_index], node_layers[table_index])
    # clustering_coeffs_mean, _ = plot_for_item(local_clustering_coeffs, node_layers[table_index])

    if len(simple_graphs) > 1:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Image Contribution", "Number of Nodes", "Text Centrality", "Image Centrality",
                            "Local Clustering Coefficient", ""),
            specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            vertical_spacing=0.05
        )
        fig.add_trace(go.Surface(z=contrib_mean, surfacecolor=contrib_mean, showscale=False), row=1, col=1)
        fig.add_trace(go.Surface(z=contrib_len, surfacecolor=contrib_len, showscale=False), row=1, col=2)
        fig.add_trace(go.Surface(z=txt_centrality_mean, surfacecolor=txt_centrality_mean, showscale=False), row=2,
                      col=1)
        fig.add_trace(go.Surface(z=img_centrality_mean, surfacecolor=img_centrality_mean, showscale=False), row=2,
                      col=2)
        # fig.add_trace(go.Surface(z=clustering_coeffs_mean, surfacecolor=clustering_coeffs_mean, showscale=False), row=3,
        #               col=1)

        xaxis_config = {'title': 'Token', 'tickvals': list(range(len(stringified_tokens))),
                        'ticktext': stringified_tokens,
                        'tickfont': {'size': 14}}

        scene_layout = {'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
                        'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}, 'camera': {'eye': {'x': 1.75, 'y': 1.75, 'z': 1.75}}}

        fig.update_layout(
            scene=scene_layout,
            scene2=scene_layout,
            scene3=scene_layout,
            scene4=scene_layout,
            scene5=scene_layout,
            width=1800,
            height=2500,
        )
    else:
        fig = make_subplots(
            rows=3, cols=2,
            vertical_spacing=0.05
        )

        # Plot for the 2d case instead of 3d
        fig.add_trace(go.Scatter(y=contrib_mean.squeeze(), mode='lines+markers', name='Image Contribution'), row=1,
                      col=1)
        fig.add_trace(go.Scatter(y=contrib_len.squeeze(), mode='lines+markers', name='Number of Nodes'), row=1, col=2)
        fig.add_trace(go.Scatter(y=txt_centrality_mean.squeeze(), mode='lines+markers', name='Text Centrality'), row=2,
                      col=1)
        fig.add_trace(go.Scatter(y=img_centrality_mean.squeeze(), mode='lines+markers', name='Image Centrality'), row=2,
                      col=2)
        # fig.add_trace(
        #     go.Scatter(y=clustering_coeffs_mean.squeeze(), mode='lines+markers', name='Local Clustering Coefficient'),
        #     row=3, col=1)

        fig.update_layout(
            width=1800,
            height=1800,
        )

    st.plotly_chart(fig)

    js = jaccard_similarity[table_index]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Jaccard Similarity", "Global Clustering Coefficient"))

    fig.add_trace(go.Scatter(x=list(range(len(js))), y=js, mode='lines+markers', name='Jaccard Similarity'), row=1,
                  col=1)
    # fig.add_trace(go.Scatter(x=list(range(len(global_clustering_coeffs))), y=global_clustering_coeffs, mode='lines+markers',
    #                          name='Global Clustering Coefficient'), row=1, col=2)

    fig.update_layout(
        width=1800,
        title='Jaccard Similarity and Global Clustering Coefficient',
        xaxis1=dict(title='Sample Index', tickvals=list(range(len(stringified_tokens))), ticktext=stringified_tokens,
                    tickangle=-60),
        xaxis2=dict(title='Sample Index', tickvals=list(range(len(stringified_tokens))), ticktext=stringified_tokens,
                    tickangle=-60),
        yaxis=dict(title='Value')
    )

    st.plotly_chart(fig)
