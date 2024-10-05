import json

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


    run_dirs = [
        "Unlabeled_COCO/molmo_3000_2024_09_29-04_08_54",
        "Unlabeled_COCO/molmo_72b_merged_600",
        "Unlabeled_COCO/llava_3000_2024_09_26-02_13_31",
        "Unlabeled_COCO/llava_3000_copy",
        "WhatsUp_A/llava_base_2024_09_14-22_49_27",
        "WhatsUp_A/llava_gen_2024_09_22-23_12_59",
        "WhatsUp_B/llava_base_2024_09_14-22_49_28",
        "WhatsUp_B/llava_gen_2024_09_22-23_12_59",
        "SEED-Bench-2_Part_8/llava_base_2024_09_14-22_52_15",
        "SEED-Bench-2_Part_16/llava_base_2024_09_14-22_52_21",
        "SEED-Bench-2_Part_4/llava_base_2024_09_14-22_52_18",
        "SEED-Bench-2_Part_15/llava_base_2024_09_14-22_52_15",
        "SEED-Bench-2_Part_2/llava_base_2024_09_14-22_52_11",
        "SEED-Bench-2_Part_14/llava_base_2024_09_14-22_52_14",
        "SEED-Bench-2_Part_13/llava_base_2024_09_14-22_52_14",
        "SEED-Bench-2_Part_10/llava_base_2024_09_14-22_52_17",
        "SEED-Bench-2_Part_5/llava_base_2024_09_14-22_52_17",
        "SEED-Bench-2_Part_11/llava_base_2024_09_14-22_52_14",
        "SEED-Bench-2_Part_6/llava_base_2024_09_14-22_52_14",
        "SEED-Bench-2_Part_12/llava_base_2024_09_14-22_52_14",
        "SEED-Bench-2_Part_7/llava_base_2024_09_14-22_52_15",
        "SEED-Bench-2_Part_9/llava_base_2024_09_14-22_52_17",
        "SEED-Bench-2_Part_1/llava_base_2024_09_14-22_52_11",
        "SEED-Bench-2_Part_3/llava_base_2024_09_14-22_52_12",  
        "Unlabeled_COCO/llava_2000_reverse_2024_09_14-22_41_57",

    ]

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
    run_dir = st.selectbox("Run Directory", run_dirs, index=0, on_change=st.session_state.clear)

    if 'llava' in run_dir:
        model_name = 'llava'
        processor = load_processor("llava-hf/llava-1.5-13b-hf")
    elif 'molmo' in run_dir:
        model_name = 'molmo'
        processor = load_processor("allenai/Molmo-72B-0924")
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

    simple_graphs, full_graphs = read_graphs(f"{base_dir}/{run_dir}", table_index, len(node_layers[table_index]))

    print("Read graphs")

    img_beg, img_end = table_row.img_begin, table_row.img_end

    tokens = tokens_to_strings(table_row.generated_ids, model_name, processor)

    if 'current_token' not in st.session_state:
        st.session_state.current_token = len(tokens) - 1

    num_before_graphs = len(tokens) - 1 - len(full_graphs)

    local_clustering_coeffs = [np.nan_to_num(g.vp.local_clustering.a) for g in simple_graphs]
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
                            "Image Centrality Difference", "SBM Clustering", "Nested SBM Clustering"], index=1):
            case "Modality Ratio":
                node_style_map = create_node_style_map(simple_graphs,
                                                       [g.vp.img_contrib.a for g in simple_graphs], "value")
            case "Text Centrality":
                node_style_map = create_node_style_map(simple_graphs, txt_centrality[table_index], "value")
            case "Image Centrality":
                node_style_map = create_node_style_map(simple_graphs, img_centrality[table_index], "value")
            case "Local Clustering":
                node_style_map = create_node_style_map(simple_graphs, local_clustering_coeffs, "value")
            case "Centrality Sum":
                node_style_map = create_node_style_map(simple_graphs, total_centrality[table_index], "value")
            case "Centrality Intersection":
                node_style_map = create_node_style_map(simple_graphs, intersection_centrality[table_index], "value")
            case "Text Centrality Difference":
                node_style_map = create_node_style_map(simple_graphs, txt_difference[table_index], "value")
            case "Image Centrality Difference":
                node_style_map = create_node_style_map(simple_graphs, img_difference[table_index], "value")
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
    clustering_coeffs_mean, _ = plot_for_item(local_clustering_coeffs, node_layers[table_index])

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
        fig.add_trace(go.Surface(z=clustering_coeffs_mean, surfacecolor=clustering_coeffs_mean, showscale=False), row=3,
                      col=1)

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
        fig.add_trace(
            go.Scatter(y=clustering_coeffs_mean.squeeze(), mode='lines+markers', name='Local Clustering Coefficient'),
            row=3, col=1)

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
