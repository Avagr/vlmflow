from pathlib import Path

from PIL import Image
from graph_tool import Graph, PropertyMap  # noqa
import graph_tool.all as gt
import torch

from flow.graphs import build_graph_from_contributions
from ui import contribution_graph
from utils.dirs_to_name import dirs_to_name
from utils.streamlit import *

if __name__ == "__main__":

    # import streamlit.watcher
    #
    # streamlit.watcher._POLLING_PERIOD_SECS = 0.01  # Probably very bad but anything slower and it takes ages to update

    @st.cache_resource(hash_funcs={list[Graph]: id, Graph: id})
    def min_block(graphs: list[Graph], nested: bool = False):
        if nested:
            return [gt.minimize_nested_blockmodel_dl(g) for g in graphs]
        return [
            gt.minimize_blockmodel_dl(g, state=gt.ModularityState).get_blocks()
            for g in graphs]


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

    table, node_layers, node_pos, centrality_results, graph_metrics = read_metric_results(f"{base_dir}/{run_dir}")
    image_dir = get_image_dir(run_dir)

    with st.sidebar:
        table_index = st.number_input("Table index", value=0, min_value=0, max_value=len(table) - 1)
        table_row = table.iloc[table_index]
        st.write(processor.tokenizer.decode(table_row.generated_ids, skip_special_tokens=True))
        if 'answer' in table_row:
            st.write(("\nCORRECT\n" if table_row.match else "\nWRONG\n") + "\nCorrect answer:", table_row.answer)
        # Show the image
        if "home" not in table_row.image:
            img_path = f"{image_dir}/{table_row.image}"
        else:
            img_path = table_row.image
        st.image(Image.open(img_path), caption=table_row.image,
                 # use_container_width=True,
                 width=300,
                 )

    simple_graphs, full_graphs = read_graphs(f"{base_dir}/{run_dir}", table_index, len(node_layers[table_index]))
    (txt_centrality, img_centrality, total_centrality, intersection_centrality, txt_difference,
     img_difference) = process_centrality(simple_graphs, adaptive_normalization=True)

    # --- Custom Thresholding ---

    # @st.cache_resource()
    # def load_unprocessed_graph(row_idx):
    #     return torch.load(f"{base_dir}/{run_dir}/graphs/{row_idx}_graph_tensors.pkl", weights_only=True)
    #
    # @st.cache_resource()
    # def threshold_graph(t, row_idx, img_begin, img_end):
    #     img_beg, img_end = table_row.img_begin, table_row.img_end
    #     attn, attn_res, ffn, ffn_res = load_unprocessed_graph(row_idx)
    #     return build_graph_from_contributions(
    #         attn=attn, attn_res=attn_res, ffn=ffn, ffn_res=ffn_res,
    #         img_begin=img_begin, img_end=img_end, threshold=t, top_token_num=-1
    #     )
    #
    #
    # selected_threshold = st.slider(min_value=0.005, max_value=0.040, value=0.01, step=0.001, label="Threshold", format="%.3f")
    # fg, sg, _ = threshold_graph(selected_threshold, table_row.idx, table_row.img_begin, table_row.img_end)
    # simple_graphs = [sg]
    # full_graphs = [fg]

    # ----------------------

    tokens = tokens_to_strings(table_row.generated_ids, model_name, processor)

    if 'current_token' not in st.session_state:
        st.session_state.current_token = len(tokens) - 1

    num_before_graphs = len(tokens) - 1 - len(full_graphs)

    match st.selectbox("Visualize metric",
                       ["Nothing", "Test", "Modality Ratio", "Text Centrality", "Image Centrality",
                        "Closeness Centrality", "Token Closeness Centrality", "Closeness",
                        "Centrality Sum", "Centrality Intersection", "Text Centrality Difference",
                        "Image Centrality Difference", "SBM Clustering", "K-core decomposition",
                        "Nested SBM Clustering"], index=2):
        case "Test":
            katz = [gt.katz(graph, alpha=0.9, norm=True).a for graph in simple_graphs]
            node_style_map = create_node_style_map(simple_graphs, katz, "value")
        case "Modality Ratio":
            node_style_map = create_node_style_map(simple_graphs,
                                                   [g.vp.img_contrib.a for g in simple_graphs], "value")
        case "Closeness Centrality":
            closeness_centrality = get_closeness_centrality(simple_graphs)
            node_style_map = create_node_style_map(simple_graphs, closeness_centrality, "value")

        case "Token Closeness Centrality":
            distance_closeness_centrality = get_token_closeness_centrality(simple_graphs)
            node_style_map = create_node_style_map(simple_graphs, distance_closeness_centrality, "value")

        case "Closeness":
            closeness = get_closeness(simple_graphs)
            node_style_map = create_node_style_map(simple_graphs, closeness, "value")

        case "Local Clustering":
            local_clustering = [np.nan_to_num(g.vp.local_clustering.a) for g in simple_graphs]
            node_style_map = create_node_style_map(simple_graphs, local_clustering, "value")
        case "Text Centrality":
            node_style_map = create_node_style_map(simple_graphs, txt_centrality, "value")
        case "Image Centrality":
            node_style_map = create_node_style_map(simple_graphs, img_centrality, "value")
        # case "Local Clustering":
        #     node_style_map = create_node_style_map(simple_graphs, local_clustering_coeffs, "value")
        case "Centrality Sum":
            node_style_map = create_node_style_map(simple_graphs, total_centrality, "value")
        case "Centrality Intersection":
            node_style_map = create_node_style_map(simple_graphs, intersection_centrality, "value")
        case "Text Centrality Difference":
            node_style_map = create_node_style_map(simple_graphs, txt_difference, "value")
        case "Image Centrality Difference":
            node_style_map = create_node_style_map(simple_graphs, img_difference, "value")
        case "K-core decomposition":
            # print(simple_graphs)
            kcore = [gt.kcore_decomposition(g) for g in simple_graphs]
            node_style_map = create_node_style_map(simple_graphs, kcore, "cluster")
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
            node_style_map=node_style_map,
        )

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

    # stringified_tokens = tokens[-table_row.num_generated_tokens:]
    #
    # contrib_mean, contrib_len = plot_for_item([g.vp.img_contrib.a for g in simple_graphs], node_layers[table_index])
    # txt_centrality_mean, _ = plot_for_item(txt_centrality, node_layers[table_index])
    # img_centrality_mean, _ = plot_for_item(img_centrality, node_layers[table_index])
    # clustering_coeffs_mean, _ = plot_for_item(local_clustering_coeffs, node_layers[table_index])

    # if len(simple_graphs) > 1:
    #     fig = make_subplots(
    #         rows=3, cols=2,
    #         subplot_titles=("Image Contribution", "Number of Nodes", "Text Centrality", "Image Centrality",
    #                         "Local Clustering Coefficient", ""),
    #         specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}],
    #                [{'type': 'surface'}, {'type': 'surface'}]],
    #         vertical_spacing=0.05
    #     )
    #     fig.add_trace(go.Surface(z=contrib_mean, surfacecolor=contrib_mean, showscale=False), row=1, col=1)
    #     fig.add_trace(go.Surface(z=contrib_len, surfacecolor=contrib_len, showscale=False), row=1, col=2)
    #     fig.add_trace(go.Surface(z=txt_centrality_mean, surfacecolor=txt_centrality_mean, showscale=False), row=2,
    #                   col=1)
    #     fig.add_trace(go.Surface(z=img_centrality_mean, surfacecolor=img_centrality_mean, showscale=False), row=2,
    #                   col=2)
    #     # fig.add_trace(go.Surface(z=clustering_coeffs_mean, surfacecolor=clustering_coeffs_mean, showscale=False), row=3,
    #     #               col=1)
    #
    #     xaxis_config = {'title': 'Token', 'tickvals': list(range(len(stringified_tokens))),
    #                     'ticktext': stringified_tokens,
    #                     'tickfont': {'size': 14}}
    #
    #     scene_layout = {'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
    #                     'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}, 'camera': {'eye': {'x': 1.75, 'y': 1.75, 'z': 1.75}}}
    #
    #     fig.update_layout(
    #         scene=scene_layout,
    #         scene2=scene_layout,
    #         scene3=scene_layout,
    #         scene4=scene_layout,
    #         scene5=scene_layout,
    #         width=1800,
    #         height=2500,
    #     )
    # else:
    #     fig = make_subplots(
    #         rows=3, cols=2,
    #         vertical_spacing=0.05
    #     )
    #
    #     # Plot for the 2d case instead of 3d
    #     fig.add_trace(go.Scatter(y=contrib_mean.squeeze(), mode='lines+markers', name='Image Contribution'), row=1,
    #                   col=1)
    #     fig.add_trace(go.Scatter(y=contrib_len.squeeze(), mode='lines+markers', name='Number of Nodes'), row=1, col=2)
    #     fig.add_trace(go.Scatter(y=txt_centrality_mean.squeeze(), mode='lines+markers', name='Text Centrality'), row=2,
    #                   col=1)
    #     fig.add_trace(go.Scatter(y=img_centrality_mean.squeeze(), mode='lines+markers', name='Image Centrality'), row=2,
    #                   col=2)
    #     # fig.add_trace(
    #     #     go.Scatter(y=clustering_coeffs_mean.squeeze(), mode='lines+markers', name='Local Clustering Coefficient'),
    #     #     row=3, col=1)
    #
    #     fig.update_layout(
    #         width=1800,
    #         height=1800,
    #     )
    #
    # st.plotly_chart(fig)

    # js = jaccard_similarity[table_index]
    #
    # fig = make_subplots(rows=1, cols=2, subplot_titles=("Jaccard Similarity", "Global Clustering Coefficient"))
    #
    # fig.add_trace(go.Scatter(x=list(range(len(js))), y=js, mode='lines+markers', name='Jaccard Similarity'), row=1,
    #               col=1)
    # fig.add_trace(go.Scatter(x=list(range(len(global_clustering_coeffs))), y=global_clustering_coeffs, mode='lines+markers',
    #                          name='Global Clustering Coefficient'), row=1, col=2)

    # fig.update_layout(
    #     width=1800,
    #     title='Jaccard Similarity and Global Clustering Coefficient',
    #     xaxis1=dict(title='Sample Index', tickvals=list(range(len(stringified_tokens))), ticktext=stringified_tokens,
    #                 tickangle=-60),
    #     xaxis2=dict(title='Sample Index', tickvals=list(range(len(stringified_tokens))), ticktext=stringified_tokens,
    #                 tickangle=-60),
    #     yaxis=dict(title='Value')
    # )
    #
    # st.plotly_chart(fig)
