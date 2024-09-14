from PIL import Image
from graph_tool import Graph, PropertyMap  # noqa
import graph_tool.all as gt
from graph_tool.util import find_vertex
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flow.analysis import ClusteringCoefficient
from ui import contribution_graph
from utils.misc import plot_image_with_heatmap
from utils.streamlit import *


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

model_name = "llava-hf/llava-1.5-13b-hf"
processor = load_processor(model_name)

image_dir = "/home/projects/bagon/shared/coco/unlabeled2017"
base_dir = "/home/projects/shimon/agroskin/projects/vlmflow/results"
# run_dir = st.text_input("Run Directory", value="Unlabeled_COCO/llava_test_2024_09_01-20_44_23")
run_dir = st.text_input("Run Directory", value="Unlabeled_COCO/llava_tensors_300_2024_09_04-22_18_05")
table, full_graphs, simple_graphs, node_layers, modality_ratio_results, centrality_results = read_results(
    f"{base_dir}/{run_dir}")

(txt_centrality, img_centrality, total_centrality, intersection_centrality, txt_difference, img_difference,
 jaccard_similarity) = process_centrality(centrality_results)

with st.sidebar:
    table_index = st.number_input("Table index", value=0, min_value=0, max_value=len(table) - 1)
    table_row = table.iloc[table_index]
    st.write(table_row.generated_caption)
    # st.image(f"{image_dir}/{table_row.image}")

tokens, img_beg, img_end = tokens_to_strings(table_row.generated_ids, processor)

if 'current_token' not in st.session_state:
    st.session_state.current_token = len(tokens) - 1

num_before_graphs = len(tokens) - 1 - len(full_graphs[table_index])

clustering_fun = ClusteringCoefficient()
local_clustering_coeffs = []
global_clustering_coeffs = []
for g in simple_graphs[table_index]:
    l, g = clustering_fun(g, node_layers[table_index])
    local_clustering_coeffs.append(np.nan_to_num(l))
    global_clustering_coeffs.append(g)

match st.selectbox("Visualize metric",
                   ["Nothing", "Modality Ratio", "Local Clustering", "Text Centrality", "Image Centrality",
                    "Centrality Sum",
                    "Centrality Intersection", "Text Centrality Difference", "Image Centrality Difference",
                    "SBM Clustering", "Nested SBM Clustering"], index=2):
    case "Modality Ratio":
        node_style_map = create_node_style_map(simple_graphs[table_index],
                                               [g.vp.img_contrib.a for g in simple_graphs[table_index]], "value")
    case "Text Centrality":
        node_style_map = create_node_style_map(simple_graphs[table_index], txt_centrality[table_index], "value")
    case "Image Centrality":
        node_style_map = create_node_style_map(simple_graphs[table_index], img_centrality[table_index], "value")
    case "Local Clustering":
        node_style_map = create_node_style_map(simple_graphs[table_index], local_clustering_coeffs, "value")
    case "Centrality Sum":
        node_style_map = create_node_style_map(simple_graphs[table_index], total_centrality[table_index], "value")
    case "Centrality Intersection":
        node_style_map = create_node_style_map(simple_graphs[table_index], intersection_centrality[table_index],
                                               "value")
    case "Text Centrality Difference":
        node_style_map = create_node_style_map(simple_graphs[table_index], txt_difference[table_index], "value")
    case "Image Centrality Difference":
        node_style_map = create_node_style_map(simple_graphs[table_index], img_difference[table_index], "value")
    case "SBM Clustering":
        clustering = min_block(simple_graphs[table_index])
        node_style_map = create_node_style_map(simple_graphs[table_index], clustering, "cluster")
    case "Nested SBM Clustering":
        clustering = min_block(simple_graphs[table_index], nested=True)
        i = st.number_input("Nested level", value=0, min_value=0, max_value=3)
        clustering = [c.project_level(i).get_blocks() for c in clustering]
        node_style_map = create_node_style_map(simple_graphs[table_index], clustering, "cluster")
    case _:
        node_style_map = None

graph_output = contribution_graph(
    40,
    tokens,
    [[] for _ in range(num_before_graphs)] + [get_edge_list(graph) for graph in full_graphs[table_index]],
    key=f"graph_{run_dir}_{table_index}",
    node_style_map=node_style_map
)

# print(res, len(tokens), num_before_graphs, len(full_graphs[table_index]))

if graph_output is not None:
    st.session_state.current_token = graph_output

heatmap = np.zeros((24, 24))
graph_idx = st.session_state.current_token - num_before_graphs - 1
if graph_idx >= 0:
    graph = simple_graphs[table_index][graph_idx]
    starting_nodes = find_vertex(graph, graph.vp.layer_num, 0)
    for node in starting_nodes:
        node_pos = graph.vp.token_num[node]
        if img_beg <= node_pos < img_end:
            node_pos -= img_beg
            heatmap[node_pos // 24, node_pos % 24] = 1

with st.sidebar:
    fig, ax = plt.subplots()
    heatmap_overlay = plot_image_with_heatmap(Image.open(f"{image_dir}/{table_row.image}"), heatmap, ax=ax)
    st.pyplot(fig)

tokenized_prompt = processor.tokenizer(table_row.prompt)["input_ids"]
stringified_token = processor.tokenizer.convert_ids_to_tokens(table_row.generated_ids[len(tokenized_prompt):])

contrib_mean, contrib_len = plot_for_item(modality_ratio_results[table_index], node_layers[table_index])
txt_centrality_mean, _ = plot_for_item(txt_centrality[table_index], node_layers[table_index])
img_centrality_mean, _ = plot_for_item(img_centrality[table_index], node_layers[table_index])
clustering_coeffs_mean, _ = plot_for_item(local_clustering_coeffs, node_layers[table_index])

fig = make_subplots(rows=3, cols=2,
                    subplot_titles=("Image Contribution", "Number of Nodes", "Text Centrality", "Image Centrality",
                                    "Local Clustering Coefficient", ""),
                    specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}],
                           [{'type': 'surface'}, {'type': 'surface'}]],
                    vertical_spacing=0.05
                    )
fig.add_trace(go.Surface(z=contrib_mean, surfacecolor=contrib_mean, showscale=False), row=1, col=1)
fig.add_trace(go.Surface(z=contrib_len, surfacecolor=contrib_len, showscale=False), row=1, col=2)
fig.add_trace(go.Surface(z=txt_centrality_mean, surfacecolor=txt_centrality_mean, showscale=False), row=2, col=1)
fig.add_trace(go.Surface(z=img_centrality_mean, surfacecolor=img_centrality_mean, showscale=False), row=2,col=2)
fig.add_trace(go.Surface(z=clustering_coeffs_mean, surfacecolor=clustering_coeffs_mean, showscale=False), row=3, col=1)

xaxis_config = {'title': 'Token', 'tickvals': list(range(len(stringified_token))), 'ticktext': stringified_token,
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

st.plotly_chart(fig)

js = jaccard_similarity[table_index]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Jaccard Similarity", "Global Clustering Coefficient"))

fig.add_trace(go.Scatter(x=list(range(len(js))), y=js, mode='lines+markers', name='Jaccard Similarity'), row=1, col=1)
fig.add_trace(go.Scatter(x=list(range(len(global_clustering_coeffs))), y=global_clustering_coeffs, mode='lines+markers', name='Global Clustering Coefficient'), row=1, col=2)




fig.update_layout(
    width=1800,
    title='Jaccard Similarity and Global Clustering Coefficient',
    xaxis1=dict(title='Sample Index', tickvals=list(range(len(stringified_token))), ticktext=stringified_token, tickangle=-60),
    xaxis2=dict(title='Sample Index', tickvals=list(range(len(stringified_token))), ticktext=stringified_token, tickangle=-60),
    yaxis=dict(title='Value')
)

st.plotly_chart(fig)
