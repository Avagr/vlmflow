import pickle

from PIL import Image
from graph_tool import Graph, PropertyMap  # noqa
from graph_tool.util import find_vertex
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui import contribution_graph
from utils.misc import plot_image_with_heatmap

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


@st.cache_resource
def read_results(path: str):
    # for key in centrality.keys():
    #     for sample in centrality[key]:
    #         for i in range(len(sample)):
    #             sample[i] = sample[i] / sample[i].max()

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

    for txt_sample, img_sample in zip(centrality['txt_centrality'], centrality['img_centrality']):
        txt_centrality.append([])
        img_centrality.append([])
        total_centrality.append([])
        intersection_centrality.append([])
        for txt_tok, img_tok in zip(txt_sample, img_sample):
            txt_centrality[-1].append(txt_tok / txt_tok.max())
            img_centrality[-1].append(img_tok / img_tok.max())
            total_centrality[-1].append((txt_tok + img_tok))
            total_centrality[-1][-1] = total_centrality[-1][-1] / total_centrality[-1][-1].max()
            intersection_centrality[-1].append(np.where(txt_tok > img_tok, img_tok, txt_tok))
            intersection_centrality[-1][-1] = intersection_centrality[-1][-1] / intersection_centrality[-1][-1].max()

    return txt_centrality, img_centrality, total_centrality, intersection_centrality


@st.cache_resource
def load_processor(model_id):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)


@st.cache_data
def tokens_to_strings(token_ids):
    res = []
    img_beg, img_end = None, None
    for i, tok in enumerate(token_ids):
        if tok == 32000:
            res.extend([f"I_{img_count}" for img_count in range(576)])
            img_beg = i
            img_end = i + 576
        else:
            str_repr = processor.tokenizer.decode(tok)
            match str_repr:
                case '':
                    str_repr = "_"
                case '\n':
                    str_repr = "\\n"
            res.append(str_repr)
    return res, img_beg, img_end


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


model_name = "llava-hf/llava-1.5-13b-hf"
processor = load_processor(model_name)

image_dir = "/home/projects/bagon/shared/coco/unlabeled2017"
base_dir = "/home/projects/shimon/agroskin/projects/vlmflow/results"
run_dir = st.text_input("Run Directory", value="Unlabeled_COCO/llava_test_2024_09_01-20_44_23")
# run_dir = st.text_input("Run Directory", value="Unlabeled_COCO/llava_tensors_300_2024_09_01-21_04_28")
table, full_graphs, simple_graphs, node_layers, modality_ratio_results, centrality_results = read_results(
    f"{base_dir}/{run_dir}")

txt_centrality, img_centrality, total_centrality, intersection_centrality = process_centrality(centrality_results)

with st.sidebar:
    table_index = st.number_input("Table index", value=0, min_value=0, max_value=len(table) - 1)
    table_row = table.iloc[table_index]
    st.write(table_row.generated_caption)
    # st.image(f"{image_dir}/{table_row.image}")

tokens, img_beg, img_end = tokens_to_strings(table_row.generated_ids)

if 'current_token' not in st.session_state:
    st.session_state.current_token = len(tokens) - 1

num_before_graphs = len(tokens) - 1 - len(full_graphs[table_index])


def create_node_style_map(graphs: list[Graph], metrics: list[PropertyMap]):
    style_map: list[list] = []
    for simple_graph, metric in zip(graphs, metrics):
        style_map.append([])
        for v, layer_num, token_num in simple_graph.get_vertices(
                vprops=[simple_graph.vp.layer_num, simple_graph.vp.token_num]):
            if layer_num > 0:
                metric_value = metric[v]
                color = f"rgb({int(255 * (1 - metric_value))}, {int(255 * (1 - metric_value))}, 255)"
                style_map[-1].append([f"{layer_num - 1}_{token_num}", [color, metric_value]])
    return style_map


match st.selectbox("Visualize metric",
                   ["Nothing", "Modality Ratio", "Text Centrality", "Image Centrality", "Centrality Sum",
                    "Centrality Intersection"], index=1):
    case "Modality Ratio":
        node_style_map = create_node_style_map(simple_graphs[table_index],
                                               [g.vp.img_contrib.a for g in simple_graphs[table_index]])
    case "Text Centrality":
        node_style_map = create_node_style_map(simple_graphs[table_index], txt_centrality[table_index])
    case "Image Centrality":
        node_style_map = create_node_style_map(simple_graphs[table_index], img_centrality[table_index])
    case "Centrality Sum":
        node_style_map = create_node_style_map(simple_graphs[table_index], total_centrality[table_index])
    case "Centrality Intersection":
        node_style_map = create_node_style_map(simple_graphs[table_index], intersection_centrality[table_index])
    case _:
        node_style_map = None

res = contribution_graph(
    40,
    tokens,
    [[] for _ in range(num_before_graphs)] + [get_edge_list(graph) for graph in full_graphs[table_index]],
    key=f"graph_{run_dir}_{table_index}",
    node_style_map=node_style_map
)

# print(res, len(tokens), num_before_graphs, len(full_graphs[table_index]))

if res is not None:
    st.session_state.current_token = res

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


tokenized_prompt = processor.tokenizer(table_row.prompt)["input_ids"]
stringified_token = processor.tokenizer.convert_ids_to_tokens(table_row.generated_ids[len(tokenized_prompt):])

contrib_mean, contrib_len = plot_for_item(modality_ratio_results[table_index], node_layers[table_index])
txt_centrality_mean, _ = plot_for_item(centrality_results['txt_centrality'][table_index], node_layers[table_index])
img_centrality_mean, _ = plot_for_item(centrality_results['img_centrality'][table_index], node_layers[table_index])

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Image Contribution", "Number of Nodes", "Text Centrality", "Image Centrality"),
                    specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}]]
                    )
fig.add_trace(go.Surface(z=contrib_mean, surfacecolor=contrib_mean, colorscale='Viridis'), row=1, col=1)
fig.add_trace(go.Surface(z=contrib_len, surfacecolor=contrib_len, colorscale='Viridis'), row=1, col=2)
fig.add_trace(go.Surface(z=txt_centrality_mean, surfacecolor=txt_centrality_mean, colorscale='Viridis'), row=2, col=1)
fig.add_trace(go.Surface(z=img_centrality_mean, surfacecolor=img_centrality_mean, colorscale='Viridis'), row=2, col=2)
xaxis_config = {'title': 'Token', 'tickvals': list(range(len(stringified_token))), 'ticktext': stringified_token,
                'tickfont': {'size': 14}}

fig.update_layout(
    scene={'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
           'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}},
    scene2={'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
            'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}},
    scene3={'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
            'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}},
    scene4={'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
            'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}},
    width=1800,
    height=1800,
)

st.plotly_chart(fig)
