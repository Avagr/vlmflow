import pickle

from graph_tool import Graph  # noqa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui import contribution_graph

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
    return (
        pd.read_parquet(f"{path}/results_table.parquet"),
        pickle.load(open(f"{path}/full_graph_dict.pkl", "rb")),
        # pickle.load(open(f"{path}/simple_graph_dict.pkl", "rb")),
        pickle.load(open(f"{path}/node_layers_dict.pkl", "rb")),
        pickle.load(open(f"{path}/modality_ratio_results.pkl", "rb")),
    )


@st.cache_resource
def load_processor(model_id):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)


@st.cache_data
def tokens_to_strings(token_ids):
    res = []
    for tok in token_ids:
        if tok == 32000:
            res.extend([f"I_{img_count}" for img_count in range(576)])
        else:
            str_repr = processor.tokenizer.decode(tok)
            match str_repr:
                case '':
                    str_repr = "_"
                case '\n':
                    str_repr = "\\n"
            res.append(str_repr)
    return res


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

base_dir = "/home/projects/shimon/agroskin/projects/vlmflow/results"
run_dir = st.text_input("Run Directory", value="Unlabeled_COCO/llava_tensors_300_2024_09_01-21_04_28")
table, full_graphs, node_layers, modality_ratio_results = read_results(f"{base_dir}/{run_dir}")

with st.sidebar:
    table_index = st.number_input("Table index", value=0, min_value=0, max_value=len(table) - 1)
    table_row = table.iloc[table_index]
    st.write(table_row.generated_caption)
    st.image(f"/home/projects/bagon/shared/coco/unlabeled2017/{table_row.image}")

graph_container = st.container()

tokens = tokens_to_strings(table_row.generated_ids)

with graph_container:
    contribution_graph(
        40,
        tokens,
        [[] for _ in range(len(tokens) - 1 - len(full_graphs[table_index]))] + [get_edge_list(graph) for graph in
                                                                                full_graphs[table_index]],
        key=f"graph_{run_dir}_{table_index}",
    )


def plot_for_item(generated_tokens, modality_res):
    z_mean, z_len = [], []
    for i in range(len(modality_res)):
        z_mean.append([])
        z_len.append([])
        for k, v in modality_res[i].items():
            z_mean[i].append(np.mean(v))
            z_len[i].append(len(v))

    z_mean = np.array(z_mean).transpose(1, 0)
    z_len = np.array(z_len).transpose(1, 0)

    stringified_token = processor.tokenizer.convert_ids_to_tokens(generated_tokens)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Image Contribution", "Number of Nodes"),
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    fig.add_trace(go.Surface(z=z_mean, surfacecolor=z_mean, colorscale='Viridis'), row=1, col=1)
    fig.add_trace(go.Surface(z=z_len, surfacecolor=z_len, colorscale='Viridis'), row=1, col=2)
    xaxis_config = {'title': 'Token', 'tickvals': list(range(len(stringified_token))), 'ticktext': stringified_token,
                    'tickfont': {'size': 14}}

    fig.update_layout(
        scene={'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
               'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}},
        scene2={'yaxis': {'title': 'Layer'}, 'xaxis': xaxis_config, 'zaxis': {'title': 'Value'},
                'aspectratio': {'x': 1.5, 'y': 1, 'z': 1}},
        width=1800,
        height=1000,
    )
    return fig


tokenized_prompt = processor.tokenizer(table_row.prompt)["input_ids"]
st.plotly_chart(plot_for_item(table_row.generated_ids[len(tokenized_prompt):], modality_ratio_results[table_index]))
