import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from matplotlib import pyplot as plt
from transformers import LlavaForConditionalGeneration, AutoProcessor

from flow.graph import build_full_graph, apply_graph_threshold
from models.transparent_models import TransparentLlava
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


@st.cache_resource
def load_model(model_name, _device="cuda", _dtype=torch.bfloat16):
    return TransparentLlava(
        name=model_name,
        llava=LlavaForConditionalGeneration.from_pretrained(model_name,
                                                            torch_dtype=torch.bfloat16,
                                                            device_map="auto"),
        processor=AutoProcessor.from_pretrained(model_name),
        device=_device,
        dtype=_dtype
    )


@st.cache_resource
def build_graph(_model: TransparentLlava, prompt, img_path, threshold):  # TODO: MODEL NOT CACHED
    image = Image.open(img_path)
    with st.sidebar.expander("Uploaded image", expanded=True):
        st.image(image)
    with (torch.inference_mode()):
        inputs = _model.processor(prompt, image, return_tensors="pt").to("cuda")
        results = _model(**inputs, output_attentions=True)
        pred = _model.processor.tokenizer.decode(torch.argmax(results.logits[0, -1]))
    with st.sidebar.expander("Model prediction", expanded=True):
        st.write(f"'{pred}'")
    return build_full_graph(_model, 0, threshold, True)


@st.cache_resource(hash_funcs={nx.DiGraph: id})
def threshold_graph(full_graph, n_layers, n_tokens, threshold):
    return apply_graph_threshold(full_graph, n_layers, n_tokens, [n_tokens - 1], threshold)


model_name = "llava-hf/llava-1.5-13b-hf"
st.set_page_config(layout="wide")
st.markdown(margins_css, unsafe_allow_html=True)
default_prompt = (f"USER: <image>\nWhich sentence best describes this image?\nA. A scarf to the right of a "
                  f"armchair\nB. A scarf on a armchair\nC. A scarf under a armchair\nD. A scarf to the left of a "
                  f"armchair\nASSISTANT: ")
default_path = "/home/projects/shimon/agroskin/datasets/whatsup/data/controlled_images/scarf_right_of_armchair.jpeg"

run_id = "llava_base_2024_07_28-03_20_47"
base_dir = "/home/projects/shimon/agroskin/projects/vlmflow/results/WhatsUp_B"

table = pd.read_pickle(f"{base_dir}/{run_id}/results_table.pkl")

table_index = st.number_input("Table index", value=0, min_value=0, max_value=len(table) - 1)

text = table.iloc[table_index]["prompt"]
image_path = f'/home/projects/shimon/agroskin/datasets/whatsup/{table.iloc[table_index]["image"]}'
st.write(text)

# text = st.text_area("Prompt", value=default_prompt, height=200)
# image_path = st.text_input("Image", value=default_path)

with st.sidebar.expander("Graph", expanded=True):
    norm_threshold = st.number_input("Renormalization Threshold", step=0.0001, min_value=0., max_value=0.5, format="%.4f", value=0.0001)
    sparse_threshold = st.number_input("Sparsification Threshold", step=0.0001, min_value=0., max_value=0.5, format="%.4f", value=0.025)
    renormalize_after_threshold = st.checkbox("Renormalize after threshold", value=False)

if st.button("Draw Graph"):
    model = load_model(model_name)
    graph = build_graph(model, text, image_path, norm_threshold if renormalize_after_threshold else None)
    tokens = model.tokens()[0]
    n_tokens = tokens.shape[0]
    sparse_graph = threshold_graph(graph, model.model_info().n_layers, n_tokens, sparse_threshold)

    graph_container = st.container()

    with graph_container:
        contribution_graph(
            model.model_info(),
            model.tokens_to_strings(tokens),
            sparse_graph,
            key=f"graph_{hash(text + image_path)}",
        )
    pruned_graph = sparse_graph[0]
    starting_nodes = {}
    for node in pruned_graph.nodes:
        # print(pruned_graph.in_edges(node))
        if len(pruned_graph.in_edges(node)) == 0:
            # sum up all the weights of out edges:
            starting_nodes[node] = sum([c["weight"] for u, v, c in pruned_graph.out_edges(node, data=True)])

    beg, end = model.image_token_pos(0)
    heatmap = np.zeros((24, 24))
    for node, weight in starting_nodes.items():
        node_pos = int(node.split("_")[-1])
        if beg <= node_pos < end:
            node_pos -= beg
            heatmap[node_pos // 24, node_pos % 24] = weight
    with st.sidebar.expander("Heatmap", expanded=True):
        fig, ax = plt.subplots()
        heatmap_overlay = plot_image_with_heatmap(Image.open(image_path), heatmap, ax=ax)
        st.pyplot(fig)
