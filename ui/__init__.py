# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

import networkx as nx
import streamlit.components.v1 as components

from models.transparent_llm import ModelInfo
from ui.graph_selection import GraphSelection, UiGraphNode

_RELEASE = True

if _RELEASE:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        "path": os.path.join(parent_dir, "frontend/build"),
    }
else:
    config = {
        "url": "http://localhost:3001",
    }

_component_func = components.declare_component("contribution_graph", **config)


def is_node_valid(node: UiGraphNode, n_layers: int, n_tokens: int):
    return node.layer < n_layers and node.token < n_tokens


def is_selection_valid(s: GraphSelection, n_layers: int, n_tokens: int):
    if not s:
        return True
    if s.node:
        if not is_node_valid(s.node, n_layers, n_tokens):
            return False
    if s.edge:
        for node in [s.edge.source, s.edge.target]:
            if not is_node_valid(node, n_layers, n_tokens):
                return False
    return True


def contribution_graph(
    n_layers: int,
    tokens: List[str],
    graphs_edge_lists,
    key: str,
    node_style_map: Optional[list[list]] = None,
) -> Optional[int]:
    """Create a new instance of contribution graph.

    Returns selected graph node or None if nothing was selected.
    """

    if node_style_map is None:
        node_style_map = [["I am a", "dummy map"] for _ in range(graphs_edge_lists)]

    # assert len(tokens) == len(graphs_edge_lists) + 1

    result = _component_func(
        component="graph",
        model_info={"n_layers": n_layers},
        tokens=tokens[:-1],
        tokens_top=tokens[1:],
        edges_per_token=graphs_edge_lists,
        default="Default",
        key=key,
        node_style_map=node_style_map
    )

    # check that result is a number
    if isinstance(result, int):
        return result

    return None


def selector(
    items: List[str],
    indices: List[int],
    temperatures: Optional[List[float]],
    preselected_index: Optional[int],
    key: str,
) -> Optional[int]:
    """Create a new instance of selector.

    Returns selected item index.
    """
    n = len(items)
    assert n == len(indices)
    items = [{"index": i, "text": s} for s, i in zip(items, indices)]

    if temperatures is not None:
        assert n == len(temperatures)
        for i, t in enumerate(temperatures):
            items[i]["temperature"] = t

    result = _component_func(
        component="selector",
        items=items,
        preselected_index=preselected_index,
        default=None,
        key=key,
    )

    return None if result is None else int(result)
