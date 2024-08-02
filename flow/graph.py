# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import einops
import networkx as nx
import torch
from graph_tool import Graph
from tqdm.auto import tqdm

import flow.contributions as contributions
from models.transparent_llm import TransparentLlm
from models.transparent_models import TransparentLlava


class GraphBuilder:
    """
    Constructs the contributions graph with edges given one by one. The resulting graph
    is a networkx graph that can be accessed via the `graph` field. It contains the
    following types of nodes:

    - X0_<token>: the original token.
    - A<layer>_<token>: the residual stream after attention at the given layer for the
        given token.
    - M<layer>_<token>: the ffn block.
    - I<layer>_<token>: the residual stream after the ffn block.
    """

    def __init__(self, n_layers: int, n_tokens: int):
        self._n_layers = n_layers
        self._n_tokens = n_tokens

        self.graph = nx.DiGraph()
        for layer in range(n_layers):
            for token in range(n_tokens):
                self.graph.add_node(f"A{layer}_{token}")
                self.graph.add_node(f"I{layer}_{token}")
                self.graph.add_node(f"M{layer}_{token}")
        for token in range(n_tokens):
            self.graph.add_node(f"X0_{token}")

    def get_output_node(self, token: int):
        return f"I{self._n_layers - 1}_{token}"

    def _add_edge(self, u: str, v: str, weight: float):
        # TODO(igortufanov): Here we sum up weights for multi-edges. It happens with
        # attention from the current token and the residual edge. Ideally these need to
        # be 2 separate edges, but then we need to do a MultiGraph. Multigraph is fine,
        # but when we try to traverse it, we face some NetworkX issue with EDGE_OK
        # receiving 3 arguments instead of 2.
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] += weight
        else:
            self.graph.add_edge(u, v, weight=weight)

    def add_attention_edge(self, layer: int, token_from: int, token_to: int, w: float):
        self._add_edge(
            f"I{layer - 1}_{token_from}" if layer > 0 else f"X0_{token_from}",
            f"A{layer}_{token_to}",
            w,
        )

    def add_residual_to_attn(self, layer: int, token: int, w: float):
        self._add_edge(
            f"I{layer - 1}_{token}" if layer > 0 else f"X0_{token}",
            f"A{layer}_{token}",
            w,
        )

    def add_ffn_edge(self, layer: int, token: int, w: float):
        self._add_edge(f"A{layer}_{token}", f"M{layer}_{token}", w)
        self._add_edge(f"M{layer}_{token}", f"I{layer}_{token}", w)

    def add_residual_to_ffn(self, layer: int, token: int, w: float):
        self._add_edge(f"A{layer}_{token}", f"I{layer}_{token}", w)


@torch.no_grad()
def build_full_graph(
        model: TransparentLlm,
        batch_i: int = 0,
        renormalizing_threshold: Optional[float] = None,
        memory_efficient: bool = True,
) -> nx.DiGraph:
    """
    Build the contribution graph for all blocks of the model and all tokens.

    model: The transparent llm which already did the inference.
    batch_i: Which sentence to use from the batch that was given to the model.
    renormalizing_threshold: If specified, will apply renormalizing thresholding to the
    contributions. All contributions below the threshold will be erazed and the rest
    will be renormalized.
    """
    n_layers = model.model_info().n_layers
    n_heads = model.model_info().n_heads
    n_tokens = model.tokens()[batch_i].shape[0]
    builder = GraphBuilder(n_layers, n_tokens)
    distance_norm = 1

    for layer in tqdm(range(n_layers), disable=True):  # TODO
        if memory_efficient:
            c_attn, c_resid_attn = get_attention_contributions_efficiently(model, layer, batch_i, n_tokens, n_heads,
                                                                           distance_norm)
        else:
            c_attn, c_resid_attn = contributions.get_attention_contributions(
                resid_pre=model.residual_in(layer)[batch_i].unsqueeze(0),
                resid_mid=model.residual_after_attn(layer)[batch_i].unsqueeze(0),
                decomposed_attn=model.decomposed_attn(batch_i, layer).unsqueeze(0),
            )

        if renormalizing_threshold is not None:
            c_attn, c_resid_attn = contributions.apply_threshold_and_renormalize(
                renormalizing_threshold, c_attn, c_resid_attn
            )

        contrib = c_attn[batch_i].sum(dim=-1).tolist()
        edges_to_add = []
        if layer == 0:
            for token_from in range(n_tokens):
                for token_to in range(n_tokens)[token_from:]:
                    if contrib[token_to][token_from] != 0:
                        edges_to_add.append(
                            (f"X0_{token_from}", f"A{layer}_{token_to}", contrib[token_to][token_from]))
        else:
            for token_from in range(n_tokens):
                for token_to in range(n_tokens)[token_from:]:
                    if contrib[token_to][token_from] != 0:
                        edges_to_add.append(
                            (f"I{layer - 1}_{token_from}", f"A{layer}_{token_to}", contrib[token_to][token_from]))

        builder.graph.add_weighted_edges_from(edges_to_add)

        for token in range(n_tokens):
            builder.add_residual_to_attn(
                layer, token, c_resid_attn[batch_i, token].item()
            )

        c_ffn, c_resid_ffn = contributions.get_mlp_contributions(
            resid_mid=model.residual_after_attn(layer)[batch_i].unsqueeze(0),
            resid_post=model.residual_out(layer)[batch_i].unsqueeze(0),
            mlp_out=model.ffn_out(layer)[batch_i].unsqueeze(0),
        )
        if renormalizing_threshold is not None:
            c_ffn, c_resid_ffn = contributions.apply_threshold_and_renormalize(
                renormalizing_threshold, c_ffn, c_resid_ffn
            )

        for token in range(n_tokens):
            builder.add_ffn_edge(layer, token, c_ffn[batch_i, token].item())
            builder.add_residual_to_ffn(
                layer, token, c_resid_ffn[batch_i, token].item()
            )

    return builder.graph


def apply_graph_threshold(
        graph: nx.DiGraph,
        n_layers: int,
        n_tokens: int,
        starting_tokens: List[int],
        threshold: float,
) -> List[nx.Graph]:
    """
    Given the full graph, this function returns only the trees leading to the specified
    tokens. Edges with weight below `threshold` will be ignored.
    """
    builder = GraphBuilder(n_layers, n_tokens)

    rgraph = graph.reverse(copy=False)
    search_graph = nx.subgraph_view(
        rgraph, filter_edge=lambda u, v: rgraph[u][v]["weight"] > threshold
    )

    result = []

    from tqdm.auto import tqdm

    for start in tqdm(starting_tokens, disable=True):
        assert start < n_tokens
        assert start >= 0
        edges = nx.edge_dfs(search_graph, source=builder.get_output_node(start))
        tree = search_graph.edge_subgraph(edges)
        # Reverse the edges because the dfs was going from upper layer downwards.
        result.append(tree.reverse(copy=False))
        # print(result[-1].number_of_nodes(), result[-1].number_of_edges())
    return result


@torch.compile
def pairwise_distances(rearranged, whole, p):
    return torch.nn.functional.pairwise_distance(rearranged, whole.expand(rearranged.shape), p=p)


@torch.no_grad()
def get_contribution_matrices(
        model: TransparentLlava,
        batch_i: int,
        head_batch_size: int
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    n_layers = model.model_info().n_layers
    n_heads = model.model_info().n_heads
    n_tokens = model.tokens()[batch_i].shape[0]
    distance_norm = 1

    attn_contributions = []
    res_attn_contributions = []
    ffn_contributions = []
    res_ffn_contributions = []

    for layer in range(n_layers):
        c_attn, c_resid_attn = get_attention_contributions_efficiently(
            model, layer, batch_i, n_tokens, n_heads,
            distance_norm, head_batch_size=head_batch_size
        )
        contrib = c_attn[batch_i].sum(dim=-1)

        attn_contributions.append(contrib.cpu().squeeze())
        res_attn_contributions.append(c_resid_attn.cpu().squeeze())

        c_ffn, c_resid_ffn = contributions.get_mlp_contributions(
            resid_mid=model.residual_after_attn(layer)[batch_i].unsqueeze(0),
            resid_post=model.residual_out(layer)[batch_i].unsqueeze(0),
            mlp_out=model.ffn_out(layer)[batch_i].unsqueeze(0),
        )

        ffn_contributions.append(c_ffn.cpu().squeeze())
        res_ffn_contributions.append(c_resid_ffn.cpu().squeeze())

    return attn_contributions, res_attn_contributions, ffn_contributions, res_ffn_contributions


def get_attention_contributions_efficiently(model, layer, batch_i, n_tokens, n_heads, distance_norm, head_batch_size=1):
    one_off = model.residual_in(layer)[batch_i].unsqueeze(0)
    whole = model.residual_after_attn(layer)[batch_i].unsqueeze(0)
    distances = []
    for head_i in range(0, n_heads, head_batch_size):
        decomposed_attn = model.decomposed_attn_head_slice(head_i, head_i + head_batch_size, batch_i, layer)

        rearranged = einops.rearrange(
            decomposed_attn,
            "pos key_pos head d_model -> key_pos head pos d_model",
        )
        distances.append(pairwise_distances(rearranged, whole, p=distance_norm))
    distance = torch.cat(distances, dim=1).flatten(start_dim=0, end_dim=1)
    distance = torch.cat([distance, torch.nn.functional.pairwise_distance(one_off, whole, p=distance_norm)],
                         dim=0).unsqueeze(1)
    whole_norm = torch.norm(whole, p=distance_norm, dim=-1)
    distance = (whole_norm - distance).clip(min=1e-5)
    s = distance.sum(dim=0, keepdim=True)
    contrib = distance / s
    parts_contributions, one_off_contributions = torch.split(
        contrib, contrib.shape[0] - 1
    )
    attn_contribution, residual_contribution = (
        parts_contributions.unflatten(0, (n_tokens, n_heads)),
        one_off_contributions[0].clone(),
    )
    c_attn = einops.rearrange(attn_contribution, "key_pos head batch pos -> batch pos key_pos head")
    c_resid_attn = residual_contribution
    return c_attn, c_resid_attn


def build_thresholded_graph(top_token_num: int, threshold: float, attn_contribs: list[torch.Tensor],
                            ffn_contribs: list[torch.Tensor], ffn_res_contribs: list[torch.Tensor]) -> Graph:
    threshold_attn = [(c > threshold).tolist() for c in attn_contribs]
    threshold_ffn = [(c > threshold).tolist() for c in ffn_contribs]
    threshold_ffn_res = [(c > threshold).tolist() for c in ffn_res_contribs]
    edge_list = []
    num_layers = len(attn_contribs)
    num_tokens = attn_contribs[0].size(0)
    if top_token_num < 0:
        top_token_num = num_tokens - 1
    stack = [(top_token_num, num_layers - 1)]

    visited_nodes = set()

    while len(stack) > 0:
        token_num, layer_num = stack[-1]
        visited_nodes.add((token_num, layer_num))
        stack.pop()

        if threshold_ffn[layer_num][token_num]:
            weight = ffn_contribs[layer_num][token_num].item()
            edge_list.append((f"A{layer_num}_{token_num}", f"M{layer_num}_{token_num}", weight))
            edge_list.append((f"M{layer_num}_{token_num}", f"I{layer_num}_{token_num}", weight))
        if threshold_ffn_res[layer_num][token_num]:
            edge_list.append((f"A{layer_num}_{token_num}", f"I{layer_num}_{token_num}",
                              ffn_res_contribs[layer_num][token_num].item()))

        for token_from in range(num_tokens)[:token_num + 1]:
            if threshold_attn[layer_num][token_num][token_from]:
                if layer_num > 0:
                    edge_list.append((f"I{layer_num - 1}_{token_from}", f"A{layer_num}_{token_num}",
                                      attn_contribs[layer_num][token_num][token_from].item()))
                    if (token_from, layer_num - 1) not in visited_nodes:
                        stack.append((token_from, layer_num - 1))
                else:
                    edge_list.append(
                        (f"X0_{token_from}", f"A0_{token_num}", attn_contribs[layer_num][token_num][token_from].item()))
    return Graph(directed=True, g=edge_list, eprops=[("weight", "float")], hashed=True)
