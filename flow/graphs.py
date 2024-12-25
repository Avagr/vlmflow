from graph_tool import Graph
import torch


def build_graph_from_contributions(
        threshold: float, top_token_num: int,
        attn, attn_res, ffn, ffn_res, img_begin: int, img_end: int,
):
    num_layers = len(attn)
    full_attn = [torch.diag(attn_res.squeeze()) + attn for attn, attn_res in zip(attn, attn_res)]
    full_graph, simple_graph = build_thresholded_graph(top_token_num, threshold, full_attn, ffn, ffn_res)
    prune_hanging_branches(simple_graph)

    full_graph.gp.num_layers = full_graph.new_gp("int", val=num_layers)
    simple_graph.gp.num_layers = simple_graph.new_gp("int", val=num_layers)

    # MAYBE also add modality to the full_graph
    simple_graph.vp.modality = simple_graph.new_vertex_property("string", val='mixed')
    simple_graph.vp.token_num = simple_graph.new_vertex_property("int")
    simple_graph.vp.layer_num = simple_graph.new_vertex_property("int")
    node_layers: dict[int, list[int]] = {k: [] for k in range(num_layers + 1)}
    for v in range(simple_graph.num_vertices()):
        layer, num = map(int, simple_graph.vp.ids[v].split('_'))
        simple_graph.vp.token_num[v] = num
        simple_graph.vp.layer_num[v] = layer
        node_layers[layer].append(v)
        if layer == 0:
            if img_begin <= num <= img_end:
                simple_graph.vp.modality[v] = 'img'
            else:
                simple_graph.vp.modality[v] = 'txt'

    return full_graph, simple_graph, node_layers


def build_thresholded_graph(
        top_token_num: int, threshold: float, attn_contribs: list[torch.Tensor],
        ffn_contribs: list[torch.Tensor], ffn_res_contribs: list[torch.Tensor]
) -> tuple[Graph, Graph]:

    threshold_attn = [(c > threshold).tolist() for c in attn_contribs]
    threshold_ffn = [(c > threshold).tolist() for c in ffn_contribs]
    threshold_ffn_res = [(c > threshold).tolist() for c in ffn_res_contribs]
    edge_list = []
    simplified_edge_list = []
    num_layers = len(attn_contribs)
    num_tokens = attn_contribs[0].size(0)
    # 0-based indexing for layers, but we use 1-based indexing in node names, with 0 being the initial tokens
    if top_token_num < 0:
        top_token_num = num_tokens + top_token_num
    stack = [(top_token_num, num_layers - 1)]
    visited_nodes = set()

    while len(stack) > 0:
        token_num, layer_num = stack.pop()
        visited_nodes.add((token_num, layer_num))

        if threshold_ffn[layer_num][token_num]:
            weight = ffn_contribs[layer_num][token_num].item()
            edge_list.append((f"A{layer_num + 1}_{token_num}", f"M{layer_num + 1}_{token_num}", weight))
            edge_list.append((f"M{layer_num + 1}_{token_num}", f"I{layer_num + 1}_{token_num}", weight))
        if threshold_ffn_res[layer_num][token_num]:
            edge_list.append((f"A{layer_num + 1}_{token_num}", f"I{layer_num + 1}_{token_num}",
                              ffn_res_contribs[layer_num][token_num].item()))

        for token_from in range(num_tokens)[:token_num + 1]:
            if threshold_attn[layer_num][token_num][token_from]:
                weight = attn_contribs[layer_num][token_num][token_from].item()
                if layer_num > 0:
                    edge_list.append((f"I{layer_num}_{token_from}", f"A{layer_num + 1}_{token_num}", weight))
                    simplified_edge_list.append((f"{layer_num}_{token_from}", f"{layer_num + 1}_{token_num}", weight))
                    if (token_from, layer_num - 1) not in visited_nodes:
                        stack.append((token_from, layer_num - 1))

                else:
                    edge_list.append((f"X0_{token_from}", f"A1_{token_num}", weight))
                    simplified_edge_list.append((f"0_{token_from}", f"1_{token_num}", weight))

    return (Graph(directed=True, g=edge_list, eprops=[("weight", "float")], hashed=True),
            Graph(directed=True, g=simplified_edge_list, eprops=[("weight", "float")], hashed=True))


def prune_hanging_branches(graph: Graph):
    nodes_to_remove = set()
    for v in graph.vertices():
        if len(graph.get_in_neighbors(v)) == 0 and graph.vp.ids[v][0] != '0':
            nodes_to_remove.add(v)
            # DFS to find nodes that become hanging on this deletion
            stack = [n for n in graph.get_out_neighbors(v)]
            while stack:
                w = stack.pop()
                only_removed_parents = True
                for node in graph.get_in_neighbors(w):
                    if node not in nodes_to_remove:
                        only_removed_parents = False
                        break
                if only_removed_parents:
                    stack.extend([n for n in graph.get_out_neighbors(w)])
                    nodes_to_remove.add(w)

    graph.remove_vertex(nodes_to_remove)
