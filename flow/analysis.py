from graph_tool import Graph


def modality_ratio(graph: Graph, node_layers: dict[int, list[int]]) -> dict[int, list[float]]:
    graph.vp.txt_contrib = graph.new_vertex_property("float", val=0.)
    graph.vp.img_contrib = graph.new_vertex_property("float", val=0.)

    for v in node_layers[0]:
        if graph.vp.modality[v] == 'txt':
            graph.vp.txt_contrib[v] = 1.
        elif graph.vp.modality[v] == 'img':
            graph.vp.img_contrib[v] = 1.

    for layer in range(1, graph.gp.num_layers + 1):
        for v in node_layers[layer]:
            w_sum = 0.
            img_sum = 0.
            txt_sum = 0.
            for s, _, w in graph.iter_in_edges(v, [graph.ep.weight]):
                img_sum += graph.vp.img_contrib[s] * w
                txt_sum += graph.vp.txt_contrib[s] * w
                w_sum += w
            graph.vp.txt_contrib[v] = txt_sum / w_sum
            graph.vp.img_contrib[v] = img_sum / w_sum

    return {k: [graph.vp.img_contrib[v] for v in lst] for k, lst in node_layers.items()}