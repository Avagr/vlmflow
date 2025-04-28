from abc import abstractmethod, ABC
from numbers import Number

from graph_tool import Graph, edge_endpoint_property, GraphView
from graph_tool.search import dfs_iterator
from graph_tool.spectral import adjacency
from graph_tool.topology import shortest_distance
import numpy as np
import torch

EPS = 1e-5


class BaseVertexMetric(ABC):
    labels: list[str] = []
    name = "base"

    @staticmethod
    @abstractmethod
    def __call__(graph: Graph, node_layers: dict[int, list[int]]) -> tuple[Graph, list[np.ndarray]]:
        pass


class ModalityContribution(BaseVertexMetric):
    name = "modality_contribution"
    labels = ["txt_contrib", "img_contrib"]

    @staticmethod
    def __call__(graph: Graph, node_layers: dict[int, list[int]]) -> tuple[Graph, list[np.ndarray]]:
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

        return graph, [graph.vp.txt_contrib.a, graph.vp.img_contrib.a]


class ModalityCentrality(BaseVertexMetric):
    name = "modality_centrality"
    labels = ["txt_centrality", "img_centrality"]

    @staticmethod
    def __call__(graph: Graph, node_layers: dict[int, list[int]]) -> tuple[Graph, list[np.ndarray]]:
        graph.vp.img_centrality = graph.new_vertex_property("short", val=0)
        graph.vp.txt_centrality = graph.new_vertex_property("short", val=0)

        for v in node_layers[0]:
            img_set = set()
            txt_set = set()
            if graph.vp.modality[v] == 'txt':
                for b, e in dfs_iterator(graph, v, array=True):
                    txt_set.add(b)
                    txt_set.add(e)
                graph.vp.txt_centrality.a[list(txt_set)] += 1
            elif graph.vp.modality[v] == 'img':
                for b, e in dfs_iterator(graph, v, array=True):
                    img_set.add(b)
                    img_set.add(e)
                graph.vp.img_centrality.a[list(img_set)] += 1

        return graph, [graph.vp.txt_centrality.a, graph.vp.img_centrality.a]


class LocalClusteringCoefficient(BaseVertexMetric):
    name = "local_clustering_coefficient"
    labels = ["local_clustering_coefficient"]

    @staticmethod
    def __call__(graph: Graph, node_layers: dict[int, list[int]]) -> tuple[Graph, list[np.ndarray]]:
        graph.vp.local_clustering = graph.new_vertex_property("double", val=np.nan)
        adj = adjacency(graph).astype(np.short).T
        num_paths = (adj @ adj).toarray()
        for v in graph.iter_vertices():
            if 0 < graph.vp.layer_num[v] < len(node_layers) - 1:
                in_neigh = graph.get_in_neighbours(v, vprops=[graph.vp.token_num])
                out_neigh = graph.get_out_neighbours(v, vprops=[graph.vp.token_num])
                num_possible_paths = in_neigh.shape[0] * out_neigh.shape[0]
                if np.isin(in_neigh[:, 1], out_neigh[:, 1], assume_unique=True).any():
                    num_possible_paths -= 1
                if num_possible_paths == 0:
                    graph.vp.local_clustering[v] = 0
                    continue
                in_out_paths = num_paths[in_neigh[:, 0, None], out_neigh[:, 0]]
                graph.vp.local_clustering[v] = (in_out_paths > 1).sum() / num_possible_paths

        return graph, [graph.vp.local_clustering.a]


class OldClosenessCentrality(BaseVertexMetric):
    name = "old_closeness_centrality"
    labels = ["old_closeness_centrality", "old_img_closeness"]

    @staticmethod
    def __call__(graph: Graph, _) -> tuple[Graph, list[np.ndarray]]:
        graph.vp.old_closeness_centrality = graph.new_vertex_property("double", val=0)
        graph.vp.old_img_closeness = graph.new_vertex_property("int", val=0)

        vertices = graph.get_vertices(vprops=[graph.vp.token_num])
        all_img_vertices = vertices[
            (graph.vp.token_num.a >= graph.gp.img_begin) & (graph.vp.token_num.a < graph.gp.img_end)]

        if graph.num_vertices() == 0 or all_img_vertices.shape[0] == 0:
            return graph, [graph.vp.old_closeness_centrality.a, graph.vp.old_img_closeness.a]

        distances = shortest_distance(GraphView(graph, reversed=True), directed=True).get_2d_array(
            pos=all_img_vertices[:, 0]
        )

        normalized_inverse_distances = np.divide(distances.shape[0] - 1., distances,
                                                 where=np.isfinite(distances) & (distances != 0),
                                                 out=np.zeros_like(distances, dtype=float))

        graph.vp.old_closeness_centrality.a = normalized_inverse_distances.sum(axis=0)

        graph.vp.old_img_closeness.a = distances.min(axis=0)
        reached = graph.vp.old_img_closeness.a < 1000000
        graph.vp.old_img_closeness.a = np.where(reached, graph.gp.num_layers - graph.vp.old_img_closeness.a, 0)

        return graph, [graph.vp.old_closeness_centrality.a, graph.vp.old_img_closeness.a]


class ClosenessCentrality(BaseVertexMetric):
    name = "closeness_centrality"
    labels = ["closeness_centrality", "img_closeness"]

    @staticmethod
    def __call__(graph: Graph, _) -> tuple[Graph, list[np.ndarray]]:
        graph.vp.closeness_centrality = graph.new_vertex_property("double", val=0)
        graph.vp.img_closeness = graph.new_vertex_property("int", val=0)

        vertices = graph.get_vertices(vprops=[graph.vp.token_num])
        all_img_vertices = vertices[(graph.vp.token_num.a >= graph.gp.img_begin) &
                                    (graph.vp.token_num.a < graph.gp.img_end) &
                                    (graph.vp.layer_num.a == 0)] # Only the image tokens in the first layer

        if graph.num_vertices() == 0 or all_img_vertices.shape[0] == 0:
            return graph, [graph.vp.closeness_centrality.a, graph.vp.img_closeness.a]

        distances = shortest_distance(GraphView(graph, reversed=True), directed=True).get_2d_array(
            pos=all_img_vertices[:, 0]
        )

        normalized_inverse_distances = np.divide(distances.shape[0] - 1., distances,
                                                 where=np.isfinite(distances) & (distances != 0),
                                                 out=np.zeros_like(distances, dtype=float))

        graph.vp.closeness_centrality.a = normalized_inverse_distances.sum(axis=0)

        graph.vp.img_closeness.a = distances.min(axis=0)
        reached = graph.vp.img_closeness.a < 1000000
        graph.vp.img_closeness.a = np.where(reached, graph.gp.num_layers - graph.vp.img_closeness.a, 0)

        return graph, [graph.vp.closeness_centrality.a, graph.vp.img_closeness.a]


class BaseGraphMetric(ABC):
    labels = []
    name = "base"

    @staticmethod
    @abstractmethod
    def __call__(graph: Graph) -> list:
        pass


@torch.compile()
def density_helper(layer_nums, token_nums, img_tokens_mask):
    mask = (layer_nums[:, None] == (layer_nums - 1)) & (token_nums[:, None] <= token_nums)
    img_mask = mask & (img_tokens_mask[:, None] & img_tokens_mask)
    txt_mask = mask & (~img_tokens_mask[:, None] & ~img_tokens_mask)
    return mask.sum(), img_mask.sum(), txt_mask.sum()


class GraphDensity(BaseGraphMetric):
    name = "graph_density"
    labels = ["graph_density", "img_density", "txt_density"]

    @staticmethod
    def __call__(graph: Graph) -> list[Number]:
        if graph.num_edges() == 0:
            return [0, 0, 0]
        img_tokens_mask = (graph.vp.token_num.a >= graph.gp.img_begin) & (graph.vp.token_num.a < graph.gp.img_end)
        img_subgraph = GraphView(graph, vfilt=img_tokens_mask)
        txt_subgraph = GraphView(graph, vfilt=~img_tokens_mask)
        layer_nums = torch.from_numpy(graph.vp.layer_num.a).cuda()
        token_nums = torch.from_numpy(graph.vp.token_num.a).cuda()
        img_tokens_mask = torch.from_numpy(img_tokens_mask).cuda()
        # mask = (layer_nums[:, None] == (layer_nums - 1)) & (token_nums[:, None] <= token_nums)
        # img_mask = mask & (img_tokens_mask[:, None] & img_tokens_mask)
        # txt_mask = mask & (~img_tokens_mask[:, None] & ~img_tokens_mask)
        # graph_density = graph.num_edges() / mask.sum().cpu().item() if mask.sum() > 0 else 0
        # img_density = img_subgraph.num_edges() / img_mask.sum().cpu().item()  if img_mask.sum() > 0 else 0
        # txt_density = txt_subgraph.num_edges() / txt_mask.sum().cpu().item()  if txt_mask.sum() > 0 else 0
        mask_sum, img_mask_sum, txt_mask_sum = density_helper(layer_nums, token_nums, img_tokens_mask)
        graph_density = graph.num_edges() / mask_sum.cpu().item() if mask_sum > 0 else 0
        img_density = img_subgraph.num_edges() / img_mask_sum.cpu().item() if img_mask_sum > 0 else 0
        txt_density = txt_subgraph.num_edges() / txt_mask_sum.cpu().item() if txt_mask_sum > 0 else 0
        return [graph_density, img_density, txt_density]


class GraphSaturation(BaseGraphMetric):
    name = "graph_saturation"
    labels = ["node_saturation", "edge_saturation",
              "node_img_saturation", "edge_img_saturation",
              "node_txt_saturation", "edge_txt_saturation"]

    @staticmethod
    def __call__(graph: Graph) -> list[Number]:
        if graph.num_edges() == 0:
            return [0, 0, 0, 0, 0, 0]
        num_layers = graph.gp.num_layers + 1
        num_tokens = graph.vp.token_num.a.max().item() + 1
        num_img_tokens = graph.gp.img_end - graph.gp.img_begin
        num_txt_tokens = num_tokens - num_img_tokens
        img_tokens_mask = (graph.vp.token_num.a >= graph.gp.img_begin) & (graph.vp.token_num.a < graph.gp.img_end)
        img_subgraph = GraphView(graph, vfilt=img_tokens_mask)
        txt_subgraph = GraphView(graph, vfilt=~img_tokens_mask)
        node_sat = graph.num_vertices() / (num_layers * num_tokens)
        edge_sat = graph.num_edges() / (num_layers * (num_tokens * (num_tokens + 1) / 2))
        node_img_sat = img_subgraph.num_vertices() / (num_layers * num_img_tokens)
        edge_img_sat = img_subgraph.num_edges() / (num_layers * (num_img_tokens * (num_img_tokens + 1) / 2))
        node_txt_sat = txt_subgraph.num_vertices() / (num_layers * num_txt_tokens)
        edge_txt_sat = txt_subgraph.num_edges() / (num_layers * (num_txt_tokens * (num_txt_tokens + 1) / 2))
        return [node_sat, edge_sat, node_img_sat, edge_img_sat, node_txt_sat, edge_txt_sat]


class GlobalClusteringCoefficient(BaseGraphMetric):
    name = "global_clustering_coefficient"
    labels = ["global_clustering_coefficient"]

    @staticmethod
    def __call__(graph: Graph) -> list[Number]:
        adj = adjacency(graph).astype(np.short).T
        num_paths = (adj @ adj)
        return [(num_paths > 1).sum() / (num_paths > 0).sum()]


class CrossModalEdges(BaseGraphMetric):
    name = "cross_modal_edges"
    labels = ["cross_modal_edges_layers", "cross_modal_edges_num", "cross_modal_edges_ratio"]

    @staticmethod
    def __call__(graph: Graph) -> list[Number]:
        if graph.num_edges() == 0:
            return [np.array([]), 0, 0]
        # vs = graph.get_vertices(vprops=[graph.vp.token_num, graph.vp.layer_num])
        # is_token_image = {}
        # token_vs = vs[vs[:, 2] == 0]
        # for tok in token_vs:
        #     is_token_image[tok[1]] = graph.vp.modality[tok[0]] == 'img'
        vertex_modalities = graph.new_vertex_property("bool", val=False)
        vertex_modalities.a = (graph.vp.token_num.a >= graph.gp.img_begin) & (graph.vp.token_num.a < graph.gp.img_end)
        # for i, tok_num in graph.iter_vertices(vprops=[graph.vp.token_num]):
        #     if tok_num in is_token_image and is_token_image[tok_num]:
        #         vertex_modalities[i] = 1
        source_mod = edge_endpoint_property(graph, vertex_modalities, endpoint='source').a.astype(bool)
        target_mod = edge_endpoint_property(graph, vertex_modalities, endpoint='target').a.astype(bool)
        cme_mask = (source_mod & ~target_mod)  # Not counting the txt -> img edges as it could only be from BoS
        cme_num = cme_mask.sum().item()
        cme_layers = edge_endpoint_property(graph, graph.vp.layer_num, endpoint='source').a[cme_mask]
        return [cme_layers, cme_num, cme_num / graph.num_edges()]


# class NumCrossModalEdgesCentrality(BaseGraphMetric):
#     name = "num_cross_modal_edges_centrality"
#     labels = ["num_cross_modal_edges_centrality"]
#
#     @staticmethod
#     def __call__(graph: Graph) -> list[Number]:
#         source_txt = edge_endpoint_property(graph, graph.vp.txt_centrality, endpoint='source').a
#         target_img = edge_endpoint_property(graph, graph.vp.img_centrality, endpoint='target').a
#         target_txt = edge_endpoint_property(graph, graph.vp.txt_centrality, endpoint='target').a
#
#         return [((source_txt == 0) & (target_img > 0) & (target_txt > 0)).sum().item() / graph.num_edges()]


class ResidualStreamHeights(BaseGraphMetric):
    name = "residual_stream_heights"
    labels = ["residual_stream_heights"]

    @staticmethod
    def __call__(graph: Graph) -> list[dict[int, int]]:
        vs = graph.get_vertices(vprops=[graph.vp.token_num, graph.vp.layer_num])
        token_height = {}
        for _, v, h in vs:
            if v not in token_height:
                token_height[v.item()] = h.item()
            else:
                token_height[v.item()] = max(token_height[v.item()], h.item())
        return [token_height]


class CentralityOverlap(BaseGraphMetric):
    name = "centrality_overlap"
    labels = ["jaccard_index", "txt_difference", "img_difference"]

    @staticmethod
    def __call__(graph: Graph) -> list[Number]:
        txt_centrality = graph.vp.txt_centrality.a
        img_centrality = graph.vp.img_centrality.a
        total_centrality = txt_centrality + img_centrality
        intersection_centrality = np.where(txt_centrality > img_centrality, img_centrality, txt_centrality)
        img_difference = img_centrality - intersection_centrality
        txt_difference = txt_centrality - intersection_centrality
        return [
            (intersection_centrality.sum() / total_centrality.sum()).item(),
            (txt_difference.sum() / total_centrality.sum()).item(),
            (img_difference.sum() / total_centrality.sum()).item()
        ]
