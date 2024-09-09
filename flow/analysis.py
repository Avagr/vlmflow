from abc import abstractmethod, ABC

from graph_tool import Graph
from graph_tool.search import dfs_iterator
import numpy as np


class BaseMetric(ABC):
    labels = []
    name = "base"

    @staticmethod
    @abstractmethod
    def __call__(graph: Graph, node_layers: dict[int, list[int]]) -> list[np.ndarray]:
        pass


class ModalityRatio(BaseMetric):
    name = "modality_ratio"
    labels = ["txt_contrib", "img_contrib"]

    @staticmethod
    def __call__(graph: Graph, node_layers: dict[int, list[int]]) -> list[np.ndarray]:
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

        return [graph.vp.txt_contrib.a, graph.vp.img_contrib.a]

class ModalityCentrality(BaseMetric):
    name = "modality_centrality"
    labels = ["txt_centrality", "img_centrality"]

    @staticmethod
    def __call__(graph: Graph, node_layers: dict[int, list[int]]) -> list[np.ndarray]:
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

        return [graph.vp.txt_centrality.a, graph.vp.img_centrality.a]
