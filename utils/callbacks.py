import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from flow.graph import build_full_graph, apply_graph_threshold, get_contribution_matrices


class Callback(ABC):

    @property
    @abstractmethod
    def score_labels(self) -> list[str]:
        pass

    @abstractmethod
    def __call__(self, model_wrapper, idx, texts, answers, predictions, scores):
        pass

    @abstractmethod
    def get_results(self):
        pass


class GraphCallback(Callback):

    @property
    def score_labels(self) -> list[str]:
        return ["full_graph_path"]

    def __init__(self, graph_save_dir: Path, renormalization_threshold: float | None = None,
                 sparsification_threshold: float | None = None):
        self.graph_save_dir = graph_save_dir
        self.renormalization_threshold = renormalization_threshold
        self.sparsification_threshold = sparsification_threshold
        self.graph_paths = {}

    def __call__(self, model_wrapper, idx, *_):
        batch_size = len(idx)
        graphs = []
        for i in range(batch_size):
            full_graph = build_full_graph(model_wrapper.model, i, self.renormalization_threshold, memory_efficient=True)
            if self.sparsification_threshold is not None:
                num_tokens = model_wrapper.model.tokens()[i].shape[0]
                graphs.append(apply_graph_threshold(full_graph, model_wrapper.model.model_info().n_layers, num_tokens,
                                                    [num_tokens - 1], self.sparsification_threshold)[0])
            else:
                graphs.append(full_graph)

        for i, graph in zip(idx.tolist(), graphs):
            graph_file = self.graph_save_dir / f"{i}_full_graph.pkl"
            graph.graph["idx"] = i
            pickle.dump(graph, open(graph_file, "wb"))
            self.graph_paths[i] = str(graph_file.resolve())

    def get_results(self) -> list[dict]:
        return [self.graph_paths]


class GraphTensorCallback(Callback):
    @property
    def score_labels(self) -> list[str]:
        return ["full_graph_path"]

    def __init__(self, graph_save_dir: Path):
        self.graph_save_dir = graph_save_dir
        self.graph_paths = {}

    def __call__(self, model_wrapper, idx, *_):
        batch_size = len(idx)
        graph_tensors = [get_contribution_matrices(model_wrapper.model, i, head_batch_size=4) for i in
                         range(batch_size)]

        for i, tensors in zip(idx.tolist(), graph_tensors):
            graph_file = self.graph_save_dir / f"{i}_graph_tensors.pkl"
            torch.save(tensors, graph_file)
            self.graph_paths[i] = str(graph_file.resolve())

    def get_results(self) -> list[dict]:
        return [self.graph_paths]
