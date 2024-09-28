from abc import ABC, abstractmethod
from pathlib import Path

import torch

from flow.contributions import get_contribution_matrices
from models.wrappers import GenerativeWrapper


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


class ImageBoundariesCallback(Callback):

    @property
    def score_labels(self) -> list[str]:
        return ["img_begin", "img_end"]

    def __init__(self):
        self.img_begin = {}
        self.img_end = {}

    def __call__(self, model_wrapper: GenerativeWrapper, idx, *_):
        for pos, i in enumerate(idx.tolist()):
            beg, end = model_wrapper.model.image_token_pos(pos)
            self.img_begin[i] = beg
            self.img_end[i] = end

    def get_results(self):
        return [self.img_begin, self.img_end]


class LogitLensCallback(Callback):
    @property
    def score_labels(self) -> list[str]:
        return [f'"{class_name}" class_logits' for class_name in self.target_tokens] + ["logit_argmax"]

    def __init__(self, target_vocabulary: list[str], normalize_lens: bool, tokenizer):
        self.target_tokens = target_vocabulary
        self.target_class_idx = tokenizer(target_vocabulary, add_special_tokens=False, return_tensors='pt',
                                          padding=False).input_ids.view(-1).tolist()
        assert len(self.target_class_idx) == len(target_vocabulary), "Each class should be a single token"
        self.class_logits = {k: {} for k in target_vocabulary}
        self.logit_argmax = {}
        self.normalize_lens = normalize_lens
        self.tokenizer = tokenizer

    def __call__(self, model_wrapper, idx, *_):
        model = model_wrapper.model
        num_layers = model.model_info().n_layers
        for i in idx.tolist():
            self.logit_argmax[i] = []
            for cls in self.target_tokens:
                self.class_logits[cls][i] = []

        for layer in range(num_layers):
            # Only support the last token for now, can make more general if needed
            logits = model.logit_lens(after_layer=layer, token=-1, normalize=self.normalize_lens)
            for batch_idx, i in enumerate(idx.tolist()):
                self.logit_argmax[i].append(self.tokenizer.decode(logits[batch_idx].argmax()))
                for cls, pos in zip(self.target_tokens, self.target_class_idx):
                    self.class_logits[cls][i].append(logits[batch_idx, pos].item())


    def get_results(self) -> list[dict]:
        return [self.class_logits[cls] for cls in self.target_tokens] + [self.logit_argmax]


class GraphTensorCallback(Callback):
    @property
    def score_labels(self) -> list[str]:
        return ["full_graph_path"]

    def __init__(self, graph_save_dir: Path, head_batch_size: int):
        self.graph_save_dir = graph_save_dir
        self.graph_paths = {}
        self.head_batch_size = head_batch_size

    def __call__(self, model_wrapper, idx, *_):
        batch_size = len(idx)
        graph_tensors = [get_contribution_matrices(model_wrapper.model, i, head_batch_size=self.head_batch_size) for i
                         in range(batch_size)]

        for i, tensors in zip(idx.tolist(), graph_tensors):
            graph_file = self.graph_save_dir / f"{i}_graph_tensors.pkl"
            torch.save(tensors, graph_file)
            self.graph_paths[i] = str(graph_file.resolve())

    def get_results(self) -> list[dict]:
        return [self.graph_paths]


class ResidualsByLayerCallback(Callback):
    @property
    def score_labels(self) -> list[str]:
        return ["residuals"]

    def __init__(self, residual_save_dir: Path, num_last_tokens: int, from_layer: int):
        self.residual_save_dir = residual_save_dir
        self.num_last_tokens = num_last_tokens
        self.from_layer = from_layer
        self.res_paths = {}

    def __call__(self, model_wrapper, idx, *_):
        model = model_wrapper.model
        num_layers = model.n_layers
        residuals_out = torch.stack([model.residual_out(layer).cpu() for layer in range(self.from_layer, num_layers)], dim=1)
        for i, res in zip(idx.tolist(), residuals_out):
            residual_file = self.residual_save_dir / f"{i}_residuals.pt"
            torch.save(res[:, -self.num_last_tokens:, :].clone(), residual_file)
            self.res_paths[i] = str(residual_file.resolve())

    def get_results(self) -> list[dict]:
        return [self.res_paths]


