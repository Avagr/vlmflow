# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch
from jaxtyping import Float, Int
from torch import nn


@dataclass
class ModelInfo:
    name: str

    n_layers: int
    n_heads: int
    d_model: int
    d_vocab: int


class TransparentLlm(ABC, nn.Module):
    """
    An abstract stateful interface for a language model. The model is supposed to be
    loaded at the class initialization.

    The internal state is the resulting tensors from the last call of the `run` method.
    Most of the methods could return values based on the state, but some may do cheap
    computations based on them.
    """

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """
        Gives general info about the model. This method must be available before any
        calls of the `run`.
        """
        pass

    @abstractmethod
    def forward(self, **inputs):
        pass

    @abstractmethod
    def generate(self, **inputs):
        pass

    @abstractmethod
    def batch_size(self) -> int:
        """
        The size of the batch that was used for the last call of `run`.
        """
        pass

    @abstractmethod
    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        pass

    @abstractmethod
    def image_token_pos(self, batch_i: int) -> (int, int):
        pass

    @staticmethod
    @abstractmethod
    def tokens_to_strings(tokens: Int[torch.Tensor, "pos"], tokenizer) -> List[str]:
        pass

    @abstractmethod
    def logit_lens(self, after_layer: int, token: int, normalize: bool) -> Float[torch.Tensor, "batch vocab"]:
        """
        Return the logits for the given token after the given layer using the "logit lens" technique
        """
        pass


    @abstractmethod
    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        pass

    @abstractmethod
    def unembed(
            self,
            t: Float[torch.Tensor, "d_model"],
            normalize: bool,
    ) -> Float[torch.Tensor, "vocab"]:
        """
        Project the given vector (for example, the state of the residual stream for a
        layer and token) into the output vocabulary.

        normalize: whether to apply the final normalization before the unembedding.
        Setting it to True and applying to output of the last layer gives the output of
        the model.
        """
        pass

    @abstractmethod
    def underlying_model(self) -> nn.Module:
        """
        Return the underlying model to access some additional methods that are not part of the interface.
        """
        pass

    # ================= Methods related to the residual stream =================

    @abstractmethod
    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        The state of the residual stream before entering the layer. For example, when
        layer == 0 these must the embedded tokens (including positional embedding).
        """
        pass

    @abstractmethod
    def residual_after_attn(
            self, layer: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        The state of the residual stream after attention, but before the FFN in the
        given layer.
        """
        pass

    @abstractmethod
    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        The state of the residual stream after the given layer. This is equivalent to the
        next layer's input.
        """
        pass

    # ================ Methods related to the feed-forward layer ===============

    @abstractmethod
    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        The output of the FFN layer, before it gets merged into the residual stream.
        """
        pass

    @abstractmethod
    def decomposed_ffn_out(
            self,
            batch_i: int,
            layer: int,
            pos: int,
    ) -> Float[torch.Tensor, "hidden d_model"]:
        """
        A collection of vectors added to the residual stream by each neuron. It should
        be the same as neuron activations multiplied by neuron value.
        """
        pass

    @abstractmethod
    def neuron_activations(
            self,
            batch_i: int,
            layer: int,
            pos: int,
    ) -> Float[torch.Tensor, "d_ffn"]:
        """
        The content of the hidden layer right after the activation function was applied.
        """
        pass

    @abstractmethod
    def neuron_output(
            self,
            layer: int,
            neuron: int,
    ) -> Float[torch.Tensor, "d_model"]:
        """
        Return the value that the given neuron adds to the residual stream. It's a raw
        vector from the model parameters, no activation involved.
        """
        pass

    # ==================== Methods related to the attention ====================

    @abstractmethod
    def attention_matrix(
            self, batch_i, layer: int, head: int
    ) -> Float[torch.Tensor, "query_pos key_pos"]:
        """
        Return a lower-diagonal attention matrix.
        """
        pass

    @abstractmethod
    def attention_output(
            self,
            batch_i: int,
            layer: int,
            pos: int,
            head: int,
    ) -> Float[torch.Tensor, "d_model"]:
        """
        Return what the given head at the given layer and pos added to the residual
        stream.
        """
        pass

    @abstractmethod
    def decomposed_attn_components(
            self, batch_i: int, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def decomposed_attn_head(
            self, head_i: int, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "source target d_model"]:
        pass

    @abstractmethod
    def decomposed_attn_head_slice(
            self, head_start: int, head_end: int, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "source target slice d_model"]:
        pass

    @abstractmethod
    def decomposed_attn(
            self, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "source target head d_model"]:
        """
        Here
        - source: index of token from the previous layer
        - target: index of token on the current layer
        The decomposed attention tells what vector from source representation was used
        in order to contribute to the taget representation.
        """
        pass
