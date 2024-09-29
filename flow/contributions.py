# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import einops
import torch
from jaxtyping import Float

from models.transparent_models import TransparentLlava


@torch.no_grad()
def get_contributions(
        parts: torch.Tensor,
        whole: torch.Tensor,
        distance_norm: int = 1,
) -> torch.Tensor:
    """
    Compute contributions of the `parts` vectors into the `whole` vector.

    Shapes of the tensors are as follows:
    parts:  p_1 ... p_k, v_1 ... v_n, d
    whole:               v_1 ... v_n, d
    result: p_1 ... p_k, v_1 ... v_n

    Here
    * `p_1 ... p_k`: dimensions for enumerating the parts
    * `v_1 ... v_n`: dimensions listing the independent cases (batching),
    * `d` is the dimension to compute the distances on.

    The resulting contributions will be normalized so that
    for each v_: sum(over p_ of result(p_, v_)) = 1.
    """
    EPS = 1e-5

    k = len(parts.shape) - len(whole.shape)
    assert k >= 0
    assert parts.shape[k:] == whole.shape
    bc_whole = whole.expand(parts.shape)  # new dims p_1 ... p_k are added to the front

    distance = torch.nn.functional.pairwise_distance(parts, bc_whole, p=distance_norm)

    whole_norm = torch.norm(whole, p=distance_norm, dim=-1)
    distance = (whole_norm - distance).clip(min=EPS)

    sum = distance.sum(dim=tuple(range(k)), keepdim=True)
    return distance / sum


@torch.no_grad()
def get_mlp_contributions(
        resid_mid: Float[torch.Tensor, "batch pos d_model"],
        resid_post: Float[torch.Tensor, "batch pos d_model"],
        mlp_out: Float[torch.Tensor, "batch pos d_model"],
        distance_norm: int = 1,
) -> Tuple[Float[torch.Tensor, "batch pos"], Float[torch.Tensor, "batch pos"]]:
    """
    Returns a pair of (mlp, residual) contributions for each sentence and token.
    """

    contributions = get_contributions(
        torch.stack((mlp_out.cpu(), resid_mid.cpu())), resid_post.cpu(), distance_norm
    )
    return contributions[0], contributions[1]


@torch.no_grad()
def apply_threshold_and_renormalize(
        threshold: float,
        c_blocks: torch.Tensor,
        c_residual: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Thresholding mechanism used in the original graphs paper. After the threshold is
    applied, the remaining contributions are renormalized on order to sum up to 1 for
    each representation.

    threshold: The threshold.
    c_residual: Contribution of the residual stream for each representation. This tensor
        should contain 1 element per representation, i.e., its dimensions are all batch
        dimensions.
    c_blocks: Contributions of the blocks. Could be 1 block per representation, like
        ffn, or heads*tokens blocks in case of attention. The shape of `c_residual`
        must be a prefix if the shape of this tensor. The remaining dimensions are for
        listing the blocks.
    """

    block_dims = len(c_blocks.shape)
    resid_dims = len(c_residual.shape)
    bound_dims = block_dims - resid_dims
    assert bound_dims >= 0
    assert c_blocks.shape[0:resid_dims] == c_residual.shape

    # print(f"above threshold: {(c_blocks > threshold).sum()} out of {c_blocks.numel()}")

    c_blocks = c_blocks * (c_blocks > threshold)
    c_residual = c_residual * (c_residual > threshold)

    denom = c_residual + c_blocks.sum(dim=tuple(range(resid_dims, block_dims)))
    return (
        c_blocks / denom.reshape(denom.shape + (1,) * bound_dims),
        c_residual / denom,
    )

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

        c_ffn, c_resid_ffn = get_mlp_contributions(
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
        distances.append(pairwise_distances(rearranged, whole.to(rearranged.device), p=distance_norm))
    distance = torch.cat(distances, dim=1).flatten(start_dim=0, end_dim=1)
    distance = torch.cat([distance, torch.nn.functional.pairwise_distance(one_off.to(whole.device), whole, p=distance_norm)],
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