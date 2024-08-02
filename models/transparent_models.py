from typing import List, Any

import torch
from fancy_einsum import einsum
from jaxtyping import Float, Int

from models.transparent_llm import TransparentLlm, ModelInfo
from utils.hooks import InputsHook, OutputsHook


class TransparentLlava(TransparentLlm):

    def __init__(self, name, llava, processor, device, dtype=torch.bfloat16):
        super().__init__()
        self.name = name
        self.llava = llava
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.n_layers = self.llava.language_model.config.num_hidden_layers
        self.n_heads = self.llava.language_model.config.num_attention_heads
        self.d_model = self.llava.language_model.config.hidden_size
        self.processor.tokenizer.padding_side = 'left'
        self.image_token_id = 32000

        llm = self.llava.language_model
        self.layers = self.llava.language_model.model.layers

        self.res_in_hooks = [InputsHook(0) for _ in range(self.n_layers)] + [InputsHook(0)]
        self.res_in_hook_objs = [self.layers[i].register_forward_pre_hook(hook, with_kwargs=True) for i, hook in
                                 enumerate(self.res_in_hooks[:-1])] + [
                                    llm.model.norm.register_forward_pre_hook(self.res_in_hooks[-1], with_kwargs=True)]

        self.res_mid_hooks = [InputsHook(0) for _ in range(self.n_layers)]
        self.res_mid_hook_objs = [
            self.layers[i].post_attention_layernorm.register_forward_pre_hook(hook, with_kwargs=True) for i, hook
            in enumerate(self.res_mid_hooks)]

        self.ffn_out_hooks = [OutputsHook(0) for _ in range(self.n_layers)]
        self.ffn_out_hook_objs = [self.layers[i].mlp.register_forward_hook(hook) for i, hook in
                                  enumerate(self.ffn_out_hooks)]

        self.value_out_hooks = [OutputsHook(0) for _ in range(self.n_layers)]
        self.value_out_hooks_objs = [self.layers[i].self_attn.v_proj.register_forward_hook(hook) for i, hook in
                                     enumerate(self.value_out_hooks)]

        self.last_run_inputs = None
        self.last_run_logits = None
        self.last_run_attentions = None

    def clear_state(self):
        del self.last_run_inputs
        self.last_run_inputs = None
        del self.last_run_logits
        self.last_run_logits = None
        del self.last_run_attentions
        self.last_run_attentions = None

        for hook_collection in [self.res_in_hooks, self.res_mid_hooks, self.ffn_out_hooks, self.value_out_hooks]:
            for hook in hook_collection:
                del hook.value
                hook.value = None

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            n_params_estimate=1337_1337_1337,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_vocab=self.llava.language_model.config.vocab_size,
        )

    def forward(self, **inputs) -> Any:
        self.clear_state()
        results = self.llava(**inputs)

        self.last_run_inputs = inputs
        self.last_run_logits = results.logits
        self.last_run_attentions = results.attentions
        return results

    def batch_size(self) -> int:
        return self.last_run_logits.shape[0]

    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        input_ids = self.last_run_inputs['input_ids']
        res = torch.zeros(self.last_run_logits.shape[:2], dtype=torch.int64, device=self.device)
        res.fill_(self.image_token_id)
        img_begin = (input_ids == self.image_token_id).nonzero()
        for i, begin in img_begin:
            res[i, :begin] = input_ids[i, :begin]
            res[i, begin + 576:] = input_ids[i, begin + 1:]
        return res

    def image_token_pos(self, batch_i: int) -> (int, int):
        input_ids = self.last_run_inputs['input_ids']
        img_begin = (input_ids[batch_i] == self.image_token_id).nonzero()
        return img_begin.item(), img_begin.item() + 576

    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        res = []
        img_count = 0
        for tok in tokens:
            if tok == self.image_token_id:
                res.append(f"I_{img_count}")
                img_count += 1
            else:
                res.append(self.processor.tokenizer.decode(tok))
        return res

    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        raise NotImplementedError

    def unembed(self, t: Float[torch.Tensor, "d_model"], normalize: bool) -> Float[torch.Tensor, "vocab"]:
        raise NotImplementedError

    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self.res_in_hooks[layer].value

    def residual_after_attn(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self.res_mid_hooks[layer].value

    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self.res_in_hooks[layer + 1].value

    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self.ffn_out_hooks[layer].value

    def decomposed_ffn_out(self, batch_i: int, layer: int, pos: int) -> Float[torch.Tensor, "hidden d_model"]:
        raise NotImplementedError

    def neuron_activations(self, batch_i: int, layer: int, pos: int) -> Float[torch.Tensor, "d_ffn"]:
        raise NotImplementedError

    def neuron_output(self, layer: int, neuron: int) -> Float[torch.Tensor, "d_model"]:
        raise NotImplementedError

    def attention_matrix(self, batch_i, layer: int, head: int) -> Float[torch.Tensor, "query_pos key_pos"]:
        raise NotImplementedError

    def attention_output(self, batch_i: int, layer: int, pos: int, head: int) -> Float[torch.Tensor, "d_model"]:
        raise NotImplementedError

    @torch.no_grad()
    def decomposed_attn_head(
            self, head_i: int, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "source target d_model"]:
        batch_size, num_tokens = self.last_run_logits.shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.n_heads, self.d_model // self.n_heads)[
            batch_i, :, head_i]

        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i, head_i].to(v.dtype)
        z = einsum(
            "key_pos d_head, "
            "query_pos key_pos -> "
            "query_pos key_pos d_head",
            v,
            pattern,
        )
        o_weight = self.layers[layer].self_attn.o_proj.weight.T.view(self.n_heads,
                                                                     self.d_model // self.n_heads,
                                                                     self.d_model)[head_i]
        decomposed_attn = einsum(
            "pos key_pos d_head, "
            "d_head d_model -> "
            "pos key_pos d_model",
            z,
            o_weight
        )
        return decomposed_attn

    @torch.no_grad()
    def decomposed_attn_head_slice(
            self, head_start: int, head_end: int, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "source target head d_model"]:
        batch_size, num_tokens = self.last_run_logits.shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.n_heads, self.d_model // self.n_heads)[
            batch_i, :, head_start:head_end]

        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i, head_start:head_end].to(v.dtype)
        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )
        o_weight = self.layers[layer].self_attn.o_proj.weight.T.view(self.n_heads,
                                                                     self.d_model // self.n_heads,
                                                                     self.d_model)[head_start:head_end]
        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            o_weight
        )
        return decomposed_attn

    @torch.no_grad()
    def decomposed_attn(self, batch_i: int, layer: int) -> Float[torch.Tensor, "source target head d_model"]:
        batch_size, num_tokens = self.last_run_logits.shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.n_heads, self.d_model // self.n_heads)[
            batch_i]
        # b_v = self._model.blocks[layer].attn.b_V
        #
        # # support for gqa
        # num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
        # hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)
        #
        # v = hook_v + b_v
        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i].to(v.dtype)
        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )

        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            self.layers[layer].self_attn.o_proj.weight.T.view(self.n_heads,
                                                              self.d_model // self.n_heads,
                                                              self.d_model),
        )
        return decomposed_attn
