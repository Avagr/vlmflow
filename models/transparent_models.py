from typing import List, Any

from fancy_einsum import einsum
from opt_einsum import contract
from jaxtyping import Float, Int
import torch
from torch import nn

from models.molmo.modeling_molmo import MolmoForCausalLM
from models.molmo.preprocessing_molmo import MolmoProcessor
from models.transparent_llm import TransparentLlm, ModelInfo
from utils.hooks import InputsHook, OutputsHook

@torch.compile()
def get_decomposed_attention(v, pattern, o_weight):
    return torch.einsum("khd,hqk,hdm->qkhm", v, pattern, o_weight.to(dtype=v.dtype, device=v.device))

class TransparentLlava(TransparentLlm):


    image_token_id = 32000

    def __init__(self, name, llava, processor, device, dtype=torch.bfloat16, store_on_cpu=False):
        super().__init__()
        self.name = name
        self.wrapped_model = llava
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.store_on_cpu = store_on_cpu
        self.n_layers = self.wrapped_model.language_model.config.num_hidden_layers
        self.n_heads = self.wrapped_model.language_model.config.num_attention_heads
        self.d_model = self.wrapped_model.language_model.config.hidden_size
        self.processor.tokenizer.padding_side = 'left'

        llm = self.wrapped_model.language_model
        self.layers = self.wrapped_model.language_model.model.layers
        self.config = llm.config

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
        self.last_run_logits_shape = None
        self.last_run_attentions = None

    def underlying_model(self) -> nn.Module:
        return self.wrapped_model

    def clear_state(self):
        del self.last_run_inputs
        self.last_run_inputs = None
        del self.last_run_logits_shape
        self.last_run_logits_shape = None
        del self.last_run_attentions
        self.last_run_attentions = None

        for hook_collection in [self.res_in_hooks, self.res_mid_hooks, self.ffn_out_hooks, self.value_out_hooks]:
            for hook in hook_collection:
                del hook.value
                hook.value = None

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_vocab=self.wrapped_model.language_model.config.vocab_size,
        )

    def forward(self, **inputs) -> Any:
        self.clear_state()
        results = self.wrapped_model(**inputs)

        self.last_run_inputs = inputs
        self.last_run_logits_shape = results.logits.shape
        self.last_run_attentions = [att.cpu() for att in results.attentions] if self.store_on_cpu else results.attentions
        return results

    def generate(self, **inputs):
        return self.wrapped_model.generate(**inputs)

    def batch_size(self) -> int:
        return self.last_run_logits_shape[0]

    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        input_ids = self.last_run_inputs['input_ids']
        res = torch.zeros(self.last_run_logits_shape[:2], dtype=torch.int64, device=self.device)
        res.fill_(TransparentLlava.image_token_id)
        img_begin = (input_ids == TransparentLlava.image_token_id).nonzero()[:1]
        for i, begin in img_begin:
            res[i, :begin] = input_ids[i, :begin]
            res[i, begin + 576:] = input_ids[i, begin + 576:]
        return res

    def image_token_pos(self, batch_i: int) -> (int, int):
        input_ids = self.last_run_inputs['input_ids']
        img_begin = (input_ids[batch_i] == TransparentLlava.image_token_id).nonzero()[0]
        # print(img_begin[0], img_begin)
        return img_begin.item(), img_begin.item() + 576

    @staticmethod
    def tokens_to_strings(tokens: Int[torch.Tensor, "pos"], tokenizer) -> List[str]:
        res = []
        for tok in tokens:
            if tok == TransparentLlava.image_token_id:
                for i in range(576):
                    res.append(f"I_{i}")
            else:
                res.append(tokenizer.decode(tok))
        return res

    def logit_lens(self, after_layer: int, token: int, normalize: bool) -> Float[torch.Tensor, "batch vocab"]:
        res_after_layer = self.residual_out(after_layer)
        if normalize:
            return self.wrapped_model.language_model.lm_head(
                self.wrapped_model.language_model.model.norm(res_after_layer)[:, token, :])
        else:
            return self.wrapped_model.language_model.lm_head(res_after_layer[:, token, :])

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
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.n_heads, self.d_model // self.n_heads)[
            batch_i, :, head_i]

        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i, head_i].to(dtype=v.dtype, device=v.device)
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
    ) -> Float[torch.Tensor, "source target slice d_model"]:
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.n_heads, self.d_model // self.n_heads)[
            batch_i, :, head_start:head_end]

        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i, head_start:head_end].to(dtype=v.dtype, device=v.device)
        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )

        o_weight = self.layers[layer].self_attn.o_proj.weight.T.view(self.n_heads,
                                                                     self.d_model // self.n_heads, self.d_model)[
                   head_start:head_end]
        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z.to(dtype=o_weight.dtype, device=o_weight.device),
            o_weight
        )
        return decomposed_attn

    @torch.no_grad()
    def decomposed_attn_components(self, batch_i: int, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.n_heads, self.d_model // self.n_heads)[
            batch_i]

        pattern = self.last_run_attentions[layer].to(dtype=v.dtype, device=v.device)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i]
        o_weight = self.layers[layer].self_attn.o_proj.weight.T.view(self.n_heads, self.config.head_dim, self.d_model)
        return v, pattern, o_weight

    @torch.no_grad()
    def decomposed_attn(self, batch_i: int, layer: int) -> Float[torch.Tensor, "source target head d_model"]:
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.n_heads, self.d_model // self.n_heads)[
            batch_i]
        # # support for gqa
        # num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
        # hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)
        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i].to(dtype=v.dtype, device=v.device)
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
                                                              self.d_model).to(device=v.device),
        )
        return decomposed_attn


class TransparentPixtral(TransparentLlava):
    image_token_id = 10
    image_break_token_id = 12
    image_end_token_id = 13

    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        return self.last_run_inputs['input_ids']

    def image_token_pos(self, batch_i: int) -> (int, int):
        """
        WARNING: This (and the whole repository as of now) supports only a setting where
        all image tokens are at a single point in the input. Interleaved image-text is not currently supported.
        """
        input_ids = self.last_run_inputs['input_ids']
        img_begin = (input_ids[batch_i] == self.image_token_id).nonzero().min()
        img_end = (input_ids[batch_i] == self.image_end_token_id).nonzero().max()
        return img_begin.item(), img_end.item()

    @staticmethod
    def tokens_to_strings(tokens: Int[torch.Tensor, "pos"], tokenizer) -> List[str]:
        res = []
        img_count = 0
        img_break_count = 0
        img_end_count = 0
        for tok in tokens:
            match tok:
                case TransparentPixtral.image_token_id:
                    res.append(f"I_{img_count}")
                    img_count += 1
                case TransparentPixtral.image_break_token_id:
                    res.append(f"I_{img_break_count}_BREAK")
                    img_break_count += 1
                case TransparentPixtral.image_end_token_id:
                    res.append(f"I_{img_end_count}_END")
                    img_end_count += 1
                case _:
                    res.append(tokenizer.decode(tok))
        return res


    @torch.no_grad()
    def decomposed_attn_head_slice(
            self, head_start: int, head_end: int, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "source target slice d_model"]:
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.config.num_key_value_heads,
                                                   self.config.head_dim)

        v = v.repeat_interleave(self.n_heads // self.config.num_key_value_heads, dim=2, output_size=self.n_heads)
        v = v[batch_i, :, head_start:head_end]

        pattern = self.last_run_attentions[layer].to(dtype=v.dtype, device=v.device)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i, head_start:head_end]
        o_weight = self.layers[layer].self_attn.o_proj.weight.T.view(self.n_heads, self.config.head_dim, self.d_model)[
                   head_start:head_end]

        return get_decomposed_attention(v, pattern, o_weight)

    @torch.no_grad()
    def decomposed_attn_components(self, batch_i: int, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        v = self.value_out_hooks[layer].value.view(batch_size, num_tokens, self.config.num_key_value_heads,
                                                   self.config.head_dim)

        v = v.repeat_interleave(self.n_heads // self.config.num_key_value_heads, dim=2, output_size=self.n_heads)
        v = v[batch_i]

        pattern = self.last_run_attentions[layer].to(dtype=v.dtype, device=v.device)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i]
        o_weight = self.layers[layer].self_attn.o_proj.weight.T.view(self.n_heads, self.config.head_dim, self.d_model)
        return v, pattern, o_weight


class TransparentMolmo(TransparentLlm):
    image_start_token_id = 152064
    image_end_token_id = 152065
    image_patch_token_id = 152066
    image_col_token_id = 152067
    image_prompt_token_id = 152068  # Unsure if used

    def __init__(self, name, molmo: MolmoForCausalLM, processor: MolmoProcessor, device, dtype=torch.bfloat16):
        super().__init__()
        self.name = name
        self.molmo = molmo
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.config = molmo.model.config
        self.n_layers = self.config.n_layers
        self.n_heads = self.config.n_heads
        self.d_model = self.config.d_model
        self.processor.tokenizer.padding_side = 'left'

        llm = self.molmo.model.transformer
        self.layers = llm.blocks

        self.res_in_hooks = [InputsHook(0) for _ in range(self.n_layers)] + [InputsHook(0)]
        self.res_in_hook_objs = [self.layers[i].register_forward_pre_hook(hook, with_kwargs=True) for i, hook in
                                 enumerate(self.res_in_hooks[:-1])] + [
                                    llm.ln_f.register_forward_pre_hook(self.res_in_hooks[-1], with_kwargs=True)]

        self.res_mid_hooks = [InputsHook(0) for _ in range(self.n_layers)]
        self.res_mid_hook_objs = [
            self.layers[i].attn_norm.register_forward_pre_hook(hook, with_kwargs=True) for i, hook
            in enumerate(self.res_mid_hooks)]

        self.ffn_out_hooks = [OutputsHook(0) for _ in range(self.n_layers)]
        self.ffn_out_hook_objs = [self.layers[i].ff_out.register_forward_hook(hook) for i, hook in
                                  enumerate(self.ffn_out_hooks)]

        self.qkv_out_hooks = [OutputsHook(0) for _ in range(self.n_layers)]
        self.qkv_out_hooks_objs = [self.layers[i].att_proj.register_forward_hook(hook) for i, hook in
                                   enumerate(self.qkv_out_hooks)]

        self.last_run_inputs = None
        self.last_run_logits_shape = None
        self.last_run_attentions = None

    def clear_state(self):
        del self.last_run_inputs
        self.last_run_inputs = None
        del self.last_run_logits_shape
        self.last_run_logits_shape = None
        del self.last_run_attentions
        self.last_run_attentions = None

        for hook_collection in [self.res_in_hooks, self.res_mid_hooks, self.ffn_out_hooks, self.qkv_out_hooks]:
            for hook in hook_collection:
                del hook.value
                hook.value = None

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_vocab=self.config.vocab_size,
        )

    def forward(self, **inputs):
        self.clear_state()
        results = self.molmo(**inputs)

        self.last_run_inputs = inputs
        self.last_run_logits_shape = results.logits.shape
        self.last_run_attentions = results.attentions
        return results

    def generate(self, **inputs):
        return self.molmo.generate_from_batch(
            {"input_ids": inputs["input_ids"], "images": inputs["images"], "image_input_idx": inputs["image_input_idx"],
             "image_masks": inputs["image_masks"]},
            inputs["generation_config"],
            tokenizer=self.processor.tokenizer
        )

    def batch_size(self) -> int:
        return self.last_run_logits_shape[0]

    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        return self.last_run_inputs['input_ids']

    def image_token_pos(self, batch_i: int) -> (int, int):
        input_ids = self.last_run_inputs['input_ids']
        img_begin = (input_ids[batch_i] == self.image_start_token_id).nonzero()
        if img_begin.numel() == 0:
            return -1, -1
        img_begin = img_begin.min()
        img_end = (input_ids[batch_i] == self.image_end_token_id).nonzero().max()
        return img_begin.item(), img_end.item()

    @staticmethod
    def tokens_to_strings(tokens: Int[torch.Tensor, "pos"], tokenizer) -> List[str]:
        res = []
        img_patch_count = 0
        img_block_count = 0
        img_col_count = 0
        for tok in tokens:
            match tok:
                case TransparentMolmo.image_start_token_id:
                    res.append(f"I_{img_block_count}_START")
                case TransparentMolmo.image_end_token_id:
                    res.append(f"I_{img_block_count}_END")
                    img_block_count += 1
                    img_patch_count = 0
                    img_col_count = 0
                case TransparentMolmo.image_patch_token_id:
                    res.append(f"I_{img_block_count}_{img_patch_count}")
                    img_patch_count += 1
                case TransparentMolmo.image_col_token_id:
                    res.append(f"I_{img_block_count}_COL_{img_col_count}")
                    img_col_count += 1
                case _:
                    res.append(tokenizer.decode(tok))
        return res

    def logit_lens(self, after_layer: int, token: int, normalize: bool) -> Float[torch.Tensor, "batch vocab"]:
        res_after_layer = self.residual_out(after_layer)
        if normalize:
            return self.molmo.model.transformer.ff_out(
                self.llava.language_model.model.norm(res_after_layer)[:, token, :])
        else:
            return self.molmo.model.transformer.ff_out(res_after_layer[:, token, :])

    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        raise NotImplementedError

    def unembed(self, t: Float[torch.Tensor, "d_model"], normalize: bool) -> Float[torch.Tensor, "vocab"]:
        raise NotImplementedError

    def underlying_model(self) -> nn.Module:
        return self.molmo

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

    def decomposed_attn_head(self, head_i: int, batch_i: int, layer: int) -> Float[
        torch.Tensor, "source target d_model"]:
        raise NotImplementedError

    def decomposed_attn_head_slice(self, head_start: int, head_end: int, batch_i: int, layer: int) -> Float[
        torch.Tensor, "source target slice d_model"]:
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        qkv = self.qkv_out_hooks[layer].value

        _, _, v = qkv.split(self.molmo.model.transformer.blocks[layer].fused_dims, dim=-1)
        v = v.view(batch_size, num_tokens, self.config.effective_n_kv_heads, self.config.d_model // self.config.n_heads)
        v = v.repeat_interleave(self.config.n_heads // self.config.effective_n_kv_heads, dim=2,
                                output_size=self.config.n_heads)
        v = v[batch_i, :, head_start:head_end]

        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i, head_start:head_end].to(dtype=v.dtype, device=v.device)
        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "  
            "query_pos key_pos head d_head",
            v,
            pattern,
        )

        o_weight = self.layers[layer].attn_out.weight.T.view(self.n_heads, self.d_model // self.n_heads,
                                                             self.d_model)[head_start:head_end]
        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z.to(dtype=o_weight.dtype, device=o_weight.device),
            o_weight
        )
        return decomposed_attn

    @torch.no_grad()
    def decomposed_attn_components(self, batch_i: int, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens = self.last_run_logits_shape[:2]
        qkv = self.qkv_out_hooks[layer].value

        _, _, v = qkv.split(self.molmo.model.transformer.blocks[layer].fused_dims, dim=-1)
        v = v.view(batch_size, num_tokens, self.config.effective_n_kv_heads, self.config.d_model // self.config.n_heads)
        v = v.repeat_interleave(self.config.n_heads // self.config.effective_n_kv_heads, dim=2,
                                output_size=self.config.n_heads)
        v = v[batch_i]

        pattern = self.last_run_attentions[layer]
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = pattern[batch_i].to(dtype=v.dtype, device=v.device)
        o_weight = self.layers[layer].attn_out.weight.T.view(self.n_heads, self.d_model // self.n_heads, self.d_model)
        return v, pattern, o_weight

    def decomposed_attn(self, batch_i: int, layer: int) -> Float[torch.Tensor, "source target head d_model"]:
        raise NotImplementedError
