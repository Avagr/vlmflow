import torch
from torch import nn
from transformers import GenerationConfig

from models.transparent_llm import TransparentLlm


class GenerativeWrapper(nn.Module):
    """
    Wrapper for tasks that require generating text
    """

    def __init__(self, processor, model: TransparentLlm, device, dtype, output_attentions=True):
        super().__init__()
        self.processor = processor
        processor.tokenizer.padding_side = 'left'
        self.model = model
        self.device = device
        self.dtype = dtype
        self.ignore_index = -100
        self.text_score = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.image_token_index = 32000
        self.image_pad_index = 32001
        self.num_image_patches = 576
        self.output_attentions = output_attentions

    def generate(self, images: list[torch.Tensor], texts: list[str], config: GenerationConfig) -> list[str]:
        inputs = self.processor(text=texts, images=images, return_tensors='pt', padding=True).to(device=self.device,
                                                                                                 dtype=self.dtype)
        generated_ids = self.model.underlying_model().generate(**inputs, generation_config=config)

        # One more forward to save the activations in the transparent model
        padded_mask = torch.ones_like(generated_ids[:, :-1])
        padded_mask[:, :inputs.attention_mask.shape[1]] = inputs.attention_mask
        self.model(input_ids=generated_ids[:, :-1], pixel_values=inputs.pixel_values, attention_mask=padded_mask,
                   output_attentions=self.output_attentions, use_cache=False)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def forward(self, images: list[torch.Tensor], texts: list[str], **kwargs) -> torch.Tensor:
        inputs = self.processor(text=texts, images=images, return_tensors='pt', padding=True).to(device=self.device,
                                                                                                 dtype=self.dtype)
        return self.model(**inputs, output_attentions=self.output_attentions, **kwargs, use_cache=False)

    def score_single_tokens(self, images: list[torch.Tensor], texts: list[str],
                            candidates: list[str]) -> (torch.Tensor, dict[str, torch.Tensor]):
        labels = self.processor.tokenizer(candidates, add_special_tokens=False, return_tensors='pt',
                                          padding=False).input_ids.view(-1).to(device=self.device)
        if labels.shape[0] != len(candidates):
            raise ValueError(f"Each candidate should be a single token, {candidates} do not fit this criteria")
        inputs = self.processor(text=texts, images=images, return_tensors='pt', truncation=False, padding=True).to(
            device=self.device,
            dtype=self.dtype)
        logits = torch.softmax(self.model(**inputs, output_attentions=self.output_attentions, use_cache=False).logits,
                               dim=-1)
        return logits[:, -1, labels], {'max_tokens': self.processor.tokenizer.batch_decode(logits[:, -1].argmax(-1))}

    def score_text(self, images: list[torch.Tensor], texts: list[str]) -> (torch.Tensor, dict[str, torch.Tensor]):
        inputs = self.processor(text=texts, images=images, return_tensors='pt', truncation=False, padding=True).to(
            device=self.device,
            dtype=self.dtype)

        outputs = self.model(**inputs, output_attentions=self.output_attentions, use_cache=False)

        # Compute the true_labels taking into account the image tokens (all credit to the HuggingFace LLaVA implementation)
        input_ids, attention_mask, logits = inputs.input_ids, inputs.attention_mask, outputs.logits
        batch_size, num_tokens, vocab_size = logits.shape
        # noinspection PyTypeChecker
        image_token_mask = input_ids == self.image_token_index
        text_batch_idx, text_token_idx = torch.where(
            (~image_token_mask) & (input_ids != self.image_pad_index)
        )
        labels = torch.full((batch_size, num_tokens), self.ignore_index, dtype=input_ids.dtype, device=self.device)
        new_token_positions = torch.cumsum((image_token_mask * (self.num_image_patches - 1) + 1), -1) - 1
        text_to_overwrite = new_token_positions[text_batch_idx, text_token_idx]
        labels[text_batch_idx, text_to_overwrite] = input_ids[text_batch_idx, text_token_idx]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().view(-1)

        losses = self.text_score(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels).view(batch_size, -1)

        return losses.sum(-1) / (shift_labels != -100).sum(-1)
