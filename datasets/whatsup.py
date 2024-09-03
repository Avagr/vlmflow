import json
import re
from pathlib import Path

import torch
import wandb
from PIL import Image

from datasets.base import BaseDataset
from models.wrappers import GenerativeWrapper
from utils.eval import EvaluationResult


class WhatsUp(BaseDataset):
    table_columns = ["idx", "image", "prediction", "answer"]

    def __init__(self, root_dir: Path, json_path: Path, permute_options: bool = False):
        self.root_dir = root_dir
        self.permute_options = permute_options
        with open(json_path, 'r') as f:
            self.items = json.load(f)
        if self.permute_options:
            self.permutations = [torch.randperm(4) for _ in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(self.root_dir / item['image_path']).convert('RGB')
        options = item['caption_options']
        answer = 0
        if self.permute_options:
            permutation = self.permutations[idx]
            answer = permutation.argmin().item()
            options = [options[i] for i in permutation]
        return idx, image, options, answer

    def table_repr(self, idx, prediction, img_as_object=True):
        if self.permute_options and prediction != -1:
            prediction = self.permutations[idx][prediction].item()
        item = self.items[idx]
        return [
            idx,
            wandb.Image(str((self.root_dir / item['image_path']).resolve())) if img_as_object else item['image_path'],
            item['caption_options'][prediction] if prediction != -1 else "N/A",
            item['caption_options'][0],
        ]


class WhatsUpCollate:
    def __call__(self, batch):
        idx = []
        images = []
        options = []
        answers = []
        for i, image, option, answer in batch:
            idx.append(i)
            images.append(image)
            options.append(option)
            answers.append(answer)
        return torch.LongTensor(idx), images, options, torch.LongTensor(answers)


class WhatsUpEval:
    def __init__(self, prompt, device, eval_method="ppl", scores_hooks=None):
        split_prompt = re.split(r'(<C>)', prompt)
        assert len(split_prompt) == 3, f"Prompt should have exactly one placeholder <C>"
        self.pre_prompt = split_prompt[0]
        self.post_prompt = split_prompt[2]
        self.device = device
        self.eval_method = eval_method
        self.callbacks = scores_hooks
        if self.callbacks is None:
            self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, batch, model: GenerativeWrapper) -> EvaluationResult:
        idx, images, options, answers = batch
        batch_size = idx.shape[0]
        match self.eval_method:
            case "ppl":
                texts = [f"{self.pre_prompt}{opt[i]}{self.post_prompt}" for opt in options for i in range(4)]
                images = [img for img in images for _ in range(4)]
                scores, metrics = model.score_text(images, texts).view(batch_size, 4)
                predictions = scores.argmin(-1)

            case "abcd":
                texts = [(f"{self.pre_prompt}"
                          f"A. {opt[0]}\nB. {opt[1]}\nC. {opt[2]}\nD. {opt[3]}\n"
                          f"{self.post_prompt}") for opt in options]
                scores, metrics = model.score_single_tokens(images, texts, ['A', 'B', 'C', 'D'])
                predictions = scores.argmax(-1)

            case "abcd_top":
                texts = [(f"{self.pre_prompt}"
                          f"A. {opt[0]}\nB. {opt[1]}\nC. {opt[2]}\nD. {opt[3]}\n"
                          f"{self.post_prompt}") for opt in options]
                scores, metrics = model.score_single_tokens(images, texts, ['A', 'B', 'C', 'D'])
                predictions = scores.argmax(-1)
                for i, tok in enumerate(metrics['max_tokens']):
                    if tok not in ['A', 'B', 'C', 'D']:
                        predictions[i] = -1
            case _:
                raise ValueError(f"Unsupported evaluation method {self.eval_method}")
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(model, idx, texts, answers, predictions, scores)

        return EvaluationResult(batch_size, idx.tolist(), texts, predictions.cpu(), answers, metrics)