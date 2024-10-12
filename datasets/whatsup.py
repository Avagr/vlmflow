import json
from pathlib import Path
import re

from PIL import Image
import torch
import wandb

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
        if isinstance(prediction, str):
            return [
                idx,
                wandb.Image(str((self.root_dir / self.items[idx]['image_path']).resolve())) if img_as_object else
                self.items[idx]['image_path'],
                prediction,
                None,
            ]
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
    def __init__(self, prompt, eval_method="abcd", generation_config=None):
        if eval_method == "abcd":
            split_prompt = re.split(r'(<C>)', prompt)
            assert len(split_prompt) == 3, f"Prompt should have exactly one placeholder <C>"
            self.pre_prompt = split_prompt[0]
            self.post_prompt = split_prompt[2]
        elif eval_method == "gen":
            self.prompt = prompt
            assert generation_config is not None, "Generation config must be provided for 'gen' eval method"
            self.generation_config = generation_config
        self.eval_method = eval_method
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, batch, model: GenerativeWrapper) -> EvaluationResult:
        idx, images, options, answers = batch
        batch_size = idx.shape[0]
        match self.eval_method:
            # case "ppl":
            #     texts = [f"{self.pre_prompt}{opt[i]}{self.post_prompt}" for opt in options for i in range(4)]
            #     images = [img for img in images for _ in range(4)]
            #     scores, metrics = model.score_text(images, texts).view(batch_size, 4)
            #     predictions = scores.argmin(-1)

            case "abcd":
                texts = [(f"{self.pre_prompt}"
                          f"A. {opt[0]}\nB. {opt[1]}\nC. {opt[2]}\nD. {opt[3]}\n"
                          f"{self.post_prompt}") for opt in options]
                scores, generated_ids, num_generated_tokens = model.score_single_tokens(images, texts,
                                                                                        model.vqa_candidates)
                predictions = scores.argmax(-1).cpu()
            case "gen":
                texts = [self.prompt] * batch_size
                generated_text, generated_ids, num_generated_tokens = model.generate(images, texts,
                                                                                     self.generation_config)
                predictions = answers  # Will always result in accuracy 1
                scores = None
            case _:
                raise ValueError(f"Unsupported evaluation method {self.eval_method}")
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(model, idx, texts, answers, predictions, scores)

        return EvaluationResult(batch_size, idx.tolist(), texts, predictions, answers, generated_ids,
                                num_generated_tokens)
