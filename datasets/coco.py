import json
from pathlib import Path

from PIL import Image
import torch
import wandb

from datasets.base import BaseDataset, EvalWrapper
from models.wrappers import GenerativeWrapper
from utils.eval import EvaluationResult


class UnlabeledCoco(BaseDataset):
    table_columns = ['idx', 'image', 'generated_caption']

    def __init__(self, img_dir: Path, img_descriptions_file: Path, dataset_size: int = None):
        super().__init__()
        self.img_dir = img_dir
        with open(img_descriptions_file, 'r') as f:
            self.data = json.load(f)['images']
        if dataset_size is not None:
            self.data = self.data[:dataset_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> (int, Image):
        img_data = self.data[idx]
        img_path = self.img_dir / img_data['file_name']
        image = Image.open(img_path).convert('RGB')
        return idx, image

    def table_repr(self, idx, generated_caption, img_as_object=True):
        img_data = self.data[idx]
        img_path = self.img_dir / img_data['file_name']
        return [
            idx,
            wandb.Image(str(img_path.resolve())) if img_as_object else img_data['file_name'],
            generated_caption
        ]


class UnlabeledCocoCollate:
    def __call__(self, batch):
        idx = []
        images = []
        for i, image in batch:
            idx.append(i)
            images.append(image)
        return torch.LongTensor(idx), images


class UnlabeledCocoEval(EvalWrapper):

    def __init__(self, prompt, generation_config):
        self.prompt = prompt
        self.generation_config = generation_config
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, batch, model: GenerativeWrapper) -> 'EvaluationResult':
        idx, images = batch
        batch_size = idx.shape[0]
        texts = [self.prompt] * batch_size
        generated_text, generated_ids, num_generated_tokens = model.generate(images, texts, self.generation_config)
        for callback in self.callbacks:
            callback(model, idx, generated_text)
        return EvaluationResult(batch_size, idx.tolist(), texts, generated_text, generated_text, generated_ids,
                                num_generated_tokens)
