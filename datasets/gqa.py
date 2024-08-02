import json
import re
from pathlib import Path

import wandb
from PIL import Image

from datasets.base import BaseDataset, EvalWrapper
from models.wrappers import GenerativeWrapper
from utils.eval import EvaluationResult


class GQA(BaseDataset):
    table_columns = ["image", "question", "prediction", "answer"]

    def __init__(self, json_path: Path, img_dir: Path, dataset_size: int = None):
        super().__init__()
        self.img_dir = img_dir
        with open(json_path, 'r') as f:
            self.data = list(json.load(f).items())
        if dataset_size is not None:
            self.data = self.data[:dataset_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, Image, str, str]:
        q_id, q_data = self.data[idx]
        img_path = self.img_dir / f"{q_data['imageId']}.jpg"
        image = Image.open(img_path).convert('RGB')
        question = q_data['question']
        answer = q_data['answer']
        return q_id, image, question, answer

    def table_repr(self, idx, prediction):
        q_id, q_data = self.data[idx]
        img_path = self.img_dir / f"{q_data['imageId']}.jpg"
        return [
            wandb.Image(str(img_path.resolve())),
            q_data['question'],
            prediction,
            q_data['answer']
        ]


class GQACollate:
    def __call__(self, batch):
        idx, images, questions, answers = zip(*batch)
        return idx, images, questions, answers


class GQAEval(EvalWrapper):

    def __init__(self, prompt, device, eval_method="gen", generation_config=None):
        split_prompt = re.split(r'(<Q>)', prompt)
        assert len(split_prompt) == 3, "Prompt must contain exactly one <Q> tag"
        self.pre_prompt = split_prompt[0]
        self.post_prompt = split_prompt[2]
        self.device = device
        if eval_method == "gen":
            assert generation_config is not None, "Generation config must be provided for 'gen' eval method"
        self.eval_method = eval_method
        self.generation_config = generation_config

    def __call__(self, batch, model: GenerativeWrapper) -> EvaluationResult:
        idx, images, questions, answers = batch
        batch_size = len(idx)
        match self.eval_method:
            case "gen":
                predictions = model.generate(images, [f"{self.pre_prompt}{q}{self.post_prompt}" for q in questions],
                                             self.generation_config)
                print(predictions)
            case _:
                raise ValueError(f"Unsupported eval method '{self.eval_method}'")
        return EvaluationResult(batch_size, idx, predictions, answers)


class GQATrain:
    pass
