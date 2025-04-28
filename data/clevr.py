import json
from pathlib import Path
import re

from PIL import Image
import torch
from transformers import GenerationConfig
import wandb

from data.base import BaseDataset, EvalWrapper
from models.wrappers import GenerativeWrapper
from utils.eval import EvaluationResult


class ClevrDataset(BaseDataset):
    table_columns = ["idx", "image", "question", "family_index", "prediction", "answer"]

    def __init__(self, json_path: Path, img_dir: Path, question_family_indices: list[int] = None):
        data = json.load(open(json_path))
        self.questions: list[dict[str, str | int]] = data['questions']
        self.img_dir = img_dir
        if question_family_indices is not None:
            self.questions = [q for q in self.questions if q['question_family_index'] in question_family_indices]
        else: # All yes/no questions
            self.questions = [q for q in self.questions if q['question_family_index'] in list(range(1, 25)) + [36, 37, 38, 39] + [44, 45, 46, 47] + [73, 79, 85]]
            self.questions = self.questions[::10]


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]['question']
        answer = self.questions[idx]['answer']
        img = Image.open(self.img_dir / self.questions[idx]['image_filename'])
        return idx, img, question, answer

    def table_repr(self, idx, prediction, img_as_object=True):
        img_path = self.img_dir / self.questions[idx]['image_filename']
        image = wandb.Image(str(img_path.resolve())) if img_as_object else self.questions[idx]['image_filename']
        return [
            idx,
            image,
            self.questions[idx]['question'],
            self.questions[idx]['question_family_index'],
            prediction,
            self.questions[idx]['answer']
        ]



class ClevrCollate:
    def __call__(self, batch):
        idx = []
        images = []
        questions = []
        answers = []
        for i, img, q, a in batch:
            idx.append(i)
            images.append(img)
            questions.append(q)
            answers.append(a)
        return torch.LongTensor(idx), images, questions, answers


class ClevrEval(EvalWrapper):
    def __init__(self, prompt: str, eval_method: str = 'next_token'):
        split_prompt = re.split(r'(<Q>)', prompt)
        assert len(split_prompt) == 3, "Prompt must contain exactly one '<Q>'"
        self.prefix = split_prompt[0]
        self.suffix = split_prompt[2]
        self.eval_method = eval_method
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, batch, model: GenerativeWrapper) -> 'EvaluationResult':
        idx, images, questions, answers = batch
        batch_size = idx.shape[0]

        match self.eval_method:
            case "next_token":
                texts = [f"{self.prefix}{q}{self.suffix}" for q in questions]
                _, generated_ids, num_generated_tokens = model.generate(texts=texts, images=images,
                                                                                  config=GenerationConfig(
                                                                                      max_new_tokens=1)
                                                                                  )
                # print(model.processor.tokenizer.decode(generated_ids[0]))
                predictions = [model.processor.tokenizer.decode(gen_ids[-1]).strip().lower() for gen_ids in generated_ids]
            case _:
                raise ValueError(f"Unsupported evaluation method {self.eval_method}")

        for callback in self.callbacks:
            callback(model, idx)
        return EvaluationResult(batch_size, idx.tolist(), texts, predictions, answers, generated_ids,
                                num_generated_tokens)
