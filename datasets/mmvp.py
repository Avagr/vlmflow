import csv
from pathlib import Path
import re

from PIL import Image
import torch
import wandb

from datasets.base import BaseDataset
from models.wrappers import GenerativeWrapper
from utils.eval import EvaluationResult


class MMVP(BaseDataset):
    table_columns = ["image", "question", "options", "prediction", "answer"]

    # noinspection PyTypeChecker
    def __init__(self, csv_path: Path, img_dir: Path):
        super().__init__()
        self.img_dir = img_dir
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        self.data = [
            {'id': row['Index'],
             'question': row['Question'],
             'options': tuple(map(str.strip, re.split(r'\([a-z]\)', row['Options'])[1:])),
             'answer': 0 if row['Correct Answer'] == '(a)' else 1} for row in rows]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.img_dir / f"{self.data[idx]['id']}.jpg").convert('RGB')
        question = self.data[idx]['question']
        option1, option2 = self.data[idx]['options']
        answer = self.data[idx]['answer']
        return idx, image, question, (option1, option2), answer

    def table_repr(self, idx, prediction, img_as_object=True):
        img_path = self.img_dir / f"{self.data[idx]['id']}.jpg"
        image = wandb.Image(str(img_path.resolve())) if img_as_object else f"{self.data[idx]['id']}.jpg"
        return [
            image,
            self.data[idx]['question'],
            "\n".join(f"{c}: {ch}" for c, ch in zip("AB", self.data[idx]['options'])),
            str(prediction),
            self.data[idx]['answer']
        ]


class MMVPCollate:

    def __call__(self, batch):
        idx = []
        images = []
        questions = []
        options = []
        answers = []
        for i, image, question, option, answer in batch:
            idx.append(i)
            images.append(image)
            questions.append(question)
            options.append(option)
            answers.append(answer)
        return torch.LongTensor(idx), images, questions, options, torch.LongTensor(answers)


class MMVPEval:

    def __init__(self, prompt, eval_method="abcd"):
        """
        Supports the following scoring methods:

            * "ppl": take the answer with the lowest loss in a <question: q, answer: answer> setting
            * "abcd": take the answer with the highest score in a <question: q, choices: [a, b, c, d], answer:> setting

        :param prompt: prompt format: arbitrary string with <Q> and <C> being placeholders for the question and the choices
        :param eval_method: method to use for scoring:
        """
        split_prompt = re.split(r'(<[QC]>)', prompt)
        assert len(split_prompt) == 5, f"Prompt should have exactly 2 placeholders, got {len(split_prompt)}"
        self.pre_prompt = split_prompt[0]
        self.mid_prompt = split_prompt[2]
        self.post_prompt = split_prompt[4]
        self.eval_method = eval_method
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, batch, model: GenerativeWrapper) -> EvaluationResult:
        idx, images, questions, options, answers = batch
        # A workaround for a long-standing pytorch bug with pin_memory (https://github.com/pytorch/pytorch/issues/48419)
        batch_size = idx.shape[0]
        match self.eval_method:
            case "abcd":
                texts = [(f"{self.pre_prompt}{q}{self.mid_prompt}"
                          f"A. {opt1}\nB. {opt2}\n"
                          f"{self.post_prompt}") for q, (opt1, opt2) in zip(questions, options)]
                scores, generated_ids, num_generated_tokens = model.score_single_tokens(images, texts, model.vqa_candidates[:2])
                predictions = scores.cpu().argmax(-1)
            case _:
                raise ValueError(f"Unsupported evaluation method {self.eval_method}")

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(model, idx, texts, answers, predictions, scores)

        return EvaluationResult(batch_size, idx.tolist(), texts, predictions, answers, generated_ids,
                                num_generated_tokens)
