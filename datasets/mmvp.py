import csv
import re
from pathlib import Path

import torch
import wandb
from PIL import Image

from datasets.base import BaseDataset
from models.wrappers import GenerativeWrapper
from utils.eval import EvaluationResult


class MMVP(BaseDataset):
    table_columns = ["image1", "image2", "question", "options", "prediction", "answer"]

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
        return len(self.data) // 2

    def __getitem__(self, idx):
        image1 = Image.open(self.img_dir / f"{self.data[2 * idx]['id']}.jpg").convert('RGB')
        image2 = Image.open(self.img_dir / f"{self.data[2 * idx + 1]['id']}.jpg").convert('RGB')
        question = self.data[2 * idx]['question']
        option1, option2 = self.data[2 * idx]['options']
        answer1 = self.data[2 * idx]['answer']
        answer2 = self.data[2 * idx + 1]['answer']
        return idx, (image1, image2), question, (option1, option2), (answer1, answer2)

    def table_repr(self, idx, prediction, img_as_object=True):
        image1 = wandb.Image(str((self.img_dir / f"{self.data[2 * idx]['id']}.jpg").resolve()))
        image2 = wandb.Image(str((self.img_dir / f"{self.data[2 * idx + 1]['id']}.jpg").resolve()))
        return [
            image1,
            image2,
            self.data[2 * idx]['question'],
            "\n".join(f"{c}: {ch}" for c, ch in zip("01", self.data[2 * idx]['options'])),
            str(prediction),
            f"({self.data[2 * idx]['answer']}, {self.data[2 * idx + 1]['answer']})"
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
        return torch.LongTensor(idx), images, questions, options, answers


class MMVPEval:

    def __init__(self, prompt, eval_method="ppl"):
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

    def __call__(self, batch, model: GenerativeWrapper) -> EvaluationResult:
        idx, images, questions, options, answers = batch
        # A workaround for a long-standing pytorch bug with pin_memory (https://github.com/pytorch/pytorch/issues/48419)
        answers = [(a1, a2) for (a1, a2) in answers]
        batch_size = idx.shape[0]
        images_1 = [img1 for (img1, _) in images]
        images_2 = [img2 for (_, img2) in images]
        match self.eval_method:
            case "ppl":
                texts_1 = [f"{self.pre_prompt}{q}{self.mid_prompt}{opt1}{self.post_prompt}" for q, (opt1, _) in
                           zip(questions, options)]
                texts_2 = [f"{self.pre_prompt}{q}{self.mid_prompt}{opt2}{self.post_prompt}" for q, (_, opt2) in
                           zip(questions, options)]
                scores_11 = model.score_text(images_1, texts_1).cpu()
                scores_12 = model.score_text(images_1, texts_2).cpu()
                scores_21 = model.score_text(images_2, texts_1).cpu()
                scores_22 = model.score_text(images_2, texts_2).cpu()
                predictions_1 = (scores_11 > scores_12).int().tolist()
                predictions_2 = (scores_21 > scores_22).int().tolist()
                predictions = [(p1, p2) for p1, p2 in zip(predictions_1, predictions_2)]

            case "abcd":
                texts = [(f"{self.pre_prompt}{q}{self.mid_prompt}"
                          f"A. {opt1}\nB. {opt2}\n"
                          f"{self.post_prompt}") for q, (opt1, opt2) in zip(questions, options)]
                scores_1 = model.score_single_tokens(images_1, texts, ['A', 'B'])
                predictions_1 = scores_1.cpu().argmax(-1)
                scores_2 = model.score_single_tokens(images_2, texts, ['A', 'B'])
                predictions_2 = scores_2.cpu().argmax(-1)
                predictions = [(p1.item(), p2.item()) for p1, p2 in zip(predictions_1, predictions_2)]
            case _:
                raise ValueError(f"Unsupported evaluation method {self.eval_method}")
        return EvaluationResult(batch_size, idx, predictions, answers)
