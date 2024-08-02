import json
import re
from pathlib import Path

import torch
import wandb
from PIL import Image

from datasets.base import BaseDataset, EvalWrapper
from models.wrappers import GenerativeWrapper
from utils.eval import EvaluationResult


class SEEDBenchSingleImage(BaseDataset):
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    table_columns = ["idx", "image", "prediction", "answer"]

    def __init__(self, task_num: int, json_path: Path, root_dir: Path):
        """
        :param task_num: number between 1 and 16
        :param json_path: path to the json question file
        :param root_dir: root directory of the images, the cc3m-images directory should be placed in it
        :param img_transform: transform to apply to the images
        """
        super().__init__()
        assert 0 < task_num < 17, f"task_num should be between 1 and 16, got: {task_num}"
        self.task_num = task_num
        self.root_dir = root_dir
        self.json_path = json_path
        with open(json_path, 'r') as f:
            data = json.load(f)
        for k, v in data['question_type'].items():
            if v == task_num:
                self.question_type = k
                break
        self.questions: list[dict[str, str]] = [x for x in data['questions'] if x['question_type_id'] == self.task_num]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx) -> tuple[int, Image, str, list[str], str]:
        question = self.questions[idx]
        if question['data_source'] == 'cc3m':
            img_root_dir = self.root_dir / 'cc3m-images'
        else:
            img_root_dir = self.root_dir
        image = Image.open(img_root_dir / question['data_id']).convert('RGB')
        choices = [question[f"choice_{i}"] for i in "abcd"]
        return idx, image, question['question'], choices, self.answer_map[question['answer'].strip()]

    def table_repr(self, idx, prediction, img_as_object=True):
        question = self.questions[idx]
        if question['data_source'] == 'cc3m':
            img_root_dir = self.root_dir / 'cc3m-images'
        else:
            img_root_dir = self.root_dir
        choices = [question[f"choice_{i}"] for i in "abcd"]
        return [
            idx,
            wandb.Image(str((img_root_dir / question['data_id']).resolve())) if img_as_object else question['data_id'],
            f"{chr(prediction + ord('A'))}: {choices[prediction]}",
            f"{question['answer']}: {choices[self.answer_map[question['answer'].strip()]]}"
        ]


class SEEDBenchCollate:
    def __call__(self, batch: list[(int, Image, str, list[str], int)]):
        idx = []
        images = []
        questions = []
        choices = []
        answers = []
        for i, image, question, choice, answer in batch:
            idx.append(i)
            images.append(image)
            questions.append(question)
            choices.append(choice)
            answers.append(answer)

        return torch.LongTensor(idx), images, questions, choices, torch.LongTensor(answers)


class SEEDBenchSingleImageEval(EvalWrapper):

    def __init__(self, prompt, device, eval_method="ppl"):
        """
        Supports the following scoring methods:

            * "ppl": take the answer with the lowest loss in a <question: q, answer: answer> setting
            * "abcd": take the answer with the highest score in a <question: q, choices: [a, b, c, d], answer:> setting

        :param prompt: prompt format: arbitrary string with <Q> and <C> being placeholders for the question and the choices
        :param device: pytorch device
        :param eval_method: method to use for scoring:
        """
        split_prompt = re.split(r'(<[QC]>)', prompt)
        assert len(split_prompt) == 5, f"Prompt should have exactly 2 placeholders, got {len(split_prompt)}"
        self.pre_prompt = split_prompt[0]
        self.mid_prompt = split_prompt[2]
        self.post_prompt = split_prompt[4]
        self.device = device
        self.eval_method = eval_method
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, batch, model: GenerativeWrapper) -> EvaluationResult:
        idx, images, questions, choices, answers = batch
        batch_size = idx.shape[0]
        match self.eval_method:
            case "ppl":
                texts = [f"{self.pre_prompt}{q}{self.mid_prompt}{cand[c]}{self.post_prompt}" for q, cand in
                         zip(questions, choices) for c in range(4)]
                images = [img for img in images for _ in range(4)]
                scores, metrics = model.score_text(images, texts).view(batch_size, 4)
                predictions = scores.argmin(-1).cpu()

            case "abcd":
                texts = [(f"{self.pre_prompt}{q}{self.mid_prompt}"
                          f"A. {cand[0]}\nB. {cand[1]}\nC. {cand[2]}\nD. {cand[3]}\n"
                          f"{self.post_prompt}") for q, cand in zip(questions, choices)]
                scores, metrics = model.score_single_tokens(images, texts, ['A', 'B', 'C', 'D'])
                predictions = scores.argmax(-1).cpu()
            case _:
                raise ValueError(f"Unsupported evaluation method {self.eval_method}")

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(model, idx, texts, answers, predictions, scores)

        return EvaluationResult(batch_size, idx.tolist(), texts, predictions, answers, metrics)
