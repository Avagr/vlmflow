from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm
import wandb

from datasets.base import BaseDataset, EvalWrapper


def eval_model(model, eval_wrapper: EvalWrapper, dataloader, show_tqdm=False):
    model.eval()
    size = 0
    total_matches = 0
    results = []
    for batch in tqdm(dataloader, disable=not show_tqdm):
        res = eval_wrapper(batch, model)
        results.append(res)
        total_matches += res.num_matches()
        size += res.batch_size
    return total_matches / size, results


@dataclass
class EvaluationResult:
    batch_size: int
    dataset_ids: list[int]
    texts: list[str]
    predictions: torch.Tensor | list[str] | list[tuple]
    true_labels: torch.Tensor | list[str] | list[tuple]
    generated_ids: list[list[int]]
    scores: dict[str, list] = field(default_factory=dict)

    def matches(self):
        matches = []
        if isinstance(self.predictions, torch.Tensor):
            matches = (self.predictions == self.true_labels).tolist()
        elif isinstance(self.predictions[0], str):
            for pred, label in zip(self.predictions, self.true_labels):
                if pred.strip().lower() == label.strip().lower():
                    matches.append(1)
                else:
                    matches.append(0)
        else:
            for pred, label in zip(self.predictions, self.true_labels):
                if pred == label:
                    matches.append(1)
                else:
                    matches.append(0)
        return matches

    def num_matches(self):
        return sum(self.matches())

    def to_table(self, dataset: BaseDataset) -> wandb.Table:
        matches = self.matches()
        table_data = [
            dataset.table_repr(i, self.predictions[i], img_as_object=False)
            + [self.texts[i]]
            + [self.generated_ids[i]]
            + [matches[i]]
            + [self.scores[k][i] for k in sorted(self.scores.keys())]
            for i in self.dataset_ids]
        return wandb.Table(
            data=table_data,
            columns=dataset.table_columns + ['prompt', 'generated_ids', 'match'] + sorted(self.scores.keys()),
            allow_mixed_types=True
        )


def merge_results(results, callbacks=None):
    dataset_ids = []
    texts = []
    predictions = []
    labels = []
    generated_ids = []
    scores = {k: [] for k in results[0].scores.keys()}
    for res in results:
        dataset_ids.extend(res.dataset_ids)
        texts.extend(res.texts)
        predictions.extend(res.predictions)
        labels.extend(res.true_labels)
        generated_ids.extend(res.generated_ids)
        for k, v in res.scores.items():
            if k not in scores:
                raise ValueError(f"All additional scores should be the same across results, not true for key {k}")
            scores[k].extend(v)

    if isinstance(predictions[0], torch.Tensor):
        predictions = torch.cat([res.predictions for res in results])
        labels = torch.cat([res.true_labels for res in results])

    for callback in callbacks:
        cb_labels = callback.score_labels
        cb_results = callback.get_results()
        for label in cb_labels:
            if label not in scores:
                scores[label] = []
            else:
                print(f"Ignoring {callback} label {label} as it already exists in scores dict")
        for lab, res in zip(cb_labels, cb_results):
            for i in dataset_ids:
                scores[lab].append(res[i])

    return EvaluationResult(
        batch_size=sum(res.batch_size for res in results),
        dataset_ids=dataset_ids,
        texts=texts,
        predictions=predictions,
        true_labels=labels,
        generated_ids=generated_ids,
        scores=scores
    )


def sample_results(results: EvaluationResult, dataset: BaseDataset, num_samples: int):
    matches = torch.BoolTensor(results.matches())
    correct_ind = results.dataset_ids[matches][:num_samples].tolist()
    mistakes_ind = results.dataset_ids[~matches][:num_samples].tolist()
    correct = [dataset.table_repr(i, results.predictions[i]) for i in correct_ind]
    mistakes = [dataset.table_repr(i, results.predictions[i]) for i in mistakes_ind]
    headers = dataset.table_columns
    return wandb.Table(data=correct, columns=headers), wandb.Table(data=mistakes, columns=headers)
