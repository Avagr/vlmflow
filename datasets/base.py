import abc

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    table_columns = None

    @abc.abstractmethod
    def table_repr(self, idx, prediction, img_as_object=True):
        raise NotImplementedError


class EvalWrapper(abc.ABC):
    callbacks = []

    @abc.abstractmethod
    def add_callback(self, callback):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, batch, model) -> 'EvaluationResult':
        raise NotImplementedError
