import torch


class InputsHook:
    def __init__(self, arg_ind=None, kwarg_name=None):

        if arg_ind is not None and kwarg_name is not None:
            raise ValueError("Only one of arg_ind and kwarg_name should be provided")

        if arg_ind is None and kwarg_name is None:
            raise ValueError("Either arg_ind or kwarg_name should be provided")

        self.value: torch.Tensor | None = None
        self.arg_ind = arg_ind
        self.kwarg_name = kwarg_name

    def __call__(self, _, args, kwargs):
        if self.arg_ind is not None:
            self.value = args[self.arg_ind]
        if self.kwarg_name is not None:
            self.value = kwargs[self.kwarg_name]


class OutputsHook:
    def __init__(self, arg_ind=None, kwarg_name=None):

        if arg_ind is not None and kwarg_name is not None:
            raise ValueError("Only one of arg_ind and kwarg_name should be provided")

        if arg_ind is None and kwarg_name is None:
            raise ValueError("Either arg_ind or kwarg_name should be provided")

        self.value: torch.Tensor | None = None
        self.arg_ind = arg_ind
        self.kwarg_name = kwarg_name

    def __call__(self, _, __, output):
        self.value = output
