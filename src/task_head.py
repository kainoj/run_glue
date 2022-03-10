import torch
from torch import nn


class TaskHead(nn.Module):

    def __init__(self, task_name: str, num_classes: int) -> None:
        super().__init__()
        self.task_name = task_name
        self.num_classes = num_classes

        self.cls = nn.Linear