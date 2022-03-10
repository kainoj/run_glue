from typing import Dict
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class MultitaskModel(nn.Module):

    def __init__(
        self,
        model_name: str,
        tasks: Dict[str, int],
    ) -> None:
        super().__init__()

        self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.backbone.classifier = nn.Identity()  # disable default classifier
        self.heads = nn.ModuleDict({
            name: nn.Linear(768, num_labels) for name, num_labels in tasks.items()
        })

    def forward(self, x: torch.tensor, task: str):
        y = self.backbone(**x).logits
        y = self.heads[task](y)
        return y
