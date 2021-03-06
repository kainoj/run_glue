import torch
from torch import nn
from torch.optim import AdamW

from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from datasets import load_metric


class GlueModel(LightningModule):

    """Lightning module that defines behavior of the training loop.

    Args:
        model: actual 🤗model to evaluate. Must retrun SequenceClassifierOutput on forward.
        task_name: cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, or wnli
        learning_rate, weight_decay, adam_epsilon, warmup_steps: stuff for optimizer and scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        task_name: str,
        learning_rate: float,
        weight_decay: float,
        adam_epsilon: float,
        warmup_steps: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = model
        self.lr = learning_rate
        self.metric = load_metric("glue", task_name)
        self.task_name = task_name
        self.is_regression = self.task_name == "stsb"

        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps

    def step(self, inputs):
        """A single step. Retruns a tuple (loss, logits).
        """
        outputs = self.model(**inputs)
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        """Forward, loss and log.
        """
        loss, _ = self.step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Forward, loss, calculate metric and log.
        """
        loss, logits = self.step(batch)

        preds = torch.squeeze(logits) if self.is_regression else torch.argmax(logits, axis=1)

        result = self.metric.compute(predictions=preds, references=batch['labels'])

        result["combined_score"] = torch.tensor(list(result.values())).mean().item()

        for key, val in result.items():
            self.log(f"val/{key}", val, on_step=False, on_epoch=True, prog_bar=False)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices.

        Code courtesy of Mario.
        """
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def configure_optimizers(self):
        """Configure the optimizer and the scheduler.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
