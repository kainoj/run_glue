from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets import load_dataset
from transformers import default_data_collator, AutoTokenizer


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class GlueDataModule(LightningDataModule):

    def __init__(
        self,
        task_name: str,
        tokenizer_name: str,
        max_seq_length: int,
        batch_size: int,
        cache_dir: str,
        overwrite_cache: bool,
    ):
        super().__init__()
        self.task_name = task_name
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache

    @property
    def num_labels(self) -> int:
        is_regression = self.task_name == "stsb"
        if not is_regression:
            label_list = self.data["train"].features["label"].names
            return len(label_list)
        return 1

    def prepare_data(self):
        load_dataset("glue", self.task_name, cache_dir=self.cache_dir)

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.data = load_dataset("glue", self.task_name, cache_dir=self.cache_dir)

        self.data = self.data.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not self.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        self.data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    def preprocess_function(self, examples):
        # Tokenize the texts
        sentence1_key, sentence2_key = task_to_keys[self.task_name]

        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = self.tokenizer(*args, padding="max_length", max_length=self.max_seq_length, truncation=True)

        return result

    def train_dataloader(self):
        return DataLoader(self.data['train'], batch_size=self.batch_size, collate_fn=default_data_collator)

    def val_dataloader(self):
        return DataLoader(self.data['validation'], batch_size=self.batch_size, collate_fn=default_data_collator)

    def test_dataloader(self):
        pass
