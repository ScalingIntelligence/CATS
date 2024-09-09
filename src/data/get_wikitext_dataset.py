from abc import ABC, abstractmethod
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from .dataset import Dataset


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, model_type, split="test", max_length=512):
        super().__init__(tokenizer, model_type)
        self.split = split
        self.max_length = max_length
        self.raw_dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split=self.split
        )

    def get_tokenized_dataset(self):
        return self.raw_dataset.map(
            self.preprocess, batched=True, remove_columns=["text"]
        )

    def preprocess(self, examples):
        return self.tokenizer(
            examples["text"], truncation=True, max_length=self.max_length
        )

    def compute_metrics(self, logits=None, labels=None):
        if logits is None or labels is None:
            raise ValueError("Both logits and labels must be provided.")

        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels.flatten(), predictions.flatten())}

    def get_dataloader(self, batch_size=4):
        tokenized_dataset = self.get_tokenized_dataset()
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return DataLoader(
            tokenized_dataset, batch_size=batch_size, collate_fn=data_collator
        )
