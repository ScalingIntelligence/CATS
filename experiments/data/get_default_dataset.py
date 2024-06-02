from datasets import load_dataset, load_metric
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate

from .dataset import Dataset


class DefaultDataset(Dataset):
    def __init__(self, tokenizer, model=None, model_type=None,):
        super().__init__(tokenizer, model, model_type)

    def get_tokenized_dataset(self):
        return None

    def preprocess(self, examples):
        return None

    def get_data_collator(self):
        return None

    def compute_metrics(self, evals):
        return None
