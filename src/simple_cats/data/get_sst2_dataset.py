from datasets import load_dataset, load_metric
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate

from .dataset import Dataset


class SST2_Dataset(Dataset):
    def __init__(self, tokenizer, model=None, model_type: str = ""):
        # tokenizer.pad_token = tokenizer.eos_token
        super().__init__(tokenizer, model, model_type)
        self.metric = evaluate.load("accuracy")

    def get_tokenized_dataset(self):
        dataset = load_dataset("glue", "sst2")
        tokenized_dataset = dataset.map(self.preprocess, batched=True)
        return tokenized_dataset["train"], tokenized_dataset["validation"]

    def preprocess(self, examples):
        tokenized = self.tokenizer(
            examples["sentence"], return_tensors="pt", padding=True
        )
        return dict(tokenized)

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return data_collator

    def compute_metrics(self, evals):
        logits, labels = evals
        predictions = np.argmax(logits, axis=-1)
        # print(predictions.shape, labels.shape)
        return self.metric.compute(predictions=predictions, references=labels)

    def get_compute_metrics(self):
        return self.compute_metrics
