from transformers import DataCollatorForSeq2Seq, AutoTokenizer
import evaluate
import numpy as np
from datasets import load_dataset
import pickle

from .dataset import Dataset


class Samsum(Dataset):
    def __init__(
        self,
        tokenizer,
        model,
        metrics: str = "rouge",
        prefix: str = "summarize: ",
        max_length: int = 256,
    ):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model)
        self.tokenizer = tokenizer
        self.metrics = evaluate.load(
            metrics
        )  # Load the metrics to be used by compute_metrics
        self.prefix = prefix
        self.max_length = max_length

    def get_tokenized_dataset(self):
        """
        Load and build a simple summarization dataset
        :return: tokenized dataset, data collator, and compute_metrics function
        """
        try:
            # Try to load preprocessed data
            with open("samsum_train_dataset.pkl", "rb") as f:
                train_dataset = pickle.load(f)
            with open("samsum_test_dataset.pkl", "rb") as f:
                test_dataset = pickle.load(f)
            print("Loaded preprocessed data from disk.")

        except FileNotFoundError:
            # If preprocessed data does not exist, load and preprocess
            dataset = load_dataset(
                "samsum",
                split="train",
            )
            dataset = dataset.train_test_split(test_size=0.1)
            tokenized_dataset = dataset.map(self.preprocess, batched=True)
            train_dataset = tokenized_dataset["train"]
            test_dataset = tokenized_dataset["test"]

            # Save the preprocessed data
            with open("samsum_train_dataset.pkl", "wb") as f:
                pickle.dump(train_dataset, f)
            with open("samsum_test_dataset.pkl", "wb") as f:
                pickle.dump(test_dataset, f)
            print("Preprocessed and saved data to disk.")

        return train_dataset, test_dataset

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )
        return data_collator

    def preprocess(self, examples):
        inputs = [self.prefix + doc for doc in examples["dialogue"]]
        model_inputs = self.tokenizer(
            inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        labels = self.tokenizer(
            text_target=examples["summary"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def compute_metrics(self):
        pass


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Samsum(tokenizer, None)
    dataset.get_tokenized_dataset()
