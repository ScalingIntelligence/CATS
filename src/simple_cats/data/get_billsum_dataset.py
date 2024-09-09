from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
import torch
from datasets import load_dataset
import pickle

from .dataset import Dataset


class Billsum(Dataset):
    def __init__(
        self,
        tokenizer,
        model,
        model_type: str,
        metrics: str = "rouge",
        prefix: str = "summarize: ",
    ):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model, model_type)
        self.metrics = evaluate.load(
            metrics
        )  # Load the metrics to be used by compute_metrics
        self.prefix = prefix

    def get_tokenized_dataset(self):
        """
        Load and build a simple summarization dataset
        :return: tokenized dataset, data collator, and compute_metrics function
        """
        try:
            # Try to load preprocessed data
            with open(
                f"{self.model_type}_tokenized_train_dataset.pkl", "rb"
            ) as f:
                train_dataset = pickle.load(f)
            with open(
                f"{self.model_type}_tokenized_test_dataset.pkl", "rb"
            ) as f:
                test_dataset = pickle.load(f)
            print("Loaded preprocessed data from disk.")

        except FileNotFoundError:
            # If preprocessed data does not exist, load and preprocess
            dataset = load_dataset(
                "billsum",
                split="train",
            )
            dataset = dataset.train_test_split(test_size=0.1)
            tokenized_dataset = dataset.map(self.preprocess, batched=True)
            train_dataset = tokenized_dataset["train"]
            test_dataset = tokenized_dataset["test"]

            # Save the preprocessed data
            with open(
                f"{self.model_type}_tokenized_train_dataset.pkl", "wb"
            ) as f:
                pickle.dump(train_dataset, f)
            with open(
                f"{self.model_type}_tokenized_test_dataset.pkl", "wb"
            ) as f:
                pickle.dump(test_dataset, f)
            print("Preprocessed and saved data to disk.")

        return train_dataset, test_dataset

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

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
        inputs = [self.prefix + doc for doc in examples["text"]]
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)

        # TODO: decrease max_length of labels
        #       (but then it raises ValueError: Expected input batch_size (2032) to match target batch_size (496).)
        labels = self.tokenizer(
            text_target=examples["summary"],
            padding=True,
            max_length=512,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def compute_metrics(self, pred):
        labels = pred.label_ids
        predictions = pred.predictions[0]
        # predictions = np.argmax(predictions, axis=-1)
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        result = self.metrics.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
