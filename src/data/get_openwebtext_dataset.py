from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from .dataset import Dataset


class OpenWebText(Dataset):
    def __init__(self, tokenizer, model_type, min_length=0, max_length=1024):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model_type: Type of the model
        :param min_length: Minimum length of the text
        :param max_length: Maximum length of the text
        """
        super().__init__(tokenizer, model_type)
        self.tokenizer.pad_token = tokenizer.eos_token
        self.min_length = min_length
        self.max_length = max_length

    def get_tokenized_dataset(self):
        """
        Load and build the openwebtext dataset which is already preprocessed.
        :return: dataset
        """
        dataset = load_dataset("stas/openwebtext-10k", split="train[:50]")
        dataset = dataset.train_test_split(test_size=0.2)
        tokenized_dataset = dataset.map(
            self.preprocess, batched=True, remove_columns=dataset["train"].column_names
        )
        print(tokenized_dataset)

        return tokenized_dataset["train"], tokenized_dataset["test"]

    def preprocess(self, examples):
        inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Filter out examples that are too short
        mask = [len(ids) >= self.min_length for ids in inputs["input_ids"]]

        return {
            "input_ids": [ids for ids, m in zip(inputs["input_ids"], mask) if m],
            "attention_mask": [
                mask for mask, m in zip(inputs["attention_mask"], mask) if m
            ],
        }

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        return data_collator

    def get_compute_metrics(self):
        """
        Load the compute_metrics function that is used by the HuggingFace Trainer to
        evaluate the model.
        :return:
        """
        return self.compute_metrics()

    def compute_metrics(self):
        return None
