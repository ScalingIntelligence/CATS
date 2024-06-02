from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import pickle
import torch
import os

from utils.utils import is_mainprocess
from .dataset import Dataset
from datasets import Dataset as D
from utils.constants import REFINED_WEB, MISTRAL_CONTEXT_LENGTH


class RefinedWeb(Dataset):
    def __init__(self, tokenizer, model_type, max_seq_length: int = 8096):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model_type)
        self.dataset_type = REFINED_WEB
        self.max_seq_length = max_seq_length
        self.seed = 0

    def get_tokenized_dataset(self):
        """
        Load and build the openwebtext dataset which is already preprocessed.
        :return: dataset
        """
        if self.max_seq_length == 8096:
            prefix = f"{self.model_type}_{self.dataset_type}"
        else:
            prefix = f"{self.model_type}_{self.dataset_type}_seq{self.max_seq_length}_seed{self.seed}_"

        train_path = f"datasets/{prefix}_train.pkl"
        test_path = f"datasets/{prefix}_test.pkl"
        validation_path = f"datasets/{prefix}_validation.pkl"

        for path in [train_path, test_path, validation_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            # Try to load preprocessed data
            # open(random_path, "rb")
            with open(train_path, "rb") as f:
                train_dataset = pickle.load(f)
            with open(test_path, "rb") as f:
                test_dataset = pickle.load(f)
            with open(validation_path, "rb") as f:
                val_dataset = pickle.load(f)
            print(f"Loaded preprocessed data from disk. Prefix: {prefix}")
        except FileNotFoundError:
            dataset = load_dataset(
                "tiiuae/falcon-refinedweb",
                streaming=True,
                # num_proc=5,
            )
            dataset.shuffle(buffer_size=1000000)  # seed = 42
            print("Downloaded preprocessed data from disk.")

            print("dataset: ", dataset)

            tokenized_dataset = dataset.map(
                self.preprocess,
                batched=True,
                batch_size=32,
                # num_proc=8,
                remove_columns=[
                    "content",
                    "url",
                    "timestamp",
                    "dump",
                    "segment",
                    "image_urls",
                ],
            )["train"]
            print(tokenized_dataset)

            train_dataset = D.from_list(list(tokenized_dataset.skip(10000).take(100000)))
            val_dataset = D.from_list(list(tokenized_dataset.take(50)))
            test_dataset = D.from_list(list(tokenized_dataset.skip(1000).take(500)))
            # train_dataset = tokenized_dataset

            print(train_dataset)
            print(val_dataset)

            # Save the preprocessed data
            with open(train_path, "wb") as f:
                pickle.dump(train_dataset, f)
            with open(test_path, "wb") as f:
                pickle.dump(test_dataset, f)
            with open(validation_path, "wb") as f:
                pickle.dump(val_dataset, f)
            print("Preprocessed and saved data to disk.")
        train_dataset.filter(lambda item: len(item["input_ids"]) < self.max_seq_length)
        train_dataset.shuffle()
        return train_dataset, val_dataset, test_dataset

    def preprocess(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["content"],
            truncation=True,
            max_length=self.max_seq_length,
            return_overflowing_tokens=True,
            # return_length=True,
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
        }

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        return data_collator

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = logits.max(axis=-1)
        return pred_ids[0]

    def compute_metrics(self, preds):
        logits, labels = preds

        logits = torch.from_numpy(logits).float()
        labels = torch.from_numpy(labels)

        print("logits: ", logits.shape, logits[0][:4])
        print("labels: ", labels.shape)

        # Shift the labels to the right to align with the output of the model
        shifted_logits = logits[..., :-1].contiguous()
        shifted_labels = labels[..., 1:].contiguous()

        print("shifted_logits: ", shifted_logits.shape)
        print("shifted_labels: ", shifted_labels.shape)

        # Flatten the logits and labels to calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            # shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_logits.view(-1),
            shifted_labels.view(-1),
        )

        # Calculate perplexity
        perplexity = torch.exp(torch.mean(loss))

        return {"perplexity": perplexity.item()}
