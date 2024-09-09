from transformers import DataCollatorForSeq2Seq
import evaluate
import torch
from datasets import load_dataset
import pickle
import os

from .dataset import Dataset
from utils.constants import (
    QNLI,
    SST2,
    COLA,
)


def convert_to_t5_format(example, task, tokenizer):
    if task == "qnli":
        source_text = (
            "context: "
            + example["sentence"]
            + " question: "
            + example["question"]
            + " entailment: does the context entail the question?"
        )
        # target_text = "entailment" if example["label"] == 1 else "not_entailment"
        target_text = "yes" if example["label"] == 1 else "no"
        # labels = ["entailment", "not_entailment"]
    elif task == "sst2":
        source_text = (
            "sentiment: " + example["sentence"] + " positive or negative?"
        )
        # target_text = "positive" if example["label"] == 1 else "negative"
        target_text = "yes" if example["label"] == 1 else "no"
        # labels = ["positive", "negative"]
    elif task == "cola":
        source_text = (
            "grammar: " + example["sentence"] + " acceptable or unacceptable?"
        )
        # target_text = "acceptable" if example["label"] == 1 else "unacceptable"
        target_text = "yes" if example["label"] == 1 else "no"
        # labels = ["acceptable", "unacceptable"]
    elif task == "squad":
        source_text = (
            "question: "
            + example["question"]
            + " context: "
            + example["context"]
        )
        target_text = example["answers"]["text"]
    else:
        raise ValueError(f"Invalid task ({task})")

    target_tokenized = tokenizer(
        target_text,
        return_tensors="pt",
        # padding="max_length",
        # max_length=1,
    )
    labels = target_tokenized.input_ids[0]
    # labels[labels == tokenizer.eos_token_id] = -100

    return {
        "input_text": source_text,
        "target_text": target_text,
        "input_ids": tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        ).input_ids[0],
        "labels": labels,
        # "labels_attention_mask": target_tokenized.attention_mask[0],
        # "decoder_attention_mask": target_tokenized.attention_mask[0],
    }


def get_metrics(data_type):
    return evaluate.load("glue", data_type)
    if data_type == COLA:
        metric = evaluate.load("matthews_correlation")
    elif data_type == SST2:
        metric = evaluate.load("evaluate-metric/spearmanr")
    elif data_type == QNLI:
        metric = evaluate.load("accuracy")
    else:
        metric = None
    return metric


class GLUE(Dataset):
    def __init__(
        self,
        tokenizer,
        model,
        model_type: str,
        dataset_type: str,
    ):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model, model_type)
        self.dataset_type = dataset_type
        self.metrics = get_metrics(dataset_type)

    def get_tokenized_dataset(self):
        """
        Load and build a simple summarization dataset
        :return: tokenized dataset, data collator, and compute_metrics function
        """
        try:
            # Try to load preprocessed data
            with open(
                f"{self.model_type}_{self.dataset_type}_dataset.pkl", "rb"
            ) as f:
                tokenized_dataset = pickle.load(f)
            print("Loaded preprocessed data from disk.")

        except FileNotFoundError:
            # If preprocessed data does not exist, load and preprocess
            print("No dataset found.")
            dataset = load_dataset(
                "glue",
                self.dataset_type,
            )

            tokenized_dataset = dataset.map(
                lambda x: convert_to_t5_format(
                    x, self.dataset_type, self.tokenizer
                ),
                remove_columns=dataset.column_names["train"],
                load_from_cache_file=False,
                # batched=True,
            )

            # Save the preprocessed data
            with open(
                f"{self.model_type}_{self.dataset_type}_dataset.pkl", "wb"
            ) as f:
                pickle.dump(tokenized_dataset, f)
            print("Preprocessed and saved data to disk.")

        train_dataset = tokenized_dataset["train"]
        val_dataset = tokenized_dataset["validation"]

        return train_dataset, val_dataset

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            max_length=512,
            padding=True,
        )
        return data_collator

    def get_compute_metrics(self):
        """
        Load the compute_metrics function that is used by the HuggingFace Trainer to
        evaluate the model.
        :return:
        """
        return self.compute_metrics

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

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        # labels = labels[:, 1:].reshape(-1)
        # preds = preds[:, :-1].reshape(-1)
        # labels = labels[:, :].reshape(-1)
        # preds = preds[:, :].reshape(-1)
        # labels = preds.label_ids
        # preds = preds.predictions

        labels = labels[:, 0].reshape(-1)
        preds = preds[:, 0].reshape(-1)

        if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
            print("preds : \n", preds[:8])
            print("labels: \n", labels[:8])

        # valid_token_mask = (
        #     labels != -100
        # )  # Create a mask of valid (non-padding) tokens
        # labels = labels[valid_token_mask]
        # preds = preds[valid_token_mask]

        if self.metrics.name in ["exact_match"]:
            preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True
            )
            labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )

        return self.metrics.compute(predictions=preds, references=labels)

    # def compute_metrics(self, pred):
    #     labels = pred.label_ids
    #     predictions = pred.predictions[0]
    #     decoded_preds = self.tokenizer.batch_decode(
    #         predictions, skip_special_tokens=True
    #     )
    #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    #     decoded_labels = self.tokenizer.batch_decode(
    #         labels, skip_special_tokens=True
    #     )
    #
    #     result = self.metrics.compute(
    #         predictions=decoded_preds,
    #         references=decoded_labels,
    #         use_stemmer=True,
    #     )
    #
    #     prediction_lens = [
    #         np.count_nonzero(pred != self.tokenizer.pad_token_id)
    #         for pred in predictions
    #     ]
    #     result["gen_len"] = np.mean(prediction_lens)
    #
    #     return {k: round(v, 4) for k, v in result.items()}
