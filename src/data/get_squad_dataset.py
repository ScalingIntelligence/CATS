from transformers import DefaultDataCollator, AutoTokenizer, EvalPrediction
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import torch

from .dataset import Dataset


class SQuAD(Dataset):
    def __init__(self, tokenizer, model, model_type, metrics="f1"):
        super().__init__(tokenizer, model, model_type)
        self.metrics = evaluate.load(metrics)
        self.config = model.config

    def get_tokenized_dataset(self):
        """
        Load and build a Question and Answer dataset
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :return: tokenized dataset, data collator, and compute_metrics function
        """
        squad = load_dataset("squad")
        dataset = squad.train_test_split(test_size=0.2)

        tokenized_dataset = dataset.map(self.preprocess, batched=True)

        return tokenized_dataset

    def get_data_collator(self):
        data_collator = DefaultDataCollator(
            tokenizer=self.tokenizer, model=self.model
        )
        return data_collator

    def get_compute_metrics(self):
        return self.compute_metrics

    def compute_metrics(self, eval_pred: EvalPrediction):
        logits, labels = eval_pred
        predictions = torch.argmax(logits, dim=-1)

        # Compute accuracy
        accuracy = accuracy_score(labels, predictions)

        # Compute f1 score
        f1 = f1_score(labels, predictions)

        # Return a dictionary of metrics
        return {"accuracy": accuracy, "f1": f1}

    def preprocess(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
