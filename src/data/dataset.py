from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, tokenizer, model_type):
        self.tokenizer = tokenizer
        self.model_type = model_type

    @abstractmethod
    def get_tokenized_dataset(self):
        pass

    @abstractmethod
    def preprocess(self, examples):
        pass

    def get_compute_metrics(self):
        """
        Return the compute_metrics function that is used by the HuggingFace Trainer to
        evaluate the model.
        """
        return self.compute_metrics

    @abstractmethod
    def compute_metrics(self, logits=None, labels=None):
        pass
