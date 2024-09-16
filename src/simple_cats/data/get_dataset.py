from transformers import AutoTokenizer

from .get_billsum_dataset import Billsum
from .get_squad_dataset import SQuAD
from .get_openwebtext_dataset import OpenWebText
from .get_samsum_dataset_ import Samsum
from .get_glue_dataset_classification import GLUEDataset
from .get_refinedweb_dataset import RefinedWeb

from ..cats_utils.constants import (
    COLA,
    BILLSUM,
    SQUAD,
    OPENWEBTEXT,
    SAMSUM,
    GLUE,
    SUPERGLUE,
    REFINED_WEB,
    MISTRAL_7B,
)


def get_dataset(
    dataset_name: str,
    tokenizer,
    model_type: str,
    max_seq_length: int = 1024,
    min_seq_length: int = 0,
):
    """
    Tokenize a dataset and get additional data collator and metrics necessary for training the models

    :param dataset_name: Name of the dataset to build
    :param model_name: Name of the model
    :param tokenizer: Tokenizer to tokenize the inputs
    :param model: Model to finetune / evaluate
    :return: tokenized dataset, data collator, and compute_metrics function
    """
    print("DATASET NAME: ", dataset_name)
    if dataset_name == BILLSUM:
        dataset = Billsum(tokenizer, model_type)
    elif dataset_name == SQUAD:
        dataset = SQuAD(tokenizer, model_type)
    elif dataset_name == OPENWEBTEXT:
        dataset = OpenWebText(
            tokenizer, model_type, max_length=max_seq_length, min_length=min_seq_length
        )
    elif dataset_name == REFINED_WEB:
        dataset = RefinedWeb(
            tokenizer,
            model_type,
            max_seq_length=max_seq_length,
        )
    elif dataset_name == SAMSUM:
        dataset = Samsum(tokenizer, model_type)
    elif dataset_name in GLUE + SUPERGLUE:
        dataset = GLUEDataset(tokenizer, model_type, dataset_type=dataset_name)
    else:
        assert False, f"No dataset named {dataset_name} available."

    return dataset


if __name__ == "__main__":
    model = None
    model_type = MISTRAL_7B
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = get_dataset(COLA, tokenizer, model, model_type)
