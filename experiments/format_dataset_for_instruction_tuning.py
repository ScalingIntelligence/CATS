from datasets import load_dataset

from utils.constants import (
    COLA,
    SST2,
    BOOLQ,
    QNLI,
    WIC,
    MULTIRC,
    GLUE,
    SUPERGLUE,
)


def get_dataset_for_instruction_tuning(dataset_type: str, seed: int = 42):
    if dataset_type in GLUE:
        dataset = load_dataset("glue", dataset_type)
    elif dataset_type in SUPERGLUE:
        dataset = load_dataset("super_glue", dataset_type)
    else:
        raise NotImplementedError(f"{dataset_type} is not implemented.")

    train_test_dataset = dataset["train"].train_test_split(test_size=0.1, seed=seed)
    train_dataset = train_test_dataset["train"]
    val_dataset = train_test_dataset["test"]
    test_dataset = dataset["validation"]
    dataset["train"] = train_dataset
    dataset["validation"] = val_dataset
    dataset["test"] = test_dataset
    return dataset


def get_instruction_and_keys(dataset_type):
    if dataset_type == COLA:
        keys = ["sentence"]
        instruction = f"Is the following 'sentence' grammatically correct? Respond in 'yes' or 'no'."
    elif dataset_type == SST2:
        keys = ["sentence"]
        instruction = "Is the sentiment of following 'sentence' positive? Respond in 'yes' or 'no'."
    elif dataset_type == QNLI:
        keys = ["sentence", "question"]
        instruction = "Does the 'sentence' contain the answer to the 'question'? Respond in 'yes' or 'no'."
    elif dataset_type == BOOLQ:
        keys = ["passage", "question"]
        instruction = "Based on the following 'passage', is the answer of the 'question' true? Respond in 'yes' or 'no'."
    elif dataset_type == WIC:
        keys = ["sentence1", "sentence2", "word"]
        instruction = "Is the 'word' used as a similar meaning in each 'sentence1' and 'sentence2'? Respond in 'yes' or 'no'."
    else:
        raise NotImplementedError(f"{dataset_type} is not implemented.")
    return instruction, keys


def get_formatting_func(dataset_type):
    instruction, keys = get_instruction_and_keys(dataset_type)

    def formatting_func(samples):
        formatted_prompts = []
        bos_token = "<s>"
        eos_token = "</s>"
        for idx in range(len(samples[keys[0]])):
            response = "yes" if samples["label"][idx] == 1 else "no"

            full_prompt = ""
            full_prompt += bos_token
            full_prompt += f"[INST]### Instruction: {instruction}"
            for key in keys:
                full_prompt += f"### {key}: {samples[key][idx]}"
            full_prompt += f"[/INST]"
            full_prompt += f"### Response: {response}"
            full_prompt += eos_token
            formatted_prompts.append(full_prompt)
        return formatted_prompts

    return formatting_func


def formatting_func_cola(samples):
    formatted_prompts = []
    bos_token = "<s>"
    instruction = "Is the following 'input' sentence is grammatically correct? Respond in 'yes' or 'no'."
    eos_token = "</s>"
    for idx in range(len(samples["sentence"])):
        input = samples["sentence"][idx]
        response = "yes" if samples["label"][idx] == 1 else "no"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += f"[INST]### Instruction: {instruction}"
        full_prompt += f"### Input: {input}[/INST]"
        full_prompt += f"### Response: {response}"
        full_prompt += eos_token
        formatted_prompts.append(full_prompt)
    return formatted_prompts


def formatting_func_sst2(samples):
    formatted_prompts = []
    bos_token = "<s>"
    instruction = "Is the following 'input' sentence is grammatically correct? Respond in 'yes' or 'no'."
    eos_token = "</s>"
    for idx in range(len(samples["sentence"])):
        input = samples["sentence"][idx]
        response = "yes" if samples["label"][idx] == 1 else "no"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += f"[INST]### Instruction: {instruction}"
        full_prompt += f"### Input: {input}[/INST]"
        full_prompt += f"### Response: {response}"
        full_prompt += eos_token
        formatted_prompts.append(full_prompt)
    return formatted_prompts


def formatting_func_boolq(samples):
    formatted_prompts = []
    bos_token = "<s>"
    instruction = "Based on the following 'passage', is the 'question' true or false? Respond in 'yes' or 'no'."
    eos_token = "</s>"

    for idx in range(len(samples["question"])):
        passage = samples["passage"][idx]
        question = samples["question"][idx]
        response = "yes" if samples["label"][idx] == 1 else "no"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += f"[INST]### Instruction: {instruction}"
        full_prompt += f"### Passage: {passage}"
        full_prompt += f"### Question: {question}[/INST]"
        full_prompt += f"### Response: {response}"
        full_prompt += eos_token
        formatted_prompts.append(full_prompt)

    return formatted_prompts
