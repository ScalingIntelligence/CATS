import argparse

from utils.constants import GPT2, GPT2_MEDIUM


def parse_string(s: str = None):
    """
    Parses a string representing a list of integers and ranges of integers and returns a list of integers.

    The input string should be in the format "a-b,c,d-e,...", where "a-b" represents a range of integers
    from a to b (inclusive), and "c" represents a single integer. The parts of the string should be separated by commas.

    This function can be used to parse strings representing lists of integers,
    for example when freezing specific layers or applying an accelerator to specific layers.

    :param s: The input string to parse.
    :return: A list of integers represented by the input string.
    """
    if not s or s == "None":
        return []

    result = []
    for part in s.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def get_model_type(model_name: str = None):
    model_types = [GPT2_MEDIUM, GPT2]

    for model_type in model_types:
        if model_type in model_name:
            return model_type


def string_to_dict(s: str) -> dict:
    """
    Convert a string of accelerator arguments into a dictionary.

    This function can be used to extract accelerator arguments information from the model name.

    Example:
    If the string is "a:10,k:20", this function will return a dictionary of {'a': 10, 'k': 20}.

    Args:
        s (str): The string of accelerator arguments to convert to a dictionary.

    Returns:
        dict: The dictionary form of the string.
    """
    if not s:
        return {}
    pairs = s.split(",")
    pairs = [pair.split(":") for pair in pairs]
    dic = {k: int(v) for (k, v) in pairs}
    return dic


def dict_to_string(dic: dict) -> str:
    """
    Convert a dictionary of accelerator arguments into a string.

    This function can be used to name the model in the "create_model" function inside create_models.py.

    Example:
    If the accelerator arguments are {'a': 10, 'k': 20}, the output string should be "a:10,k:20".

    :param: dic (dict): The dictionary of accelerator arguments to convert to a string.
    :return: The string form of the dictionary.
    """
    if dic == None:
        return "None"
    return ",".join([f"{k}:{v}" for (k, v) in dic.items()])


class BooleanOptionalAction(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        super(BooleanOptionalAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs="?",
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values or self.const)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for text classification")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random Seed",
    )
    parser.add_argument(
        "--base_model_repo_id", type=str, default=None, help="Base Model Repo ID (e.g. meta-llama/Llama-2-7b"
    )
    parser.add_argument("--model_name", type=str, default="Mistral_Sparse", help="Name of the model")
    parser.add_argument("--dataset_type", type=str, default="cola", help="Dataset Type")
    parser.add_argument("--checkpoint_dir", type=str, default="/scr/jay/ckpt", help="Checkpoint Directory")
    parser.add_argument("--results_dir", type=str, default="/scr/jay/ckpt", help="Log Directory")
    parser.add_argument("--num_epochs", type=float, default=7, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of maximum training steps")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient Accumulation Steps",
    )
    parser.add_argument("--use_stage3", action="store_true", default=False, help="Use Stage 3")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Whether to push the model to Hugging Face Hub",
    )
    parser.add_argument("--model_save", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Testing batch size")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--is_debugging", action="store_true", default=False)
    parser.add_argument("--is_plot", action="store_true", default=False)
    parser.add_argument(
        "--set_sparsity_aware_threshold",
        action="store_true",
        default=False,
        help="Set sparsity aware threshold",
    )
    parser.add_argument(
        "--use_graceful_regularization",
        action="store_true",
        default=False,
        help="Whether to do apply regularization before killing activations",
    )
    parser.add_argument(
        "--use_distillation",
        action="store_true",
        default=False,
        help="Whether to do apply distillation",
    )
    parser.add_argument(
        "--keep_regularization_with_kill",
        action="store_true",
        default=False,
        help="Whether to keep regularization after beginning killing activations",
    )
    parser.add_argument(
        "--is_first_training",
        type=int,
        default=0,
        help="Start training step",
    )
    parser.add_argument(
        "--use_gradual_sparsification",
        action="store_true",
        default=False,
        help="Whether to do gradually increase the threshold for killing neurons",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=40,
        help="Number of training steps required to reach the dead threshold",
    )
    parser.add_argument(
        "--increment_ratio",
        type=float,
        default=0.5,
        help="By how much to increase the dead threshold",
    )
    parser.add_argument(
        "--print_act_stats",
        action="store_true",
        default=False,
        help="Print pre/post-activation statistics.",
    )
    parser.add_argument(
        "--print_sparsity",
        action="store_true",
        default=False,
        help="Print Sparsity of the mlp layers for test dataset.",
    )
    parser.add_argument(
        "--targeted_sparsity",
        type=float,
        default=0.9,
        help="Targeted Sparsity of the sparse mlp layers in Mistral.",
    )
    parser.add_argument(
        "--process_index",
        type=int,
        default=2,
        help="Targeted Sparsity of the sparse mlp layers in Mistral.",
    )
    parser.add_argument(
        "--use_sparse_model",
        action="store_true",
        default=False,
        help="Whether to use sparse Mistral model",
    )
    parser.add_argument(
        "--use_sparse_regularization",
        action="store_true",
        default=False,
        help="Whether to use sparse regularization",
    )
    parser.add_argument(
        "--do_training",
        action="store_true",
        default=True,
        help="Whether to train a model (Useful for printing pre-training sparsity)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Whether to log in wandb",
    )
    parser.add_argument(
        "--use_spm",
        action="store_true",
        default=False,
        help="Whether to use in sparse predictor mask",
    )
    parser.add_argument(
        "--use_relu",
        action="store_true",
        default=False,
        help="Whether to use ReLU as non-linear activation function",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=False,
        help="Whether to use ReLU as non-linear activation function",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        default=False,
        help="Whether to resume training from checkpoint",
    )
    parser.add_argument(
        "--sparse_model_dir",
        type=str,
        default=None,
        help="Whether to load fine-tuned sparse model from given directory",
    )
    parser.add_argument(
        "--ds_config_path",
        type=str,
        default="ds_config.json",
        help="Deepspeed config file path",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8096,
        help="maximum seq length of an input sequence",
    )
    parser.add_argument(
        "--plot_post_training_sparsity",
        action="store_true",
        default=False,
        help="Whether to plot activation sparsity after training",
    )
    parser.add_argument("--use_lora", action=BooleanOptionalAction)
    parser.set_defaults(use_lora=True)

    return parser.parse_args()
