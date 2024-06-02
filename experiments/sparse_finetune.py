from instruct_tuning import train, parse_args
from utils.constants import COLA, WIC, BOOLQ, SST2


def main(targeted_sparsity, datset):
    args = parse_args()
    args.dataset_type = datset
    args.targeted_sparsity = targeted_sparsity
    args.use_sparse_model = True
    args.model_save = True
    args.set_sparsity_aware_threshold = True
    args.print_sparsity = True
    args.use_lora = True
    args.num_epochs = 8
    args.use_spm = False
    args.model_name = "Sparse_Mistral"
    args.use_wandb = True
    # args.gradient_checkpointing = True
    print(args)
    train(args)


if __name__ == "__main__":
    for dataset in [COLA, WIC, BOOLQ, SST2]:
        for sparsity in [0.99, 0.95, 0.9, 0.85, 0.75, 0.65, 0.5]:
            main(sparsity, dataset)
