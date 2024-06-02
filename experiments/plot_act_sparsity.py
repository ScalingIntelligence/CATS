import matplotlib.pyplot as plt
import json
import re
import os


def plot_sparsities(
    log_dir: str,
    figure_dir: str,
):
    for target_model_type in ["llama", "mistral"]:
        cats_50p = {}
        cats_70p = {}
        cats_90p = {}
        for filename in os.listdir(log_dir):
            if filename.endswith(".json"):
                # Extract the percentage value
                percentage_match = re.search(r"(\d+)p", filename)
                percentage = percentage_match.group(1) if percentage_match else None

                # Extract the model name
                model_match = re.search(r"(llama|mistral)", filename)
                model_name = model_match.group(1) if model_match else None

                if model_name != target_model_type:
                    continue

                with open(os.path.join(log_dir, filename), "r") as file:
                    data = json.load(file)

                if percentage is None:
                    continue
                if int(percentage) == 50:
                    cats_50p = data["sparsity_list"]
                elif int(percentage) == 70:
                    cats_70p = data["sparsity_list"]
                elif int(percentage) == 90:
                    cats_90p = data["sparsity_list"]

        if target_model_type == "mistral":
            title = "Mistral-7B"
        elif target_model_type == "llama":
            title = "Llama2-7B"
        else:
            raise NotImplementedError(f"{target_model_type} isn't implemented")
        plot_model_sparsity(cats_50p, cats_70p, cats_90p, title, figure_dir)

    assert (
        len(cats_50p) == len(cats_70p) == len(cats_90p)
    ), "All experiments should have been run at 50%, 70%, and 90% sparsity level"


def plot_model_sparsity(
    cats_50_sparsity, cats_70_sparsity, cats_90_sparsity, model_name: str = "Mistral-7B", figure_dir: str = None
):
    assert (
        len(cats_50_sparsity) == len(cats_70_sparsity) == len(cats_90_sparsity)
    ), "The lengths of cats_50_sparsity, cats_70_sparsity, and cats_90_sparsity must be the same."
    # Data setup
    layer_indices = list(range(len(cats_50_sparsity)))

    labels = ["CATS 50%", "CATS 70%", "CATS 90%"]
    average_sparsity = [0, 0, 0]
    color = "#227CF6"
    sparsity_per_layer = [cats_50_sparsity, cats_70_sparsity, cats_90_sparsity]

    for idx in range(3):
        average_sparsity[idx] = sum(sparsity_per_layer[idx]) / len(sparsity_per_layer[idx])
    markers = ["o", "s", "D"]

    # Plotting
    plt.figure(figsize=(10, 8))

    for i in range(3):
        plt.plot(
            layer_indices,
            sparsity_per_layer[i],
            label=labels[i],
            marker=markers[i],
            color=color,
            alpha=1 - i * 0.35,
        )
        plt.hlines(
            average_sparsity[i],
            layer_indices[0],
            layer_indices[-1],
            color=color,
            alpha=1 - i * 0.35,
            linestyles="dashed",
            label=f"{labels[i]} Average Sparsity",
        )
        plt.annotate(
            f"{average_sparsity[i]:.2f}%",
            (layer_indices[0], average_sparsity[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.xlabel("Layer Index")
    plt.ylabel("Post-Training Sparsity (%)")
    plt.legend()
    plt.title(f"Activation Sparsity of {model_name}")

    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(f"{figure_dir}/fig5:{model_name}_post_training_sp.png")
    plt.close()


if __name__ == "__main__":
    result_dir = os.getenv("CATS_RESPATH", "")
    log_dir = os.path.join(result_dir, "general_finetuning")
    figure_dir = os.path.join(result_dir, "activation_sparsity")

    plot_sparsities(log_dir, figure_dir)
