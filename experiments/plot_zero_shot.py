import os
import json
import matplotlib.pyplot as plt
import re


def get_mean_accuracy(directory, filename="results.json"):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
            acc_sum = 0
            count = 0
            # print("\n\n---directory---", directory)
            for key, value in data["results"].items():
                # print(key, value["acc,none"])
                acc_sum += value["acc,none"]
                count += 1
            avg_acc = acc_sum / count if count > 0 else 0
            return avg_acc
    return 0


def plot(
    model_name,
    parent_directories,
    base_model_name,
    sparse_model_name,
    relu_model_name,
):
    directories = []

    for parent_directory in parent_directories:
        for entry in os.scandir(parent_directory):
            if entry.is_dir() and any(
                # model_name in entry.name for model_name in [base_model_name, sparse_model_name, relu_model_name]
                re.match(model_name, entry.name)
                for model_name in [base_model_name, sparse_model_name, relu_model_name]
            ):
                directories.append(entry.path)

    # print("Directories:", directories)
    model_accuracies = {}
    base_model_accuracy = 0
    print(directories)

    for directory in directories:
        if os.path.exists(os.path.join(directory, "results.json")):
            avg_acc = get_mean_accuracy(directory)

            if base_model_name in directory:
                base_model_accuracy = avg_acc
                # print("Base model accuracy:", base_model_accuracy)
            else:
                parts = directory.split("_")
                if "relu" in directory:
                    model_type = "ReLUfication"
                    print("ReLU!!!", directory)
                elif "sparse" not in directory:
                    model_type = "Base"
                else:
                    match = re.search(r"(\d+)p", directory)
                    if match:
                        sparsity = match.group(1)
                    else:
                        sparsity = 0
                    model_type = f"CATS {sparsity}%"
                    print(model_type)
                step = parts[-1].replace("steps2", "").replace("steps", "")
                step = int(step) if step != "" else 0
                if step == 100:
                    continue
                model_accuracies[model_type] = model_accuracies.get(model_type, {})
                model_accuracies[model_type][step] = avg_acc
    print(model_accuracies)

    plt.figure(figsize=(8, 6))

    model_types = list(sorted(model_accuracies.keys()))

    for model_type in model_types:
        accuracies = model_accuracies[model_type]
        steps = sorted(accuracies.keys())
        values = [accuracies[step] for step in steps]
        color = "#227CF6"
        alpha = 1.0
        xytext = (0, 10)
        if "50" in model_type:
            xytext = (0, 14)
        elif "70" in model_type:
            alpha = 0.6
            xytext = (0, -14)
        elif "90" in model_type:
            alpha = 0.3
            xytext = (0, -14)
        elif "ReLUfication" == model_type:
            alpha = 1.0
            color = "#FC778D"

        plt.plot(
            steps,
            values,
            label=model_type.capitalize(),
            marker="o",
            linewidth=2,
            color=color,
            alpha=alpha,
        )
        for step, accuracy in accuracies.items():
            if int(step) == 1 and "50" in model_type:
                print(model_type)
                xytext = (0, -10)
            plt.annotate(
                f"{accuracy:.4f}",
                (step, accuracy),
                textcoords="offset points",
                xytext=xytext,
                ha="center",
            )

    if base_model_accuracy > 0:
        plt.plot(
            steps,
            [base_model_accuracy] * len(steps),
            label="Mistral 7B",
            linestyle="--",
            color="gray",
            linewidth=2,
        )
        plt.annotate(
            f"{base_model_accuracy:.4f}",
            (steps[0], base_model_accuracy),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )
    plt.xlabel("Finetuning Steps (batch size 16)")
    plt.ylabel("Average 0-Shot Accuracy")
    plt.legend(fontsize=12, labels=model_types + ["Mistral 7B"])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.tight_layout()

    result_root = os.getenv("CATS_RESPATH", "results")
    dirname = os.path.join(result_root, "sparse_silu/figures")
    print(">>>>>>>>>>>>DIRNAME: ", dirname)
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/{model_name}_training_steps_vs_zero_shot_accuracy.png", dpi=300)
    plt.show()
    print(f"{dirname}/{model_name}_training_steps_vs_zero_shot_accuracy.png")


if __name__ == "__main__":
    # LLAMA
    # parent_directories = [
    #     "output",
    # ]
    # plot(
    #     "llama",
    #     parent_directories,
    #     base_model_name="llama_7b_hf",
    #     sparse_model_name="llama_sparse.*2",
    #     relu_model_name="relu_llama.*",
    # )

    # MISTRAL
    parent_directories = [
        "output",
    ]
    plot(
        "mistral",
        parent_directories,
        base_model_name="mistral_base",
        sparse_model_name="mistral_sparse_.*",
        relu_model_name="relu_mistral_.*",
    )
