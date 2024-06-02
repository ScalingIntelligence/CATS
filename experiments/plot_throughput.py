import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from scipy.stats.mstats import gmean

keys = ["Llama2_base", "Llama2_sparse", "Mistral_base", "Mistral_sparse"]
key = ""
labels = [256, 512, 1024, 2048]
throughput_dict = {k: {label: [] for label in labels} for k in keys}

for idx in range(2):
    filepath = os.path.join(os.getenv("CATS_RESPATH", "results"), "throughput.csv")

    # Open the file and read data
    with open(filepath, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)

        # Extract data from each row
        for row in csv_reader:
            if row:  # Check if the row is not empty
                if str(row[0]) in keys:
                    key = str(row[0])
                    continue

                if key == "":
                    continue
                generation_length = int(row[1])
                throughput_dict[key][generation_length].append(float(row[3]))

    for model in ["Llama2", "Mistral"]:
        # Creating a DataFrame from the lists
        labels = [256, 512, 1024, 2048]
        dense_key = model + "_base"
        sparse_key = model + "_sparse"

        dense = [gmean(throughput_dict[dense_key][label]) for label in labels]
        sparse = [gmean(throughput_dict[sparse_key][label]) for label in labels]

        # Create a DataFrame
        df = pd.DataFrame(
            list(zip(dense, sparse)),
            index=labels,
            columns=["Dense", "CATS-with-Custom-Kernel"],
        )

        # Reset index to get the labels column for plotting
        df_reset = df.reset_index().rename(columns={"index": "Labels"})

        # Melt the DataFrame for seaborn
        df_melted = df_reset.melt(id_vars="Labels", var_name="Type", value_name="Value")

        legend_properties = {"weight": "bold", "size": "xx-large"}

        # Plot
        sns.set_theme()
        plt.figure(figsize=(8, 8))
        sns.barplot(x="Labels", y="Value", hue="Type", data=df_melted)
        # plt.title('Dense vs Sparse values by Label')
        plt.xlabel("Generation Length (Tokens)", fontsize="xx-large", fontweight="bold")
        plt.ylabel("Generation Throughput (Tokens/s)", fontsize="xx-large", fontweight="bold")
        plt.legend(loc="lower left", prop=legend_properties)
        plt.yticks(fontsize="xx-large", fontweight="bold")
        plt.xticks(fontsize="xx-large", fontweight="bold")  # Rotating the x-axis labels for better readability
        plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping

        save_dir = os.getenv("CATS_RESPATH", "results")
        plt.savefig(os.path.join(save_dir, f"figures/fig3: e2e_gen_{model}_fp32.pdf"))
        print((os.path.join(save_dir, f"figures/fig3: e2e_gen_{model}_fp32.pdf")))
