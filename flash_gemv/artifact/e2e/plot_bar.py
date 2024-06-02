import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


labels = [256, 512, 1024, 2048]
# Data
# Mistral bf16
# dense = [33.12512583960324, 33.2263687038353, 32.82409097744316, 32.45005347445845]
# sparse = [37.6527755389042, 38.87914322139526, 38.314685128222294, 38.07380589970877]
# Mistral fp32
dense = [23.03527826557369,22.996964395110663,22.76527968326422,22.123762001317353]
sparse = [27.89374595779419,27.892038365294187,27.597023881676318,26.86496371105544]
# Llama fp32
# dense = [22.945580978071966,21.985762485687705,21.878725073740625,20.145874435194834]
# sparse = [27.667572259727287,26.062073473397138,25.213137968323263,23.848038186808182]

# Create a DataFrame
df = pd.DataFrame(list(zip(dense, sparse)), index=labels, columns=['Dense', 'CATS-with-Custom-Kernel'])

# Reset index to get the labels column for plotting
df_reset = df.reset_index().rename(columns={'index': 'Labels'})

# Melt the DataFrame for seaborn
df_melted = df_reset.melt(id_vars='Labels', var_name='Type', value_name='Value')

legend_properties = {'weight':'bold', 'size': 'xx-large'}


# Plot
sns.set_theme()
plt.figure(figsize=(6, 6))
sns.barplot(x='Labels', y='Value', hue='Type', data=df_melted)
# plt.title('Dense vs Sparse values by Label')
plt.xlabel('Generation Length (Tokens)', fontsize='xx-large', fontweight='bold')
plt.ylabel('Generation Throughput (Tokens/s)', fontsize='xx-large', fontweight='bold')
plt.legend(loc='lower left', prop=legend_properties)
plt.yticks(fontsize='xx-large', fontweight='bold')
plt.xticks(fontsize='xx-large', fontweight='bold')  # Rotating the x-axis labels for better readability
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping

plt.savefig('e2e_gen_mistral_fp32.pdf')