import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Load data from XLSX file into DataFrame
# df = pd.read_excel('llama7B-mlp-l40s-fp16.xlsx')
# name = 'llama7B-mlp-full-fp32'
name = 'mistral7B-mlp-full-fp32'
df = pd.read_excel(f'{name}.xlsx')
# Set the line type as scatter plot+line plot
sns.set_theme()
plt.figure(figsize=(10, 6))
ax = plt.gca()  # Get the current Axes instance
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # Set tick step to 2
# Plot the data
l1 = sns.lineplot(data=df, x='Sparsity', y='Dense', marker="o", errorbar=None, linewidth=3)
# sns.scatterplot(data=df, x='Density', y='Dense')
l2 = sns.lineplot(data=df, x='Sparsity', y='Optimal', marker="o", errorbar=None, linewidth=3)
# sns.scatterplot(data=df, x='Density', y='Optimal')
l3 = sns.lineplot(data=df, x='Sparsity', y='CATS-with-Custom-Kernel', marker="o", errorbar=None, linewidth=3)
# l4 = sns.lineplot(data=df, x='Sparsity', y='Baseline', marker="o", errorbar=None, linewidth=3)
l4 = sns.lineplot(data=df, x='Sparsity', y='Method1', marker="o", errorbar=None, linewidth=3)
# sns.lineplot(data=df, x='Density', y='Baseline', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Baseline')
# sns.lineplot(data=df, x='Density', y='Method1', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Method1')
# sns.lineplot(data=df, x='Density', y='Method2', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Method2')

legend_properties = {'weight':'bold', 'size': 'x-large'}
plt.legend(['Dense', 'Optimal', 'CATS-with-Custom-Kernel', 'Baseline', 'Method1'], loc='lower left', prop=legend_properties)

# Set the y-axis name as "Latency (ms)"
plt.ylabel('Latency(ms)', fontsize='xx-large', fontweight='bold')
# Set the x-axis name as "Density (%)"
plt.xlabel('Sparsity', fontsize='xx-large', fontweight='bold')
# Plot the legend

# sns.move_legend(
#     ax, "lower center",
#     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
# )
# Save the plot to a PDF file
# plt.savefig('llama7B-latency-l40s-fp16.pdf')
plt.yticks(fontsize='xx-large', fontweight='bold')
plt.xticks(fontsize='xx-large', fontweight='bold')
plt.tight_layout()
plt.setp(l1.lines, markersize=10)
plt.setp(l2.lines, markersize=10)
plt.setp(l3.lines, markersize=10)
plt.setp(l4.lines, markersize=10)
plt.tight_layout() 

plt.savefig(f'{name}.pdf')
# Save the plot as an svg
# plt.savefig('llama7B-latency-l40s-fp16.svg')