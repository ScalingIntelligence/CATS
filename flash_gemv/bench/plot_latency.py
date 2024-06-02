import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from XLSX file into DataFrame
df = pd.read_excel('mlp-latency.xlsx')

# Set the line type as scatter plot+line plot
sns.set_theme()
# Plot the data
sns.lineplot(data=df, x='Density', y='Dense', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Dense')
sns.lineplot(data=df, x='Density', y='Optimal', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Optimal')
sns.lineplot(data=df, x='Density', y='Baseline', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Baseline')
sns.lineplot(data=df, x='Density', y='Method1', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Method1')
sns.lineplot(data=df, x='Density', y='Method2', marker="o", errorbar=None)
# sns.scatterplot(data=df, x='Density', y='Method2')

# Set the y-axis name as "Latency (ms)"
plt.ylabel('Latency(ms)')
# Set the x-axis name as "Density (%)"
plt.xlabel('Density')
# Plot the legend
plt.legend(['Dense', 'Optimal', 'Baseline', 'Method1', 'Method2'])

# sns.move_legend(
#     ax, "lower center",
#     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
# )
# Save the plot to a PDF file
plt.savefig('latency.pdf')
# Save the plot as an svg
plt.savefig('latency.svg')