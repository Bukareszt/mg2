import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
labels = ['Graph (Hidden Dim 512)', 'Graph (Hidden Dim 256)', 'Layer 8', 'Layer 9', 'Layer 10',
          'Layer 11', 'Layer 12', 'Layer 13', 'Layer 14', 'Layer 15']
values = [9.32, 11.69, 16.55, 16.99, 17.66, 16.28, 17.15, 16.27, 15.23, 15.43]

# Define types: Graph vs Hidden State
types = ['Graph'] * 2 + ['Hidden State'] * 8

# Create DataFrame
df = pd.DataFrame({
    'Model': labels,
    'MAE': values,
    'Type': types
})

# Sort by MAE
df_sorted = df.sort_values('MAE', ascending=True)

# Set Seaborn style
sns.set(style="whitegrid")

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_sorted, y='Model', x='MAE', hue='Type', dodge=False, palette='deep')

# Annotate bars with values
for i, (mae, model) in enumerate(zip(df_sorted['MAE'], df_sorted['Model'])):
    ax.text(mae + 0.3, i, f'{mae:.2f}', va='center')

# Titles and labels
plt.title('MAE Comparison: Graph vs TRAIL', fontsize=14)
plt.xlabel('Mean Absolute Error (MAE)')
plt.legend(title='Source')

plt.tight_layout()
plt.savefig('mae_comparison_hidden_states.png', dpi=300)
plt.savefig('mae_comparison_hidden_states.pdf')
plt.show()
