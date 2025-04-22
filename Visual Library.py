# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Sample Data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 3, 5, 7, 11]
})
# 1. Matplotlib Plots
plt.figure(figsize=(5, 3))
plt.plot(df['x'], df['y'], marker='o', label='Line')
plt.title('Matplotlib Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3))
plt.bar(df['x'], df['y'], color='skyblue')
plt.title('Matplotlib Bar Plot')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3))
plt.scatter(df['x'], df['y'], color='red')
plt.title('Matplotlib Scatter Plot')
plt.tight_layout()
plt.show()

# 🔹 2. Seaborn Plots
sns.set(style='darkgrid')

sns.lineplot(data=df, x='x', y='y', marker='o')
plt.title('Seaborn Line Plot')
plt.show()

sns.barplot(data=df, x='x', y='y', palette='Blues_d')
plt.title('Seaborn Bar Plot')
plt.show()

sns.scatterplot(data=df, x='x', y='y', color='green')
plt.title('Seaborn Scatter Plot')
plt.show()
#  3. Plotly Plots
fig = px.line(df, x='x', y='y', markers=True, title='Plotly Line Plot')
fig.show()

fig = px.bar(df, x='x', y='y', title='Plotly Bar Plot')
fig.show()

fig = px.scatter(df, x='x', y='y', title='Plotly Scatter Plot')
fig.show()

#  4. Pandas Built-in Plots
df.plot(kind='line', x='x', y='y', marker='o', title='Pandas Line Plot')
plt.grid(True)
plt.show()

df.plot(kind='bar', x='x', y='y', color='orange', title='Pandas Bar Plot')
plt.grid(True)
plt.show()

df.plot(kind='scatter', x='x', y='y', color='purple', title='Pandas Scatter Plot')
plt.grid(True)
plt.show()
