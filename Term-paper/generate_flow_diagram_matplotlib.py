"""
Generate PNG flow diagram using matplotlib and networkx
This uses commonly available Python libraries
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Define node positions and properties
nodes = {
    # Data preprocessing
    'A': {'pos': (5, 19), 'text': 'Raw CSV Data', 'color': '#f0f0f0'},
    'B': {'pos': (5, 18), 'text': 'Data Loading & Preparation', 'color': '#f0f0f0'},
    'C': {'pos': (5, 17), 'text': 'Feature Extraction', 'color': '#f0f0f0'},
    'D': {'pos': (5, 16), 'text': 'Train/Test Split', 'color': '#f0f0f0'},
    'E': {'pos': (5, 15), 'text': 'StandardScaler Normalization', 'color': '#f0f0f0'},
    'F': {'pos': (5, 14), 'text': 'Sequence Generation', 'color': '#f0f0f0'},
    
    # Model architecture
    'G': {'pos': (5, 12.5), 'text': 'CNN-LSTM Model', 'color': '#e1f5ff'},
    'H': {'pos': (5, 11.5), 'text': 'CNN Block 1', 'color': '#fff3e0'},
    'I': {'pos': (5, 10.5), 'text': 'CNN Block 2', 'color': '#fff3e0'},
    'J': {'pos': (5, 9.5), 'text': 'LSTM Layer', 'color': '#f3e5f5'},
    'K': {'pos': (5, 8.5), 'text': 'Dense Layer', 'color': '#e8f5e9'},
    'L': {'pos': (5, 7.5), 'text': 'Output Layer', 'color': '#e8f5e9'},
    
    # Predictions and evaluation
    'M': {'pos': (5, 6), 'text': 'Predictions', 'color': '#f0f0f0'},
    'N': {'pos': (5, 5), 'text': 'Evaluation Metrics', 'color': '#f0f0f0'},
    
    # Metrics
    'O': {'pos': (2, 3), 'text': 'Confusion Matrix', 'color': '#ffe0e0'},
    'P': {'pos': (4, 3), 'text': 'ROC Curves', 'color': '#ffe0e0'},
    'Q': {'pos': (6, 3), 'text': 'PR Curves', 'color': '#ffe0e0'},
    'R': {'pos': (8, 3), 'text': 'F1 Scores', 'color': '#ffe0e0'},
}

# Define edges
edges = [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'),
    ('G', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'K'), ('K', 'L'), ('L', 'M'),
    ('M', 'N'), ('N', 'O'), ('N', 'P'), ('N', 'Q'), ('N', 'R')
]

# Draw nodes
for node_id, props in nodes.items():
    x, y = props['pos']
    box = FancyBboxPatch(
        (x - 1.2, y - 0.3), 2.4, 0.6,
        boxstyle="round,pad=0.1",
        edgecolor='#333333',
        facecolor=props['color'],
        linewidth=2,
        zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, props['text'], ha='center', va='center', 
            fontsize=10, fontweight='bold', zorder=3)

# Draw edges
for start, end in edges:
    x1, y1 = nodes[start]['pos']
    x2, y2 = nodes[end]['pos']
    
    arrow = FancyArrowPatch(
        (x1, y1 - 0.35), (x2, y2 + 0.35),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color='#555555',
        linewidth=2,
        zorder=1
    )
    ax.add_patch(arrow)

# Add title
ax.text(5, 19.8, 'CNN-LSTM Model Flow Diagram', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#f0f0f0', edgecolor='#333333', label='Data Processing'),
    mpatches.Patch(facecolor='#e1f5ff', edgecolor='#333333', label='Model'),
    mpatches.Patch(facecolor='#fff3e0', edgecolor='#333333', label='CNN Layers'),
    mpatches.Patch(facecolor='#f3e5f5', edgecolor='#333333', label='LSTM Layer'),
    mpatches.Patch(facecolor='#e8f5e9', edgecolor='#333333', label='Dense Layers'),
    mpatches.Patch(facecolor='#ffe0e0', edgecolor='#333333', label='Evaluation Metrics'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Save figure
plt.tight_layout()
output_file = 'model_flow_diagram.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[SUCCESS] Flow diagram saved as: {output_file}")
print(f"  Resolution: 300 DPI")
print(f"  Format: PNG")
plt.close()
