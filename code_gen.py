"""
Code Visualization Script for Code Description Section
Generate architecture diagrams and comparison charts
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Create output directory
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Figure 1: Feature Engineering Comparison Table
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Feature Engineering Pipeline Comparison', 
             fontsize=14, fontweight='bold', y=0.97)
ax.axis('tight')
ax.axis('off')

feature_comparison = [
    ['Aspect', 'RF Pipeline', 'XGB Pipeline', 'MLP Pipeline'],
    ['Title Processing', 'Mr, Mrs, Miss,\nMaster, Rare', 'Mr, Mrs, Miss,\nRare (merged)', 'Mr, Mrs, Miss,\nMaster, Royalty,\nOfficer, Rare'],
    ['Age Imputation', 'RF Regressor\n(2000 trees)', 'RF Regressor\n(2000 trees)', 'Grouped median\n(Title + Pclass)'],
    ['Categorical\nEncoding', 'Label encoding\n(category codes)', 'Label encoding +\nFrequency encoding', 'One-hot encoding\n(drop_first=True)'],
    ['Binning', 'None', 'AgeBin (5 bins)\nFareBin (quartiles)', 'AgeBin (5 bins)\nFareBin (5 bins)'],
    ['Interaction\nFeatures', 'None', 'Sex × Pclass\nPclass × AgeBin', 'None'],
    ['Scaling', 'None', 'None', 'StandardScaler\n(z-score norm.)'],
    ['Family Features', 'Family_Size\n(SibSp + Parch)', 'Family_Size\nIsAlone', 'FamilySize\nIsAlone\nTicketGroup'],
    ['Output Dims', '9 features', '14 features', '20+ features\n(after one-hot)'],
    ['Best For', 'Tree models', 'Gradient boosting', 'Neural networks']
]

table = ax.table(cellText=feature_comparison, cellLoc='left', loc='center',
                colWidths=[0.15, 0.28, 0.28, 0.28])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 3)

# Style header row
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#3498DB')
    cell.set_text_props(weight='bold', color='white')

# Style aspect column
for i in [2, 4, 6, 8]:
    cell = table[(i, 0)]
    cell.set_facecolor('#ECF0F1')
    cell.set_text_props(weight='bold')

# Color code pipelines
for row in [2, 4, 6, 8]:
    table[(row, 1)].set_facecolor('#E8F8F5')  # Light green for RF
    table[(row, 2)].set_facecolor('#FEF9E7')  # Light yellow for XGB
    table[(row, 3)].set_facecolor('#EBF5FB')  # Light blue for MLP

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.3)
plt.savefig(os.path.join(output_dir, 'feature_comparison_table.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Generated: feature_comparison_table.png")
plt.close()

# ============================================================
# Figure 2: Model Hyperparameter Summary
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.axis('tight')
ax.axis('off')

model_params = [
    ['Model', 'Key Hyperparameters', 'Tuning Method', 'Feature Set'],
    ['Random Forest', 'n_estimators=1000\nmin_samples_split=12\ncriterion=gini', 
     'Manual', 'RF (9 features)'],
    ['XGBoost', 'n_estimators=600\nlearning_rate=0.03\nmax_depth=3\nsubsample=0.9',
     'Manual +\nEarly Stopping', 'XGB (14 features)'],
    ['Gradient\nBoosting', 'n_estimators: [300,500,800]\nlearning_rate: [0.01,0.02,0.05]\nmax_depth: [3,4,5]',
     'GridSearchCV\n(3-fold CV)', 'RF (9 features)'],
    ['Logistic\nRegression', 'max_iter=5000\nclass_weight=balanced\nsolver=liblinear',
     'Default', 'RF (9 features)'],
    ['SVM (RBF)', 'C: [0.5,1.0,2.0,5.0]\ngamma: [scale,0.05,0.1]\nclass_weight=balanced',
     'GridSearchCV\n(3-fold CV)', 'RF (9 features)'],
    ['KNN', 'n_neighbors: [7,11,15,21]\nweights: [uniform,distance]\nmetric=minkowski',
     'GridSearchCV\n(3-fold CV)', 'RF (9 features)'],
    ['MLP', 'hidden_dims=(256,128,64)\ndropout=0.35\nlr=1e-3\nweight_decay=1e-4',
     'Manual +\nEarly Stopping', 'MLP (20+ features)'],
]

table = ax.table(cellText=model_params, cellLoc='left', loc='center',
                colWidths=[0.16, 0.42, 0.18, 0.24])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 3)

# Style header
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#E74C3C')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors (data rows only)
for row in range(1, len(model_params), 2):
    for col in range(4):
        table[(row, col)].set_facecolor('#F8F9F9')

fig.suptitle('Model Architecture and Hyperparameter Configuration', 
             fontsize=14, fontweight='bold', y=0.965)
plt.tight_layout(rect=[0, 0.02, 1, 0.995], pad=0.0, h_pad=0.0)
plt.savefig(os.path.join(output_dir, 'model_hyperparameters.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Generated: model_hyperparameters.png")
plt.close()

# ============================================================
# Figure 3: Training Configuration Summary
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

training_config = [
    ['Configuration', 'Value', 'Purpose'],
    ['Batch Size', '128', 'Balance between speed and stability'],
    ['Max Epochs', '250', 'Sufficient for convergence with early stopping'],
    ['Learning Rate', '1e-3', 'Adam default, suitable for small networks'],
    ['Weight Decay', '1e-4', 'L2 regularization for neural networks'],
    ['Validation Ratio', '0.2 (20%)', 'Standard train-val split'],
    ['Early Stop Patience', '30 epochs', 'Prevent overfitting'],
    ['Random Seeds', '[45, 2025, 777]', 'Multi-seed validation for robustness'],
    ['Mixed Precision', 'FP16 (CUDA)', 'Accelerate training on GPU'],
    ['Gradient Clipping', 'max_norm=1.0', 'Prevent exploding gradients'],
    ['LR Scheduler', 'ReduceLROnPlateau', 'Adaptive learning rate (factor=0.5, patience=6)'],
]

table = ax.table(cellText=training_config, cellLoc='left', loc='center',
                colWidths=[0.17, 0.19, 0.39])
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.5)

# Style header
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#9B59B6')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for row in range(1, len(training_config)):
    if row % 2 == 1:
        for col in range(3):
            table[(row, col)].set_facecolor('#F4ECF7')

plt.title('Neural Network Training Configuration', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_config.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Generated: training_config.png")
plt.close()

# ============================================================
# Figure 4: Architecture Diagram (MLP Network)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw network layers
layer_positions = [
    (1, 5, 'Input\n20+ features', '#3498DB'),
    (2.5, 6.5, 'Linear(256)', '#2ECC71'),
    (2.5, 5.5, 'BatchNorm1d', '#F39C12'),
    (2.5, 4.5, 'ReLU', '#E74C3C'),
    (2.5, 3.5, 'Dropout(0.35)', '#9B59B6'),
    
    (4.5, 6.5, 'Linear(128)', '#2ECC71'),
    (4.5, 5.5, 'BatchNorm1d', '#F39C12'),
    (4.5, 4.5, 'ReLU', '#E74C3C'),
    (4.5, 3.5, 'Dropout(0.35)', '#9B59B6'),
    
    (6.5, 6.5, 'Linear(64)', '#2ECC71'),
    (6.5, 5.5, 'BatchNorm1d', '#F39C12'),
    (6.5, 4.5, 'ReLU', '#E74C3C'),
    (6.5, 3.5, 'Dropout(0.35)', '#9B59B6'),
    
    (8.5, 5, 'Linear(2)', '#2ECC71'),
    (10.2, 5, 'Output\n[Died, Survived]', '#3498DB'),
]

for x, y, label, color in layer_positions:
    bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, 
            fontweight='bold', color='white', bbox=bbox)

# Draw connections between blocks (horizontal arrows)
block_connections = [
    (1.5, 5, 2, 6.5),      # Input to first block
    (3, 3.5, 4, 6.5),      # First block to second block
    (5, 3.5, 6, 6.5),      # Second block to third block
    (7, 3.5, 8.1, 5),      # Third block to Linear(2) layer
    (8.85, 5, 9.7, 5),     # Linear(2) to Output
]

for x1, y1, x2, y2 in block_connections:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=3, color='#2C3E50', alpha=1.0, 
                               connectionstyle='arc3,rad=0'))

# Draw connections within each block (vertical arrows)
# First block: Linear(256) -> BatchNorm -> ReLU -> Dropout
vertical_connections = [
    # Block 1
    (2.5, 6.2, 2.5, 5.8),   # Linear(256) -> BatchNorm1d
    (2.5, 5.2, 2.5, 4.8),   # BatchNorm1d -> ReLU
    (2.5, 4.2, 2.5, 3.8),   # ReLU -> Dropout(0.35)
    # Block 2
    (4.5, 6.2, 4.5, 5.8),   # Linear(128) -> BatchNorm1d
    (4.5, 5.2, 4.5, 4.8),   # BatchNorm1d -> ReLU
    (4.5, 4.2, 4.5, 3.8),   # ReLU -> Dropout(0.35)
    # Block 3
    (6.5, 6.2, 6.5, 5.8),   # Linear(64) -> BatchNorm1d
    (6.5, 5.2, 6.5, 4.8),   # BatchNorm1d -> ReLU
    (6.5, 4.2, 6.5, 3.8),   # ReLU -> Dropout(0.35)
]

for x1, y1, x2, y2 in vertical_connections:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495E', alpha=0.8))

# Add annotations
ax.text(5, 8.5, 'TitanicMLP Architecture: 256 → 128 → 64 → 2', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(5, 0.5, 'Each hidden block: Linear → BatchNorm → ReLU → Dropout', 
        ha='center', fontsize=10, style='italic', color='#34495E')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mlp_architecture.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Generated: mlp_architecture.png")
plt.close()

# ============================================================
# Figure 5: Missing Value Statistics
# ============================================================
data_dir = 'data'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

missing_stats = train_df.isnull().sum()
missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
missing_pct = (missing_stats / len(train_df) * 100).round(2)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(missing_stats)), missing_stats.values, color='#E74C3C', alpha=0.7)
ax.set_yticks(range(len(missing_stats)))
ax.set_yticklabels(missing_stats.index)
ax.set_xlabel('Number of Missing Values', fontsize=12)
ax.set_title('Missing Value Distribution in Training Data', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (count, pct) in enumerate(zip(missing_stats.values, missing_pct.values)):
    ax.text(count + 10, i, f'{count} ({pct}%)', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'missing_value_stats.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Generated: missing_value_stats.png")
plt.close()

print("\n✓ All code visualization plots generated successfully!")
print(f"✓ Saved to '{output_dir}/' directory")