#!/usr/bin/env python3
"""
Generate comprehensive comparison visualizations for all experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Create output directory
output_dir = Path('/path/to/puzzle-moe/docs/figs/plots/final_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

# Experimental results
experiments = {
    'Baseline\n(4-exp, λ_LB=0.01)': {
        'macro': 0.8784,
        'NORM': 0.9363, 'MI': 0.8959, 'STTC': 0.9026, 'CD': 0.9137, 'HYP': 0.7436
    },
    'Step 1\n(3-exp, λ_LB=0.005)': {
        'macro': 0.8577,
        'NORM': 0.9391, 'MI': 0.9036, 'STTC': 0.9075, 'CD': 0.9052, 'HYP': 0.6330
    },
    'Step 2\n(4-exp, new)': {
        'macro': 0.8556,
        'NORM': 0.9399, 'MI': 0.9005, 'STTC': 0.9079, 'CD': 0.9112, 'HYP': 0.6186
    },
    'Step 4\n(No SSL)': {
        'macro': 0.8567,
        'NORM': 0.9353, 'MI': 0.9030, 'STTC': 0.9082, 'CD': 0.9052, 'HYP': 0.6317
    },
    'Ensemble\n(Average)': {
        'macro': 0.8735,
        'NORM': 0.9415, 'MI': 0.9053, 'STTC': 0.9111, 'CD': 0.9135, 'HYP': 0.6963
    }
}

classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
model_names = list(experiments.keys())

# ============================================================================
# Figure 1: Macro AUROC Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

macro_scores = [experiments[model]['macro'] for model in model_names]
colors = ['#2ecc71' if score == max(macro_scores) else '#3498db' for score in macro_scores]

bars = ax.bar(range(len(model_names)), macro_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, macro_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Macro AUROC', fontsize=13, fontweight='bold')
ax.set_title('Macro AUROC Comparison Across All Experiments', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim([0.84, 0.89])
ax.axhline(y=0.8784, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Best Model')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'macro_auroc_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'macro_auroc_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: macro_auroc_comparison.png/pdf")
plt.close()

# ============================================================================
# Figure 2: Per-Class AUROC Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Create matrix
data = np.array([[experiments[model][cls] for cls in classes] for model in model_names])

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=0.95)

# Set ticks
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(model_names)))
ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax.set_yticklabels(model_names, fontsize=11)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('AUROC', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(model_names)):
    for j in range(len(classes)):
        text = ax.text(j, i, f'{data[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('Per-Class AUROC Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'per_class_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'per_class_heatmap.pdf', bbox_inches='tight')
print(f"✓ Saved: per_class_heatmap.png/pdf")
plt.close()

# ============================================================================
# Figure 3: Radar Chart - Multi-Model Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot each model
colors_radar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for idx, (model, color) in enumerate(zip(model_names, colors_radar)):
    values = [experiments[model][cls] for cls in classes]
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax.set_ylim(0.6, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.60', '0.70', '0.80', '0.90', '1.00'], fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title('Multi-Model Comparison: Per-Class AUROC', 
             fontsize=14, fontweight='bold', pad=30, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'radar_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: radar_comparison.png/pdf")
plt.close()

# ============================================================================
# Figure 4: Grouped Bar Chart - Per-Class Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(classes))
width = 0.15

for i, (model, color) in enumerate(zip(model_names, colors_radar)):
    values = [experiments[model][cls] for cls in classes]
    offset = (i - 2) * width
    bars = ax.bar(x + offset, values, width, label=model, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
ax.set_title('Per-Class AUROC: Detailed Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax.set_ylim([0.6, 1.0])
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'grouped_bar_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'grouped_bar_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: grouped_bar_comparison.png/pdf")
plt.close()

# ============================================================================
# Figure 5: SSL Impact Comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Macro AUROC comparison
ssl_models = ['Baseline\n(4-exp, λ_LB=0.01)', 'Step 4\n(No SSL)']
ssl_scores = [experiments[model]['macro'] for model in ssl_models]
colors_ssl = ['#2ecc71', '#e74c3c']

bars1 = ax1.bar(range(len(ssl_models)), ssl_scores, color=colors_ssl, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, score in zip(bars1, ssl_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

improvement = ((ssl_scores[0] - ssl_scores[1]) / ssl_scores[1]) * 100
ax1.text(0.5, 0.87, f'+{improvement:.2f}%', ha='center', fontsize=14, 
         fontweight='bold', color='green', transform=ax1.transAxes)

ax1.set_ylabel('Macro AUROC', fontsize=13, fontweight='bold')
ax1.set_title('Impact of SSL Pre-training on Macro AUROC', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(range(len(ssl_models)))
ax1.set_xticklabels(['With SSL\nPre-training', 'Without SSL\nPre-training'], fontsize=11)
ax1.set_ylim([0.84, 0.89])
ax1.grid(axis='y', alpha=0.3)

# Per-class comparison
x = np.arange(len(classes))
width = 0.35

with_ssl = [experiments['Baseline\n(4-exp, λ_LB=0.01)'][cls] for cls in classes]
without_ssl = [experiments['Step 4\n(No SSL)'][cls] for cls in classes]

bars2_1 = ax2.bar(x - width/2, with_ssl, width, label='With SSL', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2_2 = ax2.bar(x + width/2, without_ssl, width, label='Without SSL', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_ylabel('AUROC', fontsize=13, fontweight='bold')
ax2.set_title('SSL Impact: Per-Class Performance', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(classes, fontsize=11, fontweight='bold')
ax2.set_ylim([0.6, 1.0])
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'ssl_impact_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'ssl_impact_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: ssl_impact_comparison.png/pdf")
plt.close()

# ============================================================================
# Figure 6: HYP Performance Focus (Rare Class)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

hyp_scores = [experiments[model]['HYP'] for model in model_names]
colors_hyp = ['#e74c3c' if score < 0.65 else '#f39c12' if score < 0.70 else '#2ecc71' for score in hyp_scores]

bars = ax.bar(range(len(model_names)), hyp_scores, color=colors_hyp, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, score in zip(bars, hyp_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('AUROC (HYP Class)', fontsize=13, fontweight='bold')
ax.set_title('Performance on Rare Class (HYP) - Critical Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim([0.6, 0.8])
ax.axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Acceptable Threshold')
ax.axhline(y=0.7436, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Best Performance')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=10)

# Add annotation
best_idx = np.argmax(hyp_scores)
ax.annotate('Best for\nRare Class!', 
            xy=(best_idx, hyp_scores[best_idx]), 
            xytext=(best_idx + 0.3, hyp_scores[best_idx] + 0.03),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig(output_dir / 'hyp_rare_class_focus.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'hyp_rare_class_focus.pdf', bbox_inches='tight')
print(f"✓ Saved: hyp_rare_class_focus.png/pdf")
plt.close()

print(f"\n{'='*80}")
print("✅ All comparison visualizations generated successfully!")
print(f"{'='*80}")
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  1. macro_auroc_comparison.png/pdf")
print("  2. per_class_heatmap.png/pdf")
print("  3. radar_comparison.png/pdf")
print("  4. grouped_bar_comparison.png/pdf")
print("  5. ssl_impact_comparison.png/pdf")
print("  6. hyp_rare_class_focus.png/pdf")
