#!/usr/bin/env python3
"""
Generate comparison visualizations: Before vs After Load Balancing Loss.
Shows how we solved the expert collapse problem.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
})

# Data from analysis reports
before_data = {
    'expert_usage': [0.0110, 0.0070, 0.9821],
    'sample_counts': [34, 9, 2137],
    'percentages': [1.6, 0.4, 98.0],
    'entropy': 0.0353,
    'class_routing': {
        'NORM': [0.001, 0.001, 0.998],
        'MI': [0.032, 0.004, 0.964],
        'STTC': [0.011, 0.002, 0.987],
        'CD': [0.043, 0.004, 0.954],
        'HYP': [0.013, 0.002, 0.985],
    }
}

after_data = {
    'expert_usage': [0.3333, 0.2619, 0.4048],
    'sample_counts': [752, 384, 1044],
    'percentages': [34.5, 17.6, 47.9],
    'entropy': 0.8819,
    'class_routing': {
        'NORM': [0.188, 0.200, 0.612],
        'MI': [0.488, 0.188, 0.324],
        'STTC': [0.447, 0.195, 0.358],
        'CD': [0.394, 0.234, 0.372],
        'HYP': [0.430, 0.223, 0.347],
    }
}

output_dir = Path('docs/figs/plots/collapse_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

# Color scheme
colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
expert_names = ['Expert 1', 'Expert 2', 'Expert 3']


def plot_expert_distribution():
    """Compare expert utilization distribution before/after."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Before
    ax = axes[0]
    bars = ax.bar(expert_names, before_data['percentages'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=33.33, color='gray', linestyle='--', linewidth=1, label='Ideal (33.3%)')
    ax.set_ylabel('Sample Percentage (%)')
    ax.set_title('Before: Expert Collapse (Entropy=0.035)', fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(loc='upper left')
    
    # Add percentage labels on bars
    for i, (bar, pct, count) in enumerate(zip(bars, before_data['percentages'], before_data['sample_counts'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{pct:.1f}%\n({count} samples)',
                ha='center', va='bottom', fontsize=9)
    
    # Add warning box
    ax.text(0.5, 85, '⚠️ Expert Collapse!\n98% → Expert 3',
            ha='center', va='center', fontsize=11, color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # After
    ax = axes[1]
    bars = ax.bar(expert_names, after_data['percentages'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=33.33, color='gray', linestyle='--', linewidth=1, label='Ideal (33.3%)')
    ax.set_ylabel('Sample Percentage (%)')
    ax.set_title('After: Balanced Routing (Entropy=0.882)', fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(loc='upper left')
    
    # Add percentage labels on bars
    for i, (bar, pct, count) in enumerate(zip(bars, after_data['percentages'], after_data['sample_counts'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{pct:.1f}%\n({count} samples)',
                ha='center', va='bottom', fontsize=9)
    
    # Add success box
    ax.text(0.5, 85, '✅ Balanced!\nAll experts utilized',
            ha='center', va='center', fontsize=11, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'expert_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'expert_distribution_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'expert_distribution_comparison.png'}")
    plt.close()


def plot_entropy_comparison():
    """Show entropy improvement."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    entropies = [before_data['entropy'], after_data['entropy'], 1.10]  # 1.10 is max for 3 experts
    labels = ['Before\n(Collapse)', 'After\n(Load Balance)', 'Maximum\nPossible']
    colors_bars = ['#e74c3c', '#2ecc71', '#95a5a6']
    
    bars = ax.bar(labels, entropies, color=colors_bars, alpha=0.7, edgecolor='black', width=0.6)
    ax.set_ylabel('Routing Entropy', fontsize=13)
    ax.set_title('Routing Diversity: Entropy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.3])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, entropies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotation
    ax.annotate('', xy=(1, after_data['entropy']), xytext=(0, before_data['entropy']),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(0.5, 0.45, f'25× improvement\n(+{(after_data["entropy"]/before_data["entropy"]):.1f}×)',
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'entropy_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'entropy_comparison.png'}")
    plt.close()


def plot_class_routing_heatmap():
    """Heatmap showing expert routing per class, before vs after."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = list(before_data['class_routing'].keys())
    
    # Before
    ax = axes[0]
    data_before = np.array([before_data['class_routing'][c] for c in classes])
    im = ax.imshow(data_before, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(expert_names)
    ax.set_yticks(range(5))
    ax.set_yticklabels(classes)
    ax.set_title('Before: Expert Collapse\n(All classes → Expert 3)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(3):
            text = ax.text(j, i, f'{data_before[i, j]:.3f}',
                          ha="center", va="center", color="black" if data_before[i, j] < 0.5 else "white",
                          fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Gating Weight')
    
    # After
    ax = axes[1]
    data_after = np.array([after_data['class_routing'][c] for c in classes])
    im = ax.imshow(data_after, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(expert_names)
    ax.set_yticks(range(5))
    ax.set_yticklabels(classes)
    ax.set_title('After: Specialized Routing\n(Expert 1 → Pathologies, Expert 3 → NORM)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(3):
            text = ax.text(j, i, f'{data_after[i, j]:.3f}',
                          ha="center", va="center", color="black" if data_after[i, j] < 0.5 else "white",
                          fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Gating Weight')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_routing_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'class_routing_heatmap.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'class_routing_heatmap.png'}")
    plt.close()


def plot_auroc_comparison():
    """Compare AUROC before/after for each class."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'Macro']
    auroc_before = [0.9381, 0.9142, 0.9028, 0.9124, 0.6071, 0.8549]
    auroc_after = [0.9368, 0.8749, 0.8940, 0.9188, 0.6826, 0.8614]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auroc_before, width, label='Before (Collapsed)', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, auroc_after, width, label='After (Load Balance)', 
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('AUROC', fontsize=13)
    ax.set_title('Classification Performance: Before vs After Load Balancing', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_ylim([0.5, 1.0])
    ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels and improvement indicators
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        h1 = b1.get_height()
        h2 = b2.get_height()
        ax.text(b1.get_x() + b1.get_width()/2., h1 + 0.01,
                f'{h1:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(b2.get_x() + b2.get_width()/2., h2 + 0.01,
                f'{h2:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add delta indicator
        delta = h2 - h1
        if abs(delta) > 0.005:  # Only show significant changes
            color = 'green' if delta > 0 else 'red'
            symbol = '↑' if delta > 0 else '↓'
            ax.text(x[i], max(h1, h2) + 0.05, f'{symbol}{abs(delta):.3f}',
                   ha='center', fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'auroc_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'auroc_comparison.png'}")
    plt.close()


def plot_intervention_summary():
    """Create a visual summary of the intervention."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Load Balancing Intervention: Solving Expert Collapse', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Problem statement (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    problem_text = (
        "PROBLEM\n\n"
        "⚠️ Expert Collapse:\n"
        "• 98% samples → Expert 3\n"
        "• Entropy: 0.035 (3% of max)\n"
        "• Loss of specialization\n"
        "• Reduced interpretability"
    )
    ax1.text(0.5, 0.5, problem_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7, pad=1))
    
    # 2. Solution (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    solution_text = (
        "SOLUTION\n\n"
        "✨ Load Balancing Loss:\n"
        "• λ_LB = 0.01\n"
        "• λ_symbolic: 0.3 → 0.7\n"
        "• Symbolic features enabled\n"
        "• CV² penalty"
    )
    ax2.text(0.5, 0.5, solution_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#ccf0ff', alpha=0.7, pad=1))
    
    # 3. Result (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    result_text = (
        "RESULT\n\n"
        "✅ Balanced Routing:\n"
        "• 35% / 18% / 48%\n"
        "• Entropy: 0.882 (80% of max)\n"
        "• Specialization emerged\n"
        "• +0.65% Macro AUROC"
    )
    ax3.text(0.5, 0.5, result_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7, pad=1))
    
    # 4. Expert distribution (middle row)
    ax4 = fig.add_subplot(gs[1, :])
    x_pos = [0, 0.5, 1, 1.5, 2, 2.5]
    heights_before = [1.6, 0.4, 98.0, 0, 0, 0]
    heights_after = [0, 0, 0, 34.5, 17.6, 47.9]
    
    bars_before = ax4.bar(x_pos[:3], heights_before[:3], width=0.35, 
                          label='Before', color='#e74c3c', alpha=0.7)
    bars_after = ax4.bar(x_pos[3:], heights_after[3:], width=0.35,
                        label='After', color='#2ecc71', alpha=0.7)
    
    ax4.set_ylabel('Sample %', fontsize=12)
    ax4.set_xticks([0.5, 2])
    ax4.set_xticklabels(['Before\n(Collapsed)', 'After\n(Balanced)'], fontsize=11)
    ax4.axhline(y=33.33, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Ideal')
    ax4.set_ylim([0, 105])
    ax4.set_title('Expert Utilization Distribution', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Key metrics (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    metrics = ['Entropy', 'Expert 1\nUsage', 'Expert 2\nUsage', 'Expert 3\nUsage', 'Macro\nAUROC']
    before_vals = [0.035, 1.6, 0.4, 98.0, 85.49]
    after_vals = [0.882, 34.5, 17.6, 47.9, 86.14]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax5.bar(x - width/2, before_vals, width, label='Before', color='#e74c3c', alpha=0.7)
    ax5.bar(x + width/2, after_vals, width, label='After', color='#2ecc71', alpha=0.7)
    ax5.set_ylabel('Value', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=10)
    ax5.set_title('Key Metrics Comparison', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Note: Different scales for different metrics
    ax5.text(0.5, 0.95, 'Note: Entropy in [0-1.1], Usage in %, AUROC in %',
            transform=ax5.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.savefig(output_dir / 'intervention_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'intervention_summary.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'intervention_summary.png'}")
    plt.close()


if __name__ == '__main__':
    print("=" * 80)
    print("GENERATING COLLAPSE COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    plot_expert_distribution()
    plot_entropy_comparison()
    plot_class_routing_heatmap()
    plot_auroc_comparison()
    plot_intervention_summary()
    
    print("\n" + "=" * 80)
    print(f"✅ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print("=" * 80)
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  • {f.name}")
