import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def extract_metrics(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                metric_name = line.split(':')[0]
                # Extract second Mean(STD) value
                values = line.split(',')[2].strip()
                mean = float(values.split('(')[0])
                std = float(values.split('(')[1].rstrip(')'))
                metrics[metric_name] = (mean, std)
    return metrics


def extract_metrics_json(filepath):
    """Extract metrics from JSON file, focusing on rp (replay) metrics"""
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return {
        'MSE': (metrics['rp_mses_mean'], metrics['rp_mses_std']),
        'SCC': (metrics['rp_scc_mean'], metrics['rp_scc_std']),
        'DCC': (metrics['rp_dcc_mean'], metrics['rp_dcc_std'])
    }


# Setup paths and containers
base_path = "../experiments"
samples_range = [3, 4, 6, 8, 10]
seeds = [1111, 2222, 3333, 4444, 5555]
version_map = {0: 3, 1: 4, 2: 6, 3: 8, 4: 10}  # Map version numbers to sample counts

# Add baseline path
baseline_path = "../experiments/feedforwardmask-stationary_synthetic_stationary_naive_1111_1.0/feedforwardmask-stationary/version_0/metrics_generalization_excel.txt"
baseline_small_path = "../experiments/feedforwardmask-stationary_synthetic_small_stationary_naive_1111_1.0/feedforwardmask-stationary/version_3/metrics_generalization_excel.txt"

# Containers for metrics - add small dataset containers
mse_values = {s: [] for s in samples_range}
scc_values = {s: [] for s in samples_range}
dcc_values = {s: [] for s in samples_range}
mse_values_small = {s: [] for s in samples_range}
scc_values_small = {s: [] for s in samples_range}
dcc_values_small = {s: [] for s in samples_range}

# Get baseline metrics
if os.path.exists(baseline_path):
    baseline_metrics = extract_metrics(baseline_path)
    baseline_mse = baseline_metrics['MSE'][0]
    baseline_scc = baseline_metrics['SCC'][0]
    baseline_dcc = baseline_metrics['DCC'][0]
else:
    print(f"Warning: Full baseline metrics not found at {baseline_path}")
    baseline_mse = baseline_scc = baseline_dcc = None

# Get small baseline metrics
if os.path.exists(baseline_small_path):
    baseline_small_metrics = extract_metrics(baseline_small_path)
    baseline_small_mse = baseline_small_metrics['MSE'][0]
    baseline_small_scc = baseline_small_metrics['SCC'][0]
    baseline_small_dcc = baseline_small_metrics['DCC'][0]
else:
    print(f"Warning: Small baseline metrics not found at {baseline_small_path}")
    baseline_small_mse = baseline_small_scc = baseline_small_dcc = None

# Collect metrics for original dataset
for samples in samples_range:
    for seed in seeds:
        pattern = f"pretrain_feedforwardmask_synthetic_continual_er_{seed}_{samples}samples"
        exp_path = os.path.join(base_path, pattern, "feedforwardmask/version_0/metrics.json")
        
        if os.path.exists(exp_path):
            metrics = extract_metrics_json(exp_path)
            mse_values[samples].append(metrics['MSE'][0])
            scc_values[samples].append(metrics['SCC'][0])
            dcc_values[samples].append(metrics['DCC'][0])

# Collect metrics for small dataset
for seed in seeds:
    base_pattern = f"feedforwardmask_synthetic_small_continual_er_{seed}_1.0/feedforwardmask"
    for version, samples in version_map.items():
        exp_path = os.path.join(base_path, base_pattern, f"version_{version}/metrics.json")
        
        if os.path.exists(exp_path):
            metrics = extract_metrics_json(exp_path)
            mse_values_small[samples].append(metrics['MSE'][0])
            scc_values_small[samples].append(metrics['SCC'][0])
            dcc_values_small[samples].append(metrics['DCC'][0])

# Calculate means and stds for both datasets
x = samples_range
mse_mean = [np.mean(mse_values[s]) for s in samples_range]
mse_std = [np.std(mse_values[s]) for s in samples_range]
scc_mean = [np.mean(scc_values[s]) for s in samples_range]
scc_std = [np.std(scc_values[s]) for s in samples_range]
dcc_mean = [np.mean(dcc_values[s]) for s in samples_range]
dcc_std = [np.std(dcc_values[s]) for s in samples_range]

mse_mean_small = [np.nanmean(mse_values_small[s]) for s in samples_range]
mse_std_small = [np.nanstd(mse_values_small[s]) for s in samples_range]
scc_mean_small = [np.nanmean(scc_values_small[s]) for s in samples_range]
scc_std_small = [np.nanstd(scc_values_small[s]) for s in samples_range]
dcc_mean_small = [np.nanmean(dcc_values_small[s]) for s in samples_range]
dcc_std_small = [np.nanstd(dcc_values_small[s]) for s in samples_range]

# Set global font properties
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Create figure with border and subplots
fig = plt.figure(figsize=(15, 5))  # Slightly taller to accommodate legend
fig.patch.set_linewidth(1)
fig.patch.set_edgecolor('black')

# Create subplots
ax1 = plt.subplot(131)
ax1.errorbar(x, mse_mean, yerr=mse_std, marker='o', capsize=5, label='Pre-Trained CL', linewidth=2, markersize=8)
ax1.errorbar(x, mse_mean_small, yerr=mse_std_small, marker='s', capsize=5, label='From-Scratch CL', linewidth=2, markersize=8, color='lightblue')

if baseline_mse is not None:
    ax1.axhline(y=baseline_mse, color='y', linestyle='--', label='Meta-Stationary', linewidth=2)
if baseline_small_mse is not None:
    ax1.axhline(y=baseline_small_mse, color='g', linestyle='--', label='Meta-Generalization', linewidth=2)
ax1.set_xlabel('Number of Samples', fontweight='bold')
ax1.set_ylabel('MSE↓', fontweight='bold')
ax1.set_title('MSE↓ vs # Samples', pad=10)
ax1.grid(True, linewidth=0.5)
ax1.set_xticks([3, 4, 6, 8, 10])
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.get_offset_text().set_fontweight('bold')

# SCC plot
ax2 = plt.subplot(132)
ax2.errorbar(x, scc_mean, yerr=scc_std, marker='o', capsize=5, label='Pre-Trained CL', linewidth=2, markersize=8)
ax2.errorbar(x, scc_mean_small, yerr=scc_std_small, marker='s', capsize=5, label='From-Scratch CL', linewidth=2, markersize=8, color='lightblue')
if baseline_scc is not None:
    ax2.axhline(y=baseline_scc, color='y', linestyle='--', label='Meta-Stationary', linewidth=2)
if baseline_small_scc is not None:
    ax2.axhline(y=baseline_small_scc, color='g', linestyle='--', label='Meta-Generalization', linewidth=2)
ax2.set_xlabel('Number of Samples', fontweight='bold')
ax2.set_xticks([3, 4, 6, 8, 10])
# ax2.set_ylim([0.5, 1.005])
ax2.set_ylabel('SCC↑', fontweight='bold')
ax2.set_title('SCC↑ vs # Samples', pad=10)
ax2.grid(True, linewidth=0.5)

# DCC plot
ax3 = plt.subplot(133)
ax3.errorbar(x, dcc_mean, yerr=dcc_std, marker='o', capsize=5, label='Pre-Trained CL', linewidth=2, markersize=8)
ax3.errorbar(x, dcc_mean_small, yerr=dcc_std_small, marker='s', capsize=5, label='From-Scratch CL', linewidth=2, markersize=8, color='lightblue')
if baseline_dcc is not None:
    ax3.axhline(y=baseline_dcc, color='y', linestyle='--', label='Meta-Stationary', linewidth=2)
if baseline_small_dcc is not None:
    ax3.axhline(y=baseline_small_dcc, color='g', linestyle='--', label='Meta-Generalization', linewidth=2)
ax3.set_xlabel('Number of Samples', fontweight='bold')
ax3.set_xticks([3, 4, 6, 8, 10])
# ax3.set_ylim([0.5, 1.005])
ax3.set_ylabel('DC↑', fontweight='bold')
ax3.set_title('DC↑ vs # Samples', pad=10)
ax3.grid(True, linewidth=0.5)

# Create shared legend
handles, labels = ax1.get_legend_handles_labels()
legend = fig.legend(handles, labels, 
                   loc='upper center', 
                   bbox_to_anchor=(0.52, 0.05), 
                   ncol=4, 
                   fontsize=12)

# Customize legend appearance
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_alpha(1)
legend.get_frame().set_linewidth(2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend
plt.savefig('metrics_vs_samples.png', bbox_inches='tight', dpi=300)
plt.close()