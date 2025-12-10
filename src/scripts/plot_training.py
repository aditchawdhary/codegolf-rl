#!/usr/bin/env python
"""
Plot training progress from logs.
Creates comprehensive visualizations of the training run.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_training_logs(log_file: str):
    """Load training logs from JSONL file."""
    metrics = {
        'steps': [],
        'policy_loss': [],
        'value_loss': [],
        'average_reward': [],
        'success_rate': [],
        'kl_divergence': [],
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'step' in data:
                metrics['steps'].append(data['step'])
                metrics['policy_loss'].append(data.get('policy_loss', 0))
                metrics['value_loss'].append(data.get('value_loss', 0))
                metrics['average_reward'].append(data.get('average_reward', 0))
                metrics['success_rate'].append(data.get('success_rate', 0))
                metrics['kl_divergence'].append(data.get('kl_divergence', 0))
    
    return metrics


def smooth(values, weight=0.8):
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0] if values else 0
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_overview(metrics, save_path=None):
    """
    Create comprehensive 6-panel overview of training.
    This is your main visualization!
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Training Progress Overview', fontsize=16, fontweight='bold')
    
    steps = metrics['steps']
    
    # 1. Success Rate (most important!)
    ax = axes[0, 0]
    ax.plot(steps, metrics['success_rate'], alpha=0.3, label='Raw', color='green')
    ax.plot(steps, smooth(metrics['success_rate']), label='Smoothed', color='darkgreen', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate (% of tasks solved correctly)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 2. Average Reward
    ax = axes[0, 1]
    ax.plot(steps, metrics['average_reward'], alpha=0.3, label='Raw', color='blue')
    ax.plot(steps, smooth(metrics['average_reward']), label='Smoothed', color='darkblue', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward per Task')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Policy Loss
    ax = axes[1, 0]
    ax.plot(steps, metrics['policy_loss'], alpha=0.3, label='Raw', color='red')
    ax.plot(steps, smooth(metrics['policy_loss']), label='Smoothed', color='darkred', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss (should decrease)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Value Loss
    ax = axes[1, 1]
    ax.plot(steps, metrics['value_loss'], alpha=0.3, label='Raw', color='orange')
    ax.plot(steps, smooth(metrics['value_loss']), label='Smoothed', color='darkorange', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss (should decrease)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. KL Divergence
    ax = axes[2, 0]
    ax.plot(steps, metrics['kl_divergence'], alpha=0.5, color='purple')
    ax.axhline(y=0.01, color='red', linestyle='--', label='Target KL (0.01)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence (policy change per update)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Learning Progress Summary
    ax = axes[2, 1]
    # Normalize metrics to 0-1 for comparison
    norm_success = np.array(smooth(metrics['success_rate']))
    norm_reward = np.array(smooth(metrics['average_reward']))
    norm_reward = (norm_reward - norm_reward.min()) / (norm_reward.max() - norm_reward.min() + 1e-8)
    
    ax.plot(steps, norm_success, label='Success Rate', linewidth=2, color='green')
    ax.plot(steps, norm_reward, label='Reward (normalized)', linewidth=2, color='blue')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Learning Progress (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved overview plot to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_success_rate_detail(metrics, save_path=None):
    """
    Detailed success rate plot with milestones.
    Shows when model hits key performance thresholds.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = metrics['steps']
    success = metrics['success_rate']
    success_smooth = smooth(success)
    
    # Plot raw and smoothed
    ax.plot(steps, success, alpha=0.2, color='lightgreen', label='Raw')
    ax.plot(steps, success_smooth, linewidth=2.5, color='darkgreen', label='Smoothed (EMA)')
    
    # Add milestone lines
    milestones = [0.1, 0.25, 0.5, 0.75]
    colors = ['red', 'orange', 'yellow', 'lime']
    for milestone, color in zip(milestones, colors):
        ax.axhline(y=milestone, color=color, linestyle='--', alpha=0.5, 
                   label=f'{milestone*100:.0f}% threshold')
        
        # Find when we crossed this milestone
        crossed = [i for i, s in enumerate(success_smooth) if s >= milestone]
        if crossed:
            first_cross = crossed[0]
            ax.scatter([steps[first_cross]], [success_smooth[first_cross]], 
                      color=color, s=100, zorder=5)
            ax.annotate(f'Step {steps[first_cross]}', 
                       xy=(steps[first_cross], success_smooth[first_cross]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color=color)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Training Success Rate with Milestones', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add stats text box
    final_success = success_smooth[-1]
    max_success = max(success_smooth)
    textstr = f'Final: {final_success:.1%}\nPeak: {max_success:.1%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved success rate detail to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_loss_curves(metrics, save_path=None):
    """
    Combined loss curves (policy + value).
    Shows if model is learning or overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = metrics['steps']
    
    # Policy Loss
    policy_smooth = smooth(metrics['policy_loss'])
    ax1.plot(steps, metrics['policy_loss'], alpha=0.2, color='lightcoral')
    ax1.plot(steps, policy_smooth, linewidth=2, color='darkred', label='Policy Loss')
    ax1.fill_between(steps, 0, policy_smooth, alpha=0.1, color='red')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Policy Loss (PPO Objective)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Value Loss
    value_smooth = smooth(metrics['value_loss'])
    ax2.plot(steps, metrics['value_loss'], alpha=0.2, color='lightsalmon')
    ax2.plot(steps, value_smooth, linewidth=2, color='darkorange', label='Value Loss')
    ax2.fill_between(steps, 0, value_smooth, alpha=0.1, color='orange')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Value Loss (TD Error)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved loss curves to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_reward_distribution(metrics, save_path=None):
    """
    Histogram showing reward distribution over time.
    Shows if rewards are improving.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = metrics['steps']
    rewards = metrics['average_reward']
    
    # Reward over time
    reward_smooth = smooth(rewards)
    ax1.plot(steps, rewards, alpha=0.3, color='skyblue')
    ax1.plot(steps, reward_smooth, linewidth=2, color='darkblue', label='Smoothed')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero reward')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reward distribution (early vs late)
    n = len(rewards)
    early_rewards = rewards[:n//3]
    late_rewards = rewards[2*n//3:]
    
    ax2.hist(early_rewards, bins=20, alpha=0.5, color='red', label=f'Early (steps 0-{n//3})')
    ax2.hist(late_rewards, bins=20, alpha=0.5, color='green', label=f'Late (steps {2*n//3}-{n})')
    ax2.set_xlabel('Average Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution: Early vs Late Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reward distribution to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_augmentation_impact(metrics, num_augmentations=16, save_path=None):
    """
    Show effective batch size with augmentation.
    Visualizes the 16x data multiplier.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = metrics['steps']
    success = smooth(metrics['success_rate'])
    
    # Plot success rate
    ax.plot(steps, success, linewidth=2.5, color='purple', label='Success Rate')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'Impact of {num_augmentations}x Data Augmentation', 
                 fontsize=14, fontweight='bold')
    
    # Add annotation showing effective samples
    batch_size = 8  # Adjust based on your config
    effective_batch = batch_size * num_augmentations
    
    textstr = f'Batch Size: {batch_size}\n' \
              f'Augmentations: {num_augmentations}x\n' \
              f'Effective Batch: {effective_batch}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    ax.text(0.70, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved augmentation impact to {save_path}")
    else:
        plt.show()
    
    return fig


def generate_training_report(log_file: str, output_dir: str = "plots"):
    """
    Generate complete training report with all visualizations.
    """
    print("\n" + "="*60)
    print("GENERATING TRAINING REPORT")
    print("="*60)
    
    # Load data
    print(f"\nLoading logs from: {log_file}")
    metrics = load_training_logs(log_file)
    print(f"✓ Loaded {len(metrics['steps'])} training steps")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_training_overview(
        metrics, 
        save_path=output_path / "1_training_overview.png"
    )
    
    plot_success_rate_detail(
        metrics,
        save_path=output_path / "2_success_rate_detail.png"
    )
    
    plot_loss_curves(
        metrics,
        save_path=output_path / "3_loss_curves.png"
    )
    
    plot_reward_distribution(
        metrics,
        save_path=output_path / "4_reward_distribution.png"
    )
    
    plot_augmentation_impact(
        metrics,
        num_augmentations=16,
        save_path=output_path / "5_augmentation_impact.png"
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    final_success = metrics['success_rate'][-1]
    max_success = max(metrics['success_rate'])
    final_reward = metrics['average_reward'][-1]
    final_loss = metrics['policy_loss'][-1]
    
    print(f"\nFinal Metrics:")
    print(f"  Success Rate: {final_success:.2%}")
    print(f"  Peak Success: {max_success:.2%}")
    print(f"  Avg Reward:   {final_reward:.3f}")
    print(f"  Policy Loss:  {final_loss:.3f}")
    
    print(f"\nPlots saved to: {output_path}/")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot training progress")
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to training log file (.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    generate_training_report(args.log_file, args.output_dir)


if __name__ == "__main__":
    main()
