"""
Metrics tracker for logging training progress.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict


class MetricsTracker:
    """Tracks and logs training metrics."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics_history = defaultdict(list)
        
        # Create experiment log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
    
    def log_training_step(self, step: int, metrics: Dict[str, Any]):
        """
        Log metrics from a training step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metrics
        """
        # Add timestamp and step
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        # Append to history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_evaluation(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log evaluation metrics.
        
        Args:
            epoch: Epoch number
            metrics: Evaluation metrics
        """
        log_entry = {
            "type": "evaluation",
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_metric_history(self, metric_name: str) -> List[Any]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest value for each metric.
        
        Returns:
            Dictionary of latest metrics
        """
        latest = {}
        for key, values in self.metrics_history.items():
            if values:
                latest[key] = values[-1]
        return latest
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary of statistics for each metric
        """
        import numpy as np
        
        summary = {}
        for key, values in self.metrics_history.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "latest": float(values[-1])
                }
        
        return summary
    
    def export_results(self, path: str):
        """
        Export all metrics to a JSON file.
        
        Args:
            path: Path to save results
        """
        results = {
            "experiment_name": self.experiment_name,
            "metrics_history": dict(self.metrics_history),
            "summary": self.get_summary_statistics(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def print_summary(self):
        """Print a summary of training metrics."""
        print(f"\n{'='*60}")
        print(f"Training Summary: {self.experiment_name}")
        print(f"{'='*60}")
        
        summary = self.get_summary_statistics()
        
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            print(f"  Latest: {stats['latest']:.4f}")
            print(f"  Mean:   {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print(f"\n{'='*60}\n")
    
    def generate_learning_curves(self, save_path: Optional[str] = None):
        """
        Generate learning curve plots.
        
        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots for different metrics
            metrics_to_plot = [
                "average_reward",
                "success_rate",
                "policy_loss",
                "value_loss"
            ]
            
            available_metrics = [m for m in metrics_to_plot if m in self.metrics_history]
            
            if not available_metrics:
                print("No metrics available for plotting")
                return
            
            fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 3*len(available_metrics)))
            
            if len(available_metrics) == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, available_metrics):
                values = self.metrics_history[metric]
                ax.plot(values)
                ax.set_xlabel("Step")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_title(f"{metric.replace('_', ' ').title()} over Training")
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Learning curves saved to {save_path}")
            else:
                plt.show()
        
        except ImportError:
            print("matplotlib not available for plotting")


class EarlyStopping:
    """Early stopping based on validation performance."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "max" for metrics to maximize, "min" for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
