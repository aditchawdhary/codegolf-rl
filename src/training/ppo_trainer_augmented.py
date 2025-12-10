"""
PPO trainer with data augmentation support.
Augments tasks in memory during training for better generalization.
"""

from typing import List
from src.data import Task, ProblemAugmenter
from .ppo_trainer import PPOTrainer, TrainingMetrics


class AugmentedPPOTrainer(PPOTrainer):
    """PPO trainer with in-memory data augmentation."""
    
    def __init__(
        self, 
        *args, 
        num_augmentations: int = 16,
        augmentation_types: List[str] = None,
        **kwargs
    ):
        """
        Initialize augmented trainer.
        
        Args:
            num_augmentations: Number of augmentations per task (default: 16)
            augmentation_types: Types of augmentation to use
            *args, **kwargs: Passed to PPOTrainer
        """
        super().__init__(*args, **kwargs)
        
        self.num_augmentations = num_augmentations
        self.augmenter = ProblemAugmenter(augmentation_types)
        
        print(f"[AugmentedPPOTrainer] Initialized with {num_augmentations}x augmentation")
        print(f"[AugmentedPPOTrainer] Augmentation types: {self.augmenter.augmentation_types}")
    
    def train_step(self, tasks: List[Task]) -> TrainingMetrics:
        """
        Training step with augmentation.
        
        Process:
        1. Augment each task N times in memory (e.g., 8 tasks → 128 tasks)
        2. Collect trajectories on augmented tasks
        3. Run PPO update
        4. Augmented tasks are garbage collected after step
        
        Args:
            tasks: Original tasks (e.g., 8 tasks)
            
        Returns:
            Training metrics
        """
        print(f"\n[AugmentedPPOTrainer] train_step called with {len(tasks)} tasks")
        
        # Augment tasks in memory (8 → 128 if num_augmentations=16)
        print(f"[AugmentedPPOTrainer] Augmenting {len(tasks)} → {len(tasks) * self.num_augmentations} tasks...")
        augmented_tasks = []
        
        for task in tasks:
            # Generate N augmented versions (includes original)
            augmented = self.augmenter.augment_multiple(
                task, 
                n=self.num_augmentations
            )
            augmented_tasks.extend(augmented)
        
        print(f"[AugmentedPPOTrainer] ✓ Augmented to {len(augmented_tasks)} tasks")
        
        # Call parent's train_step with augmented tasks
        # This collects trajectories, computes advantages, and updates networks
        metrics = super().train_step(augmented_tasks)
        
        # Augmented tasks are garbage collected here
        print(f"[AugmentedPPOTrainer] ✓ Training step complete\n")
        
        return metrics
