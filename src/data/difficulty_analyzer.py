"""
Difficulty analyzer for categorizing tasks by complexity.
"""

from typing import List, Dict
import numpy as np
from .models import Task, Example


class DifficultyAnalyzer:
    """Analyzes and categorizes task difficulty."""
    
    def compute_complexity_score(self, task: Task) -> float:
        """
        Compute a complexity score for a task.
        
        The score is based on multiple factors:
        - Grid size (larger grids are more complex)
        - Number of unique values (more values = more complex)
        - Output/input size ratio (transformations are more complex)
        - Number of training examples (fewer examples = harder to learn)
        
        Args:
            task: Task to analyze
            
        Returns:
            Complexity score (higher = more difficult)
        """
        if not task.train_examples:
            return 0.0
        
        # Compute metrics for all training examples
        grid_sizes = []
        unique_values = []
        size_ratios = []
        
        for example in task.train_examples:
            # Grid size complexity
            input_size = len(example.input_grid) * len(example.input_grid[0]) if example.input_grid else 0
            output_size = len(example.output_grid) * len(example.output_grid[0]) if example.output_grid else 0
            grid_sizes.append(max(input_size, output_size))
            
            # Unique values complexity
            input_values = set()
            for row in example.input_grid:
                input_values.update(row)
            unique_values.append(len(input_values))
            
            # Size ratio complexity (how much the grid changes)
            if input_size > 0:
                size_ratios.append(output_size / input_size)
            else:
                size_ratios.append(1.0)
        
        # Aggregate metrics
        avg_grid_size = np.mean(grid_sizes)
        avg_unique_values = np.mean(unique_values)
        avg_size_ratio = np.mean(size_ratios)
        num_examples = len(task.train_examples)
        
        # Compute complexity score (normalized to 0-100 range)
        # Larger grids, more unique values, and size changes increase complexity
        # More training examples decrease complexity (easier to learn)
        complexity = (
            (avg_grid_size / 100.0) * 30 +  # Grid size component (0-30)
            (avg_unique_values / 10.0) * 20 +  # Unique values component (0-20)
            abs(avg_size_ratio - 1.0) * 30 +  # Size change component (0-30)
            (10 / max(num_examples, 1)) * 20  # Example scarcity component (0-20)
        )
        
        # Store detailed metrics
        task.complexity_metrics = {
            "avg_grid_size": float(avg_grid_size),
            "avg_unique_values": float(avg_unique_values),
            "avg_size_ratio": float(avg_size_ratio),
            "num_train_examples": num_examples,
        }
        
        task.difficulty_score = min(complexity, 100.0)
        return task.difficulty_score
    
    def categorize_by_difficulty(self, tasks: List[Task]) -> Dict[str, List[Task]]:
        """
        Categorize tasks into difficulty levels.
        
        Args:
            tasks: List of tasks to categorize
            
        Returns:
            Dictionary mapping difficulty level to list of tasks
        """
        # Compute scores for all tasks
        for task in tasks:
            if task.difficulty_score == 0.0:
                self.compute_complexity_score(task)
        
        # Sort tasks by difficulty
        sorted_tasks = sorted(tasks, key=lambda t: t.difficulty_score)
        
        # Categorize into easy, medium, hard
        n = len(sorted_tasks)
        easy_cutoff = n // 3
        medium_cutoff = 2 * n // 3
        
        return {
            "easy": sorted_tasks[:easy_cutoff],
            "medium": sorted_tasks[easy_cutoff:medium_cutoff],
            "hard": sorted_tasks[medium_cutoff:],
        }
    
    def get_difficulty_distribution(self, tasks: List[Task]) -> Dict[str, int]:
        """
        Get the distribution of tasks across difficulty levels.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Dictionary with counts for each difficulty level
        """
        categories = self.categorize_by_difficulty(tasks)
        return {
            level: len(task_list)
            for level, task_list in categories.items()
        }
