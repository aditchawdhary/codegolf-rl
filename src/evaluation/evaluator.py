"""
Performance evaluator for measuring model performance.
"""

from typing import List, Dict, Any
from src.models import PolicyNetwork
from src.data import Task
from .inference import InferencePipeline


class PerformanceEvaluator:
    """Evaluates model on validation tasks."""
    
    def __init__(self, inference_pipeline: InferencePipeline):
        """
        Initialize evaluator.
        
        Args:
            inference_pipeline: Inference pipeline
        """
        self.inference = inference_pipeline
    
    def evaluate(
        self,
        tasks: List[Task],
        num_samples: int = 1,
        strategy: str = "temperature"
    ) -> Dict[str, Any]:
        """
        Evaluate model on a set of tasks.
        
        Args:
            tasks: List of tasks to evaluate on
            num_samples: Number of samples per task
            strategy: Sampling strategy
            
        Returns:
            Evaluation metrics
        """
        total_tasks = len(tasks)
        solved_tasks = 0
        total_pass_rate = 0.0
        code_lengths = []
        
        for task in tasks:
            solutions = self.inference.generate_solution(
                task,
                strategy=strategy,
                num_samples=num_samples
            )
            
            # Check if any solution succeeded
            task_solved = any(sol["success"] for sol in solutions)
            if task_solved:
                solved_tasks += 1
            
            # Track best pass rate for this task
            best_pass_rate = max(sol["test_pass_rate"] for sol in solutions)
            total_pass_rate += best_pass_rate
            
            # Track code lengths of valid solutions
            for sol in solutions:
                if sol["syntax_valid"] and sol["code"]:
                    code_lengths.append(len(sol["code"]))
        
        success_rate = solved_tasks / total_tasks if total_tasks > 0 else 0.0
        avg_pass_rate = total_pass_rate / total_tasks if total_tasks > 0 else 0.0
        avg_code_length = sum(code_lengths) / len(code_lengths) if code_lengths else 0.0
        
        return {
            "success_rate": success_rate,
            "average_pass_rate": avg_pass_rate,
            "solved_tasks": solved_tasks,
            "total_tasks": total_tasks,
            "average_code_length": avg_code_length,
            "num_valid_solutions": len(code_lengths)
        }
    
    def compute_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute success rate from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Success rate (0-1)
        """
        if not results:
            return 0.0
        
        successes = sum(1 for r in results if r.get("success", False))
        return successes / len(results)
