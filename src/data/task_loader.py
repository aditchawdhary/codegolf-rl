"""
Task loader for reading and parsing code golf task JSON files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import Task, Example


class TaskLoader:
    """Loads task JSON files and extracts examples."""
    
    def __init__(self, task_dir: str = "google-code-golf-2025"):
        """
        Initialize the task loader.
        
        Args:
            task_dir: Directory containing task JSON files
        """
        self.task_dir = Path(task_dir)
        if not self.task_dir.exists():
            raise ValueError(f"Task directory does not exist: {task_dir}")
    
    def load_task(self, task_id: int) -> Task:
        """
        Load a single task by ID.
        
        Args:
            task_id: Task ID (1-400)
            
        Returns:
            Task object with all examples
            
        Raises:
            FileNotFoundError: If task file doesn't exist
            ValueError: If task data is invalid
        """
        task_file = self.task_dir / f"task{task_id:03d}.json"
        
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
        with open(task_file, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        required_fields = ["train", "test", "arc-gen"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Task {task_id} missing required field: {field}")
        
        # Parse examples
        train_examples = self._parse_examples(data["train"])
        test_examples = self._parse_examples(data["test"])
        arc_gen_examples = self._parse_examples(data["arc-gen"])
        
        return Task(
            task_id=task_id,
            train_examples=train_examples,
            test_examples=test_examples,
            arc_gen_examples=arc_gen_examples
        )
    
    def load_all_tasks(self) -> List[Task]:
        """
        Load all tasks from the task directory.
        
        Returns:
            List of Task objects
        """
        tasks = []
        for task_id in range(1, 401):
            try:
                task = self.load_task(task_id)
                tasks.append(task)
            except FileNotFoundError:
                # Skip missing tasks
                continue
        return tasks
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all tasks.
        
        Returns:
            Dictionary with task statistics
        """
        tasks = self.load_all_tasks()
        
        total_train = sum(len(t.train_examples) for t in tasks)
        total_test = sum(len(t.test_examples) for t in tasks)
        total_arc_gen = sum(len(t.arc_gen_examples) for t in tasks)
        
        return {
            "num_tasks": len(tasks),
            "total_train_examples": total_train,
            "total_test_examples": total_test,
            "total_arc_gen_examples": total_arc_gen,
            "avg_train_per_task": total_train / len(tasks) if tasks else 0,
            "avg_test_per_task": total_test / len(tasks) if tasks else 0,
            "avg_arc_gen_per_task": total_arc_gen / len(tasks) if tasks else 0,
        }
    
    def _parse_examples(self, examples_data: List[Dict]) -> List[Example]:
        """
        Parse example data into Example objects.
        
        Args:
            examples_data: List of example dictionaries
            
        Returns:
            List of Example objects
        """
        examples = []
        for ex_data in examples_data:
            if "input" not in ex_data or "output" not in ex_data:
                raise ValueError("Example missing input or output field")
            
            example = Example(
                input_grid=ex_data["input"],
                output_grid=ex_data["output"]
            )
            examples.append(example)
        
        return examples
