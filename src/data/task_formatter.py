"""
Task formatter for converting tasks into LLM-friendly prompts.
"""

from typing import List
from .models import Task, Example


class TaskFormatter:
    """Converts tasks into LLM-friendly prompts."""
    
    def __init__(self, include_instructions: bool = True):
        """
        Initialize the task formatter.
        
        Args:
            include_instructions: Whether to include code generation instructions
        """
        self.include_instructions = include_instructions
    
    def format_prompt(self, task: Task, include_test: bool = False) -> str:
        """
        Format a complete task as a prompt for the LLM.
        
        Args:
            task: Task to format
            include_test: Whether to include test examples in the prompt
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add instructions
        if self.include_instructions:
            prompt_parts.append(self._get_instructions())
        
        # Add training examples
        prompt_parts.append("# Training Examples:")
        for idx, example in enumerate(task.train_examples, 1):
            prompt_parts.append(f"\n## Example {idx}:")
            prompt_parts.append(self.format_examples([example]))
        
        # Optionally add test examples
        if include_test and task.test_examples:
            prompt_parts.append("\n# Test Examples:")
            for idx, example in enumerate(task.test_examples, 1):
                prompt_parts.append(f"\n## Test {idx}:")
                prompt_parts.append(self.format_examples([example]))
        
        # Add code generation prompt
        prompt_parts.append("\n# Task:")
        prompt_parts.append("Write a Python function that transforms the input grid to the output grid.")
        prompt_parts.append("Your function should work for all examples shown above.")
        
        return "\n".join(prompt_parts)
    
    def format_examples(self, examples: List[Example]) -> str:
        """
        Format a list of examples as text.
        
        Args:
            examples: List of examples to format
            
        Returns:
            Formatted examples string
        """
        formatted = []
        for example in examples:
            formatted.append("Input:")
            formatted.append(self._grid_to_text(example.input_grid))
            formatted.append("\nOutput:")
            formatted.append(self._grid_to_text(example.output_grid))
        return "\n".join(formatted)
    
    def create_training_batch(self, tasks: List[Task], batch_size: int = 8) -> List[List[Task]]:
        """
        Create batches of tasks for training.
        
        Args:
            tasks: List of tasks to batch
            batch_size: Number of tasks per batch
            
        Returns:
            List of task batches
        """
        batches = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _grid_to_text(self, grid: List[List[int]]) -> str:
        """
        Convert a grid to text representation.
        
        Args:
            grid: 2D grid of integers
            
        Returns:
            Text representation of the grid
        """
        return "\n".join([" ".join(map(str, row)) for row in grid])
    
    def _text_to_grid(self, text: str) -> List[List[int]]:
        """
        Convert text representation back to a grid.
        
        Args:
            text: Text representation of a grid
            
        Returns:
            2D grid of integers
        """
        lines = text.strip().split("\n")
        grid = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.strip().split()]
                grid.append(row)
        return grid
    
    def _get_instructions(self) -> str:
        """Get the code generation instructions."""
        return """# Code Golf Task

You are given a pattern recognition task. Study the input-output examples and write a Python function that transforms the input to the output.

The function should:
1. Take a 2D grid (list of lists) as input
2. Return a 2D grid (list of lists) as output
3. Work correctly for all provided examples
4. Be as concise as possible (this is code golf!)
"""
