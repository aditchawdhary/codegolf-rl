"""
Inference pipeline for generating solutions to new tasks.
"""

import ast
from typing import List, Optional, Dict, Any
from src.models import PolicyNetwork
from src.data import Task, TaskFormatter
from src.training import CodeSandbox


class InferencePipeline:
    """Pipeline for generating and validating code solutions."""
    
    def __init__(
        self,
        policy: PolicyNetwork,
        sandbox: CodeSandbox,
        formatter: Optional[TaskFormatter] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            policy: Trained policy network
            sandbox: Code execution sandbox
            formatter: Task formatter (creates one if None)
        """
        self.policy = policy
        self.sandbox = sandbox
        self.formatter = formatter or TaskFormatter()
    
    def generate_solution(
        self,
        task: Task,
        strategy: str = "temperature",
        num_samples: int = 1,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate solution(s) for a task.
        
        Args:
            task: Task to solve
            strategy: Sampling strategy ("greedy", "temperature", "beam")
            num_samples: Number of solutions to generate
            temperature: Sampling temperature
            max_retries: Maximum retry attempts for failed generation
            
        Returns:
            List of solution dictionaries with code and metadata
        """
        solutions = []
        
        # Format task as prompt
        prompt = self.formatter.format_prompt(task, include_test=False)
        
        for i in range(num_samples):
            solution = self._generate_single_solution(
                prompt,
                task,
                strategy,
                temperature,
                max_retries
            )
            solutions.append(solution)
        
        return solutions
    
    def _generate_single_solution(
        self,
        prompt: str,
        task: Task,
        strategy: str,
        temperature: float,
        max_retries: int
    ) -> Dict[str, Any]:
        """Generate a single solution with retries."""
        for attempt in range(max_retries):
            try:
                # Generate code based on strategy
                if strategy == "greedy":
                    code = self.policy.generate(
                        prompt,
                        temperature=0.0,
                        max_new_tokens=512
                    )
                elif strategy == "temperature":
                    code = self.policy.generate(
                        prompt,
                        temperature=temperature,
                        max_new_tokens=512
                    )
                elif strategy == "beam":
                    # Simplified beam search (just use low temperature)
                    code = self.policy.generate(
                        prompt,
                        temperature=0.3,
                        top_k=5,
                        max_new_tokens=512
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                # Validate syntax
                is_valid = self.validate_syntax(code)
                
                # Format code
                formatted_code = self.format_code(code)
                
                # Test on task examples
                results = self.sandbox.execute(formatted_code, task)
                
                return {
                    "code": formatted_code,
                    "raw_code": code,
                    "syntax_valid": is_valid,
                    "test_pass_rate": results.test_pass_rate,
                    "success": results.success,
                    "errors": results.errors,
                    "strategy": strategy,
                    "attempt": attempt + 1
                }
            
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed
                    return {
                        "code": "",
                        "raw_code": "",
                        "syntax_valid": False,
                        "test_pass_rate": 0.0,
                        "success": False,
                        "errors": f"Generation failed: {str(e)}",
                        "strategy": strategy,
                        "attempt": attempt + 1
                    }
        
        return {
            "code": "",
            "syntax_valid": False,
            "test_pass_rate": 0.0,
            "success": False,
            "errors": "Max retries exceeded",
            "strategy": strategy
        }
    
    def validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax.
        
        Args:
            code: Python code string
            
        Returns:
            True if syntax is valid
        """
        return self.sandbox.validate_syntax(code)
    
    def format_code(self, code: str) -> str:
        """
        Format and clean generated code.
        
        Args:
            code: Raw generated code
            
        Returns:
            Formatted code
        """
        # Remove common artifacts
        code = code.strip()
        
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        
        return code
    
    def batch_inference(
        self,
        tasks: List[Task],
        strategy: str = "temperature",
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate solutions for multiple tasks.
        
        Args:
            tasks: List of tasks
            strategy: Sampling strategy
            **kwargs: Additional arguments for generate_solution
            
        Returns:
            List of solution lists (one per task)
        """
        all_solutions = []
        
        for task in tasks:
            solutions = self.generate_solution(
                task,
                strategy=strategy,
                **kwargs
            )
            all_solutions.append(solutions)
        
        return all_solutions
    
    def evaluate_solution(
        self,
        code: str,
        task: Task
    ) -> Dict[str, Any]:
        """
        Evaluate a solution against a task.
        
        Args:
            code: Solution code
            task: Task to evaluate against
            
        Returns:
            Evaluation results
        """
        results = self.sandbox.execute(code, task)
        
        return {
            "test_pass_rate": results.test_pass_rate,
            "success": results.success,
            "syntax_valid": results.syntax_valid,
            "errors": results.errors,
            "outputs": results.outputs
        }
