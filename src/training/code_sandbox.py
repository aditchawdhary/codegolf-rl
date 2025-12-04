"""
Safe code execution sandbox for evaluating generated code.
"""

import ast
import sys
import io
import signal
import traceback
from contextlib import contextmanager
from typing import List, Any, Optional, Dict
import resource
from .reward_calculator import ExecutionResults
from src.data import Task


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


class CodeSandbox:
    """Isolated execution environment for generated code."""
    
    def __init__(
        self,
        timeout: float = 5.0,
        memory_limit: int = 512 * 1024 * 1024,  # 512 MB
        enable_restrictions: bool = True
    ):
        """
        Initialize code sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            memory_limit: Maximum memory usage in bytes
            enable_restrictions: Whether to enable security restrictions
        """
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.enable_restrictions = enable_restrictions
    
    def execute(self, code: str, task: Task) -> ExecutionResults:
        """
        Execute generated code and evaluate against task examples.
        
        Args:
            code: Generated Python code
            task: Task with test examples
            
        Returns:
            ExecutionResults with test outcomes
        """
        # First validate syntax
        if not self.validate_syntax(code):
            return ExecutionResults(
                success=False,
                outputs=[],
                errors="Syntax error in generated code",
                syntax_valid=False
            )
        
        # Execute code and test against examples
        try:
            # Extract the function from code
            namespace = {}
            
            with self._timeout_context(self.timeout):
                with self._memory_limit_context(self.memory_limit):
                    with self._restricted_context():
                        exec(code, namespace)
            
            # Find the main function (assume it's the first function defined)
            func = self._extract_function(namespace)
            
            if func is None:
                return ExecutionResults(
                    success=False,
                    outputs=[],
                    errors="No function found in generated code"
                )
            
            # Test against examples
            results = self._test_function(func, task)
            return results
            
        except TimeoutException:
            return ExecutionResults(
                success=False,
                outputs=[],
                errors="Execution timeout",
                timeout=True
            )
        except MemoryError:
            return ExecutionResults(
                success=False,
                outputs=[],
                errors="Memory limit exceeded",
                memory_exceeded=True
            )
        except Exception as e:
            return ExecutionResults(
                success=False,
                outputs=[],
                errors=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
    
    def validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax.
        
        Args:
            code: Python code string
            
        Returns:
            True if syntax is valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _extract_function(self, namespace: Dict) -> Optional[callable]:
        """
        Extract the main function from namespace.
        
        Args:
            namespace: Execution namespace
            
        Returns:
            Function object or None
        """
        # Look for callable functions (not built-ins)
        for name, obj in namespace.items():
            if (callable(obj) and 
                not name.startswith('_') and 
                name not in ['__builtins__', '__name__', '__doc__', '__package__', '__loader__', '__spec__']):
                return obj
        return None
    
    def _test_function(self, func: callable, task: Task) -> ExecutionResults:
        """
        Test function against task examples.
        
        Args:
            func: Function to test
            task: Task with examples
            
        Returns:
            ExecutionResults with test outcomes
        """
        outputs = []
        passed = 0
        total = 0
        errors = []
        
        # Test on training examples
        for example in task.train_examples:
            total += 1
            try:
                with self._timeout_context(self.timeout):
                    result = func(example.input_grid)
                    outputs.append(result)
                    
                    # Check if output matches expected
                    if self._grids_equal(result, example.output_grid):
                        passed += 1
                    
            except Exception as e:
                errors.append(f"Example {total}: {type(e).__name__}: {str(e)}")
                outputs.append(None)
        
        test_pass_rate = passed / total if total > 0 else 0.0
        success = test_pass_rate == 1.0
        
        error_msg = "\n".join(errors) if errors else None
        
        return ExecutionResults(
            success=success,
            outputs=outputs,
            errors=error_msg,
            test_pass_rate=test_pass_rate,
            syntax_valid=True
        )
    
    def _grids_equal(self, grid1: Any, grid2: List[List[int]]) -> bool:
        """
        Check if two grids are equal.
        
        Args:
            grid1: First grid
            grid2: Second grid
            
        Returns:
            True if grids are equal
        """
        if not isinstance(grid1, list):
            return False
        
        if len(grid1) != len(grid2):
            return False
        
        for row1, row2 in zip(grid1, grid2):
            if not isinstance(row1, list):
                return False
            if len(row1) != len(row2):
                return False
            if row1 != row2:
                return False
        
        return True
    
    @contextmanager
    def _timeout_context(self, seconds: float):
        """
        Context manager for timeout.
        
        Args:
            seconds: Timeout in seconds
        """
        def timeout_handler(signum, frame):
            raise TimeoutException("Code execution timed out")
        
        # Set up signal handler (Unix only)
        if sys.platform != 'win32':
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            if sys.platform != 'win32':
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    @contextmanager
    def _memory_limit_context(self, max_memory: int):
        """
        Context manager for memory limit.
        
        Args:
            max_memory: Maximum memory in bytes
        """
        original_limit = None
        
        if sys.platform != 'win32' and self.enable_restrictions:
            try:
                # Set memory limit (Unix only)
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                original_limit = (soft, hard)
                # Only set if our limit is less than current hard limit
                if max_memory < hard or hard == resource.RLIM_INFINITY:
                    resource.setrlimit(resource.RLIMIT_AS, (max_memory, hard))
            except (ValueError, OSError):
                # Memory limiting not supported on this system
                pass
        
        try:
            yield
        finally:
            if original_limit and sys.platform != 'win32' and self.enable_restrictions:
                try:
                    # Restore original limit
                    resource.setrlimit(resource.RLIMIT_AS, original_limit)
                except (ValueError, OSError):
                    pass
    
    @contextmanager
    def _restricted_context(self):
        """
        Context manager for restricted execution.
        
        Restricts access to dangerous operations.
        """
        # For now, just a placeholder
        # In production, you'd want to use more sophisticated sandboxing
        # like Docker containers, separate processes, or RestrictedPython
        
        try:
            yield
        finally:
            pass
    
    def capture_output(self, code: str, inputs: List[Any]) -> List[Any]:
        """
        Execute code and capture outputs for given inputs.
        
        Args:
            code: Python code
            inputs: List of input values
            
        Returns:
            List of output values
        """
        outputs = []
        
        try:
            namespace = {}
            exec(code, namespace)
            func = self._extract_function(namespace)
            
            if func is None:
                return outputs
            
            for inp in inputs:
                try:
                    with self._timeout_context(self.timeout):
                        output = func(inp)
                        outputs.append(output)
                except Exception:
                    outputs.append(None)
        
        except Exception:
            pass
        
        return outputs
