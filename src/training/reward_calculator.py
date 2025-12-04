"""
Reward calculator for code generation tasks.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ExecutionResults:
    """Results from code execution."""
    success: bool
    outputs: list
    errors: Optional[str] = None
    execution_time: float = 0.0
    test_pass_rate: float = 0.0
    syntax_valid: bool = True
    timeout: bool = False
    memory_exceeded: bool = False


class RewardCalculator:
    """Computes reward signals for RL training."""
    
    def __init__(
        self,
        max_reward: float = 1.0,
        syntax_error_penalty: float = -0.5,
        runtime_error_penalty: float = -0.3,
        timeout_penalty: float = -0.4,
        memory_penalty: float = -0.4,
        code_length_weight: float = 0.1,
        max_code_length: int = 1000
    ):
        """
        Initialize reward calculator.
        
        Args:
            max_reward: Maximum reward for perfect solution
            syntax_error_penalty: Penalty for syntax errors
            runtime_error_penalty: Penalty for runtime errors
            timeout_penalty: Penalty for timeout
            memory_penalty: Penalty for memory exceeded
            code_length_weight: Weight for code length component
            max_code_length: Maximum expected code length for normalization
        """
        self.max_reward = max_reward
        self.syntax_error_penalty = syntax_error_penalty
        self.runtime_error_penalty = runtime_error_penalty
        self.timeout_penalty = timeout_penalty
        self.memory_penalty = memory_penalty
        self.code_length_weight = code_length_weight
        self.max_code_length = max_code_length
        
        # Track reward statistics for normalization
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
    
    def compute_reward(
        self,
        code: str,
        results: ExecutionResults,
        task_difficulty: float = 1.0
    ) -> float:
        """
        Compute reward for generated code.
        
        Args:
            code: Generated code string
            results: Execution results
            task_difficulty: Task difficulty multiplier (0-1)
            
        Returns:
            Reward value
        """
        # Start with base reward from test pass rate
        reward = self.compute_test_pass_reward(results)
        
        # Add code quality reward
        quality_reward = self.compute_code_quality_reward(code)
        reward += quality_reward * self.code_length_weight
        
        # Apply error penalties
        if not results.syntax_valid:
            reward += self.syntax_error_penalty
        elif results.timeout:
            reward += self.timeout_penalty
        elif results.memory_exceeded:
            reward += self.memory_penalty
        elif results.errors and not results.success:
            # Runtime error
            error_severity = self._assess_error_severity(results.errors)
            reward += self.runtime_error_penalty * error_severity
        
        # Scale by task difficulty (harder tasks give more reward)
        reward *= (0.5 + 0.5 * task_difficulty)
        
        # Track for normalization
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        return reward
    
    def compute_test_pass_reward(self, results: ExecutionResults) -> float:
        """
        Compute reward based on test pass rate.
        
        Args:
            results: Execution results
            
        Returns:
            Reward from test passing
        """
        if not results.syntax_valid:
            return 0.0
        
        # Linear reward based on pass rate
        # 100% pass rate = max_reward
        # 0% pass rate = 0
        return self.max_reward * results.test_pass_rate
    
    def compute_code_quality_reward(self, code: str) -> float:
        """
        Compute reward based on code quality (primarily length).
        
        Shorter code is better (code golf!), but we normalize to avoid
        extreme penalties for reasonable solutions.
        
        Args:
            code: Generated code string
            
        Returns:
            Quality reward (0-1, higher is better)
        """
        code_length = len(code.strip())
        
        if code_length == 0:
            return 0.0
        
        # Normalize length (shorter is better)
        # Use sigmoid-like function to avoid extreme values
        normalized_length = code_length / self.max_code_length
        quality = 1.0 / (1.0 + normalized_length)
        
        return quality
    
    def normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using running statistics.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized reward
        """
        if len(self.reward_history) < 10:
            # Not enough data for normalization
            return reward
        
        # Update statistics
        self.reward_mean = np.mean(self.reward_history)
        self.reward_std = np.std(self.reward_history)
        
        if self.reward_std < 1e-6:
            return reward - self.reward_mean
        
        # Z-score normalization
        normalized = (reward - self.reward_mean) / self.reward_std
        return normalized
    
    def _assess_error_severity(self, error_message: str) -> float:
        """
        Assess error severity from error message.
        
        Args:
            error_message: Error message string
            
        Returns:
            Severity multiplier (0-1, higher is more severe)
        """
        if not error_message:
            return 0.0
        
        error_lower = error_message.lower()
        
        # More severe errors
        severe_errors = [
            'nameerror',
            'attributeerror',
            'typeerror',
            'valueerror',
            'indexerror',
            'keyerror',
        ]
        
        # Less severe errors
        mild_errors = [
            'assertionerror',
            'stopiteration',
        ]
        
        for severe in severe_errors:
            if severe in error_lower:
                return 1.0
        
        for mild in mild_errors:
            if mild in error_lower:
                return 0.5
        
        # Unknown error type
        return 0.7
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """
        Get reward statistics.
        
        Returns:
            Dictionary with reward statistics
        """
        if not self.reward_history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
        
        return {
            "mean": float(np.mean(self.reward_history)),
            "std": float(np.std(self.reward_history)),
            "min": float(np.min(self.reward_history)),
            "max": float(np.max(self.reward_history)),
            "count": len(self.reward_history)
        }
    
    def reset_statistics(self):
        """Reset reward statistics."""
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
