"""
Policy network for RL training.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from .code_llm import CodeLLM
from .config import ModelConfig


class PolicyNetwork(CodeLLM):
    """Policy network for RL training, extends CodeLLM."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize policy network.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
    
    def sample_action(
        self,
        state: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: int = 512
    ) -> Tuple[str, torch.Tensor]:
        """
        Sample an action (code generation) from the policy.
        
        Args:
            state: Input state (prompt)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_code, log_prob)
        """
        # Generate code
        code = self.generate(
            state,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        # Compute log probability of the generated code
        log_prob = self.compute_log_prob(state, code)
        
        return code, log_prob
    
    def compute_log_prob(self, state: str, action: str, requires_grad: bool = True) -> torch.Tensor:
        """
        Compute log probability of an action given a state.
        
        Args:
            state: Input state (prompt)
            action: Action (generated code)
            requires_grad: Whether to track gradients (True for training, False for inference)
            
        Returns:
            Log probability tensor
        """
        # Get log probabilities for each token
        token_log_probs = self.get_logprobs(state, action, requires_grad=requires_grad)
        
        # Sum log probabilities (log of product = sum of logs)
        total_log_prob = token_log_probs.sum()
        
        return total_log_prob
    
    def get_entropy(self, state: str, num_samples: int = 5) -> torch.Tensor:
        """
        Estimate policy entropy for exploration tracking.
        
        Args:
            state: Input state (prompt)
            num_samples: Number of samples for entropy estimation
            
        Returns:
            Entropy estimate
        """
        # Sample multiple actions
        log_probs = []
        
        for _ in range(num_samples):
            _, log_prob = self.sample_action(state, temperature=1.0)
            log_probs.append(log_prob)
        
        # Estimate entropy from samples
        # H(p) ≈ -E[log p(x)]
        log_probs_tensor = torch.stack(log_probs)
        entropy = -log_probs_tensor.mean()
        
        return entropy
    
    def get_action_probabilities(
        self,
        state: str,
        actions: list
    ) -> torch.Tensor:
        """
        Get probabilities for a list of actions.
        
        Args:
            state: Input state
            actions: List of action strings
            
        Returns:
            Tensor of log probabilities
        """
        log_probs = []
        
        for action in actions:
            log_prob = self.compute_log_prob(state, action)
            log_probs.append(log_prob)
        
        return torch.stack(log_probs)
