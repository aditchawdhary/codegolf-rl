"""
Advantage estimation using Generalized Advantage Estimation (GAE).
"""

import torch
from typing import List
from .trajectory import Trajectory


class AdvantageEstimator:
    """Implements GAE for advantage estimation."""
    
    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        """
        Initialize advantage estimator.
        
        Args:
            gamma: Discount factor
            lambda_: GAE lambda parameter
        """
        self.gamma = gamma
        self.lambda_ = lambda_
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        next_value: float = 0.0
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.
        
        GAE formula: A_t = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value of next state (0 for terminal)
            
        Returns:
            Advantages tensor
        """
        advantages = []
        gae = 0.0
        
        # Convert values to floats
        values_float = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values_float[t + 1]
            
            # TD error: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val - values_float[t]
            
            # GAE: A_t = delta_t + gamma*lambda*A_{t+1}
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def compute_returns(
        self,
        rewards: List[float],
        gamma: float = None
    ) -> torch.Tensor:
        """
        Compute discounted returns.
        
        Args:
            rewards: List of rewards
            gamma: Discount factor (uses self.gamma if None)
            
        Returns:
            Returns tensor
        """
        if gamma is None:
            gamma = self.gamma
        
        returns = []
        R = 0.0
        
        # Compute returns backwards
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        return torch.tensor(returns, dtype=torch.float32)
    
    def compute_advantages_and_returns(
        self,
        trajectory: Trajectory,
        next_value: float = 0.0
    ) -> tuple:
        """
        Compute both advantages and returns for a trajectory.
        
        Args:
            trajectory: Trajectory object
            next_value: Value of next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = self.compute_gae(
            trajectory.rewards,
            trajectory.values,
            next_value
        )
        
        returns = self.compute_returns(trajectory.rewards)
        
        return advantages, returns
