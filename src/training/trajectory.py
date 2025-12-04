"""
Trajectory data structures for RL training.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class Trajectory:
    """A trajectory of states, actions, and rewards."""
    states: List[str] = field(default_factory=list)  # Prompts
    actions: List[str] = field(default_factory=list)  # Generated code
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    
    def add_step(
        self,
        state: str,
        action: str,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor
    ):
        """Add a step to the trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def __len__(self):
        """Return trajectory length."""
        return len(self.states)
    
    def is_empty(self):
        """Check if trajectory is empty."""
        return len(self.states) == 0


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    learning_rate: float = 3e-4
    batch_size: int = 4
    num_epochs: int = 4
    gamma: float = 0.99  # Discount factor
    lambda_: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_trajectories_per_update: int = 8
    max_steps_per_episode: int = 1
    target_kl: float = 0.01  # Early stopping if KL divergence too high
