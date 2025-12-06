"""
PPO (Proximal Policy Optimization) trainer for LLM code generation.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from dataclasses import dataclass
from src.models import PolicyNetwork, ValueNetwork
from src.data import Task, TaskFormatter
from .trajectory import Trajectory, PPOConfig
from .advantage_estimator import AdvantageEstimator
from .reward_calculator import RewardCalculator
from .code_sandbox import CodeSandbox


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""
    step: int
    policy_loss: float
    value_loss: float
    average_reward: float
    success_rate: float
    entropy: float
    gradient_norm: float
    learning_rate: float
    kl_divergence: float = 0.0


class PPOTrainer:
    """Main PPO training loop."""
    
    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        config: PPOConfig,
        reward_calculator: RewardCalculator,
        sandbox: CodeSandbox
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: Policy network
            value: Value network
            config: PPO configuration
            reward_calculator: Reward calculator
            sandbox: Code execution sandbox
        """
        self.policy = policy
        self.value = value
        self.config = config
        self.reward_calculator = reward_calculator
        self.sandbox = sandbox
        
        # Advantage estimator
        self.advantage_estimator = AdvantageEstimator(
            gamma=config.gamma,
            lambda_=config.lambda_
        )
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.model.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.value_head.parameters(),
            lr=config.learning_rate
        )
        
        # Training state
        self.step = 0
        self.formatter = TaskFormatter()
    
    def train_step(self, tasks: List[Task]) -> TrainingMetrics:
        """
        Perform one PPO training step.
        
        Args:
            tasks: List of tasks to train on
            
        Returns:
            Training metrics
        """
        # Collect trajectories
        trajectories = self.collect_trajectories(tasks)
        
        if not trajectories or all(t.is_empty() for t in trajectories):
            # No valid trajectories collected
            return TrainingMetrics(
                step=self.step,
                policy_loss=0.0,
                value_loss=0.0,
                average_reward=0.0,
                success_rate=0.0,
                entropy=0.0,
                gradient_norm=0.0,
                learning_rate=self.config.learning_rate
            )
        
        # Compute advantages
        for trajectory in trajectories:
            advantages, returns = self.advantage_estimator.compute_advantages_and_returns(
                trajectory,
                next_value=0.0
            )
            trajectory.advantages = advantages
            trajectory.returns = returns
        
        # Update policy and value function
        policy_loss, value_loss, kl_div = self.update_networks(trajectories)
        
        # Compute metrics
        metrics = self._compute_metrics(
            trajectories,
            policy_loss,
            value_loss,
            kl_div
        )
        
        self.step += 1
        return metrics
    
    def collect_trajectories(self, tasks: List[Task]) -> List[Trajectory]:
        """
        Collect trajectories by running policy on tasks.
        
        Args:
            tasks: List of tasks
            
        Returns:
            List of trajectories
        """
        trajectories = []
        
        for task in tasks[:self.config.num_trajectories_per_update]:
            trajectory = Trajectory()
            
            # Format task as prompt (state)
            state = self.formatter.format_prompt(task, include_test=False)
            
            # Sample action from policy
            action, log_prob = self.policy.sample_action(
                state,
                max_new_tokens=512
            )
            
            # Estimate value
            value = self.value.estimate_value(state)
            
            # Execute code and get reward
            results = self.sandbox.execute(action, task)
            reward = self.reward_calculator.compute_reward(
                action,
                results,
                task_difficulty=task.difficulty_score / 100.0
            )
            
            # Add to trajectory
            trajectory.add_step(state, action, reward, log_prob, value)
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def update_networks(self, trajectories: List[Trajectory]) -> tuple:
        """
        Update policy and value networks using PPO.
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Tuple of (policy_loss, value_loss, kl_divergence)
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        
        # Flatten trajectories into batches
        states, actions, old_log_probs, advantages, returns = self._flatten_trajectories(
            trajectories
        )
        
        # Get device and dtype from policy model
        device = self.policy.model.device
        dtype = next(self.policy.model.parameters()).dtype
        
        # Move tensors to device and convert dtype
        old_log_probs = old_log_probs.to(device=device, dtype=dtype)
        advantages = advantages.to(device=device, dtype=dtype)
        returns = returns.to(device=device, dtype=dtype)
        
        # Ensure models are in training mode
        self.policy.model.train()
        self.value.value_head.train()
        
        # PPO epochs
        for epoch in range(self.config.num_epochs):
            # Compute new log probs and values with gradient tracking
            new_log_probs = []
            new_values = []
            
            for state, action in zip(states, actions):
                log_prob = self.policy.compute_log_prob(state, action)
                value = self.value.estimate_value(state)
                new_log_probs.append(log_prob)
                new_values.append(value)
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            
            # Verify gradients are enabled
            if not new_log_probs.requires_grad:
                raise RuntimeError("Log probs don't require grad - check model training mode")
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss (detach for separate backward pass)
            value_loss = F.mse_loss(new_values, returns)
            
            # Compute KL divergence for early stopping
            kl_div = (old_log_probs - new_log_probs).mean().item()
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.policy_optimizer.step()
            
            # Update value network separately (recompute to avoid graph issues)
            self.value_optimizer.zero_grad()
            # Recompute values for clean backward pass
            new_values_for_value_update = []
            for state, action in zip(states, actions):
                value = self.value.estimate_value(state)
                new_values_for_value_update.append(value)
            new_values_for_value_update = torch.stack(new_values_for_value_update)
            value_loss_for_update = F.mse_loss(new_values_for_value_update, returns)
            value_loss_for_update.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value.value_head.parameters(),
                self.config.max_grad_norm
            )
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += abs(kl_div)
            num_updates += 1
            
            # Early stopping if KL divergence too high
            if abs(kl_div) > self.config.target_kl:
                break
        
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0.0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0.0
        avg_kl = total_kl / num_updates if num_updates > 0 else 0.0
        
        return avg_policy_loss, avg_value_loss, avg_kl
    
    def _flatten_trajectories(self, trajectories: List[Trajectory]) -> tuple:
        """Flatten trajectories into batches."""
        states = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []
        
        for traj in trajectories:
            if traj.is_empty():
                continue
            
            states.extend(traj.states)
            actions.extend(traj.actions)
            old_log_probs.extend(traj.log_probs)
            
            if traj.advantages is not None:
                for adv in traj.advantages:
                    advantages.append(adv)
            
            if traj.returns is not None:
                for ret in traj.returns:
                    returns.append(ret)
        
        old_log_probs = torch.stack(old_log_probs)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, old_log_probs, advantages, returns
    
    def _compute_metrics(
        self,
        trajectories: List[Trajectory],
        policy_loss: float,
        value_loss: float,
        kl_div: float
    ) -> TrainingMetrics:
        """Compute training metrics."""
        all_rewards = []
        successes = 0
        total = 0
        
        for traj in trajectories:
            if not traj.is_empty():
                all_rewards.extend(traj.rewards)
                # Success if reward is positive
                successes += sum(1 for r in traj.rewards if r > 0)
                total += len(traj.rewards)
        
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        success_rate = successes / total if total > 0 else 0.0
        
        # Estimate entropy (simplified)
        entropy = 0.0  # Would need multiple samples to estimate properly
        
        return TrainingMetrics(
            step=self.step,
            policy_loss=policy_loss,
            value_loss=value_loss,
            average_reward=avg_reward,
            success_rate=success_rate,
            entropy=entropy,
            gradient_norm=0.0,  # Would track from update
            learning_rate=self.config.learning_rate,
            kl_divergence=kl_div
        )
