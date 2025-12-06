#!/usr/bin/env python
"""
Distributed training script using Ray for multi-GPU parallelization.
"""

import argparse
import ray
from pathlib import Path
import torch
from typing import List
from src.models import ModelConfig, PolicyNetwork, ValueNetwork
from src.data import TaskLoader, DifficultyAnalyzer, Task
from src.training import (
    PPOConfig,
    RewardCalculator,
    CodeSandbox,
    Trajectory
)
from src.experiments import CheckpointManager, MetricsTracker, EarlyStopping
from src.evaluation import InferencePipeline, PerformanceEvaluator


@ray.remote(num_gpus=1)
class TrajectoryCollector:
    """Ray actor for collecting trajectories on a single GPU."""
    
    def __init__(self, model_config: ModelConfig, ppo_config: PPOConfig, gpu_id: int):
        """Initialize trajectory collector on specific GPU."""
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        
        # Update model config for this GPU
        model_config.device = self.device
        
        # Initialize models
        self.policy = PolicyNetwork(model_config)
        self.value = ValueNetwork(self.policy, hidden_size=768)
        
        # Initialize components
        self.reward_calculator = RewardCalculator()
        self.sandbox = CodeSandbox(timeout=5.0)
        self.ppo_config = ppo_config
        
        from src.data import TaskFormatter
        self.formatter = TaskFormatter()
        
        print(f"✓ Trajectory collector initialized on GPU {gpu_id}")
    
    def collect_trajectories(self, tasks: List[Task]) -> List[Trajectory]:
        """Collect trajectories for given tasks."""
        trajectories = []
        
        for task in tasks:
            trajectory = Trajectory()
            
            # Format task as prompt
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
    
    def update_weights(self, policy_state_dict, value_state_dict):
        """Update model weights from central coordinator."""
        self.policy.model.load_state_dict(policy_state_dict)
        self.value.value_head.load_state_dict(value_state_dict)
    
    def get_weights(self):
        """Get current model weights."""
        return {
            "policy": self.policy.model.state_dict(),
            "value": self.value.value_head.state_dict()
        }


class DistributedPPOTrainer:
    """Distributed PPO trainer using Ray."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        ppo_config: PPOConfig,
        num_gpus: int = 8
    ):
        """Initialize distributed trainer."""
        self.model_config = model_config
        self.ppo_config = ppo_config
        self.num_gpus = num_gpus
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_gpus=num_gpus)
        
        print(f"Initializing {num_gpus} trajectory collectors...")
        
        # Create trajectory collectors (one per GPU)
        self.collectors = [
            TrajectoryCollector.remote(model_config, ppo_config, gpu_id)
            for gpu_id in range(num_gpus)
        ]
        
        # Initialize central policy and value networks (on GPU 0)
        model_config.device = "cuda:0"
        self.policy = PolicyNetwork(model_config)
        self.value = ValueNetwork(self.policy, hidden_size=768)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.model.parameters(),
            lr=ppo_config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.value_head.parameters(),
            lr=ppo_config.learning_rate
        )
        
        # Advantage estimator
        from src.training import AdvantageEstimator
        self.advantage_estimator = AdvantageEstimator(
            gamma=ppo_config.gamma,
            lambda_=ppo_config.lambda_
        )
        
        self.step = 0
        print("✓ Distributed PPO trainer initialized")
    
    def collect_trajectories_parallel(self, tasks: List[Task]) -> List[Trajectory]:
        """Collect trajectories in parallel across all GPUs."""
        # Split tasks across GPUs
        tasks_per_gpu = len(tasks) // self.num_gpus
        task_chunks = [
            tasks[i * tasks_per_gpu:(i + 1) * tasks_per_gpu]
            for i in range(self.num_gpus)
        ]
        
        # Handle remainder
        remainder = len(tasks) % self.num_gpus
        if remainder > 0:
            task_chunks[-1].extend(tasks[-remainder:])
        
        # Broadcast current weights to all collectors
        policy_weights = self.policy.model.state_dict()
        value_weights = self.value.value_head.state_dict()
        
        update_futures = [
            collector.update_weights.remote(policy_weights, value_weights)
            for collector in self.collectors
        ]
        ray.get(update_futures)
        
        # Collect trajectories in parallel
        trajectory_futures = [
            collector.collect_trajectories.remote(chunk)
            for collector, chunk in zip(self.collectors, task_chunks)
        ]
        
        # Gather results
        trajectory_lists = ray.get(trajectory_futures)
        
        # Flatten list of lists
        all_trajectories = []
        for traj_list in trajectory_lists:
            all_trajectories.extend(traj_list)
        
        return all_trajectories
    
    def train_step(self, tasks: List[Task]):
        """Perform one distributed training step."""
        # Collect trajectories in parallel
        trajectories = self.collect_trajectories_parallel(tasks)
        
        if not trajectories or all(t.is_empty() for t in trajectories):
            return self._empty_metrics()
        
        # Compute advantages (on central GPU)
        for trajectory in trajectories:
            advantages, returns = self.advantage_estimator.compute_advantages_and_returns(
                trajectory,
                next_value=0.0
            )
            trajectory.advantages = advantages
            trajectory.returns = returns
        
        # Update networks (on central GPU)
        from src.training.ppo_trainer import PPOTrainer
        # Reuse the update logic from single-GPU trainer
        policy_loss, value_loss, kl_div = self._update_networks(trajectories)
        
        # Compute metrics
        metrics = self._compute_metrics(trajectories, policy_loss, value_loss, kl_div)
        
        self.step += 1
        return metrics
    
    def _update_networks(self, trajectories):
        """Update policy and value networks."""
        import torch.nn.functional as F
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        
        # Flatten trajectories
        states, actions, old_log_probs, advantages, returns = self._flatten_trajectories(trajectories)
        
        # Get device and dtype
        device = self.policy.model.device
        dtype = next(self.policy.model.parameters()).dtype
        
        # Move to device
        old_log_probs = old_log_probs.to(device=device, dtype=dtype)
        advantages = advantages.to(device=device, dtype=dtype)
        returns = returns.to(device=device, dtype=dtype)
        
        # PPO epochs
        for epoch in range(self.ppo_config.num_epochs):
            # Compute new log probs and values
            new_log_probs = []
            new_values = []
            
            for state, action in zip(states, actions):
                log_prob = self.policy.compute_log_prob(state, action)
                value = self.value.estimate_value(state)
                new_log_probs.append(log_prob)
                new_values.append(value)
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.ppo_config.clip_epsilon,
                1.0 + self.ppo_config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # KL divergence
            kl_div = (old_log_probs - new_log_probs).mean().item()
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(),
                self.ppo_config.max_grad_norm
            )
            self.policy_optimizer.step()
            
            # Update value (recompute for clean backward)
            self.value_optimizer.zero_grad()
            new_values_update = []
            for state, action in zip(states, actions):
                value = self.value.estimate_value(state)
                new_values_update.append(value)
            new_values_update = torch.stack(new_values_update)
            value_loss_update = F.mse_loss(new_values_update, returns)
            value_loss_update.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value.value_head.parameters(),
                self.ppo_config.max_grad_norm
            )
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += abs(kl_div)
            num_updates += 1
            
            # Early stopping
            if abs(kl_div) > self.ppo_config.target_kl:
                break
        
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0.0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0.0
        avg_kl = total_kl / num_updates if num_updates > 0 else 0.0
        
        return avg_policy_loss, avg_value_loss, avg_kl
    
    def _flatten_trajectories(self, trajectories):
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
    
    def _compute_metrics(self, trajectories, policy_loss, value_loss, kl_div):
        """Compute training metrics."""
        from src.training.ppo_trainer import TrainingMetrics
        
        all_rewards = []
        successes = 0
        total = 0
        
        for traj in trajectories:
            if not traj.is_empty():
                all_rewards.extend(traj.rewards)
                successes += sum(1 for r in traj.rewards if r > 0)
                total += len(traj.rewards)
        
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        success_rate = successes / total if total > 0 else 0.0
        
        return TrainingMetrics(
            step=self.step,
            policy_loss=policy_loss,
            value_loss=value_loss,
            average_reward=avg_reward,
            success_rate=success_rate,
            entropy=0.0,
            gradient_norm=0.0,
            learning_rate=self.ppo_config.learning_rate,
            kl_divergence=kl_div
        )
    
    def _empty_metrics(self):
        """Return empty metrics."""
        from src.training.ppo_trainer import TrainingMetrics
        return TrainingMetrics(
            step=self.step,
            policy_loss=0.0,
            value_loss=0.0,
            average_reward=0.0,
            success_rate=0.0,
            entropy=0.0,
            gradient_norm=0.0,
            learning_rate=self.ppo_config.learning_rate
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed training with Ray")
    
    parser.add_argument("--model-name", type=str, default="codellama/CodeLlama-7b-Python-hf")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit", None])
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64, help="Total batch size across all GPUs")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--task-dir", type=str, default="google-code-golf-2025")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--experiment-name", type=str, default="distributed_training")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    
    return parser.parse_args()


def main():
    """Main distributed training function."""
    args = parse_args()
    
    print("="*60)
    print("Distributed LLM-RL Code Golf Training")
    print(f"Using {args.num_gpus} GPUs")
    print("="*60)
    
    # Load tasks
    print(f"\nLoading tasks from {args.task_dir}...")
    loader = TaskLoader(task_dir=args.task_dir)
    all_tasks = loader.load_all_tasks()
    print(f"✓ Loaded {len(all_tasks)} tasks")
    
    # Analyze difficulty
    print("\nAnalyzing task difficulty...")
    analyzer = DifficultyAnalyzer()
    for task in all_tasks:
        analyzer.compute_complexity_score(task)
    
    categories = analyzer.categorize_by_difficulty(all_tasks)
    print(f"✓ Categorized: {len(categories['easy'])} easy, "
          f"{len(categories['medium'])} medium, {len(categories['hard'])} hard")
    
    # Split train/val
    train_tasks = all_tasks[:int(0.8 * len(all_tasks))]
    val_tasks = all_tasks[int(0.8 * len(all_tasks)):]
    print(f"✓ Split: {len(train_tasks)} train, {len(val_tasks)} val")
    
    # Initialize model config
    model_config = ModelConfig(
        model_name=args.model_name,
        device="cuda:0",  # Will be overridden per GPU
        quantization=args.quantization,
        temperature=0.7
    )
    
    # Initialize PPO config
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
    
    # Initialize distributed trainer
    print(f"\nInitializing distributed trainer with {args.num_gpus} GPUs...")
    trainer = DistributedPPOTrainer(model_config, ppo_config, num_gpus=args.num_gpus)
    
    # Initialize checkpointing and logging
    ckpt_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir, keep_last_n=5)
    metrics_tracker = MetricsTracker(log_dir=args.log_dir, experiment_name=args.experiment_name)
    early_stop = EarlyStopping(patience=20, mode="max")
    
    print(f"\n{'='*60}")
    print("Starting Distributed Training")
    print(f"{'='*60}\n")
    
    best_val_score = 0.0
    
    # Training loop
    import random
    for step in range(args.num_steps):
        # Sample tasks (larger batch for distributed training)
        batch_tasks = random.sample(train_tasks, min(args.batch_size, len(train_tasks)))
        
        # Distributed training step
        metrics = trainer.train_step(batch_tasks)
        
        # Log metrics
        metrics_dict = {
            "policy_loss": metrics.policy_loss,
            "value_loss": metrics.value_loss,
            "average_reward": metrics.average_reward,
            "success_rate": metrics.success_rate,
            "kl_divergence": metrics.kl_divergence
        }
        metrics_tracker.log_training_step(step, metrics_dict)
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}/{args.num_steps} | "
                  f"Reward: {metrics.average_reward:.3f} | "
                  f"Success: {metrics.success_rate:.2%} | "
                  f"Loss: {metrics.policy_loss:.3f}")
        
        # Evaluate
        if step % args.eval_every == 0 and step > 0:
            print(f"\nEvaluating at step {step}...")
            sandbox = CodeSandbox(timeout=5.0)
            inference = InferencePipeline(trainer.policy, sandbox)
            evaluator = PerformanceEvaluator(inference)
            
            eval_metrics = evaluator.evaluate(val_tasks[:10], num_samples=1, strategy="greedy")
            
            print(f"  Val Success Rate: {eval_metrics['success_rate']:.2%}")
            metrics_tracker.log_evaluation(step, eval_metrics)
            
            val_score = eval_metrics['success_rate']
            is_best = val_score > best_val_score
            if is_best:
                best_val_score = val_score
                print(f"  ✓ New best model! Success rate: {val_score:.2%}")
            
            if early_stop(val_score):
                print(f"\nEarly stopping triggered at step {step}")
                break
        
        # Save checkpoint
        if step % args.save_every == 0 and step > 0:
            print(f"\nSaving checkpoint at step {step}...")
            ckpt_path = ckpt_manager.save_checkpoint(
                trainer.policy,
                trainer.value,
                trainer.policy_optimizer,
                trainer.value_optimizer,
                step,
                metrics_dict,
                ppo_config.__dict__,
                is_best=False
            )
            print(f"✓ Checkpoint saved: {ckpt_path}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
