#!/usr/bin/env python
"""
Optimized Distributed Training Script with Batched Inference

Key improvements over the original:
1. Batched inference - process multiple states at once on the model server
2. Separate inference and execution phases for better parallelism
3. Clearer separation of concerns between model server and workers
4. Better GPU utilization through batching

Architecture:
- SharedModelServer: Holds the model sharded across 8 GPUs, processes batches
- TrajectoryCollector: CPU workers that execute code and compute rewards
- DistributedPPOTrainer: Orchestrates everything
"""

import argparse
import ray
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from src.models import ModelConfig, PolicyNetwork, ValueNetwork
from src.data import TaskLoader, DifficultyAnalyzer, Task, TaskFormatter
from src.training import (
    PPOConfig,
    RewardCalculator,
    CodeSandbox,
    Trajectory,
    AdvantageEstimator,
    TrainingMetrics
)
from src.experiments import CheckpointManager, MetricsTracker, EarlyStopping
from src.evaluation import InferencePipeline, PerformanceEvaluator


# ============================================================================
# SHARED MODEL SERVER (8 GPUs with Tensor Parallelism)
# ============================================================================

@ray.remote(num_gpus=8)
class SharedModelServer:
    """
    Shared model server that uses ALL 8 GPUs with tensor parallelism.
    
    This is the heart of the system. The model is SHARDED across all 8 GPUs,
    meaning different parts of each layer live on different GPUs.
    
    Key principle: Process requests in BATCHES to maximize GPU utilization.
    Instead of 16 sequential requests, we do 1 batch of 16 states.
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize model with vLLM for fast inference."""
        print("\n" + "="*70)
        print("INITIALIZING SHARED MODEL SERVER WITH VLLM")
        print("="*70)
        print("Loading model with tensor parallelism across 8 GPUs...")
        print(f"Model: {model_config.model_name}")
        print(f"Quantization: {model_config.quantization}")
        
        # Initialize vLLM for fast inference
        self.llm = LLM(
            model=model_config.model_name,
            tensor_parallel_size=8,
            gpu_memory_utilization=0.90,
            max_model_len=2048,
            trust_remote_code=True,
            quantization="bitsandbytes" if model_config.quantization else None,
            dtype="half"
        )
        
        # Sampling params for generation
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_p=0.95,
        )
        
        # Store config
        self.model_config = model_config
        
        print(f"✓ vLLM initialized and model sharded across 8 GPUs")
        print("="*70 + "\n")
    
    # ------------------------------------------------------------------------
    # BATCHED INFERENCE METHODS (Key optimization!)
    # ------------------------------------------------------------------------
    
    def generate_batch(
        self, 
        states: List[str], 
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate actions for a BATCH of states using vLLM.
        
        vLLM handles batching automatically and efficiently.
        
        Args:
            states: List of prompt strings
            max_new_tokens: Max tokens to generate per state
            temperature: Sampling temperature
        
        Returns:
            actions: List of generated code strings
            log_probs: List of log probabilities (placeholder)
        """
        print(f"  [ModelServer] Generating for {len(states)} states with vLLM")
        
        # Update sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=0.95,
        )
        
        # Generate with vLLM (automatically batched and optimized!)
        outputs = self.llm.generate(states, sampling_params)
        
        # Extract generated text
        actions = [output.outputs[0].text for output in outputs]
        
        # Placeholder log probs (vLLM can provide these if needed)
        log_probs = [torch.tensor(0.0, dtype=torch.float32) for _ in actions]
        
        print(f"  [ModelServer] Generated {len(actions)} actions")
        return actions, log_probs
    
    def estimate_values_batch(self, states: List[str]) -> List[torch.Tensor]:
        """
        Estimate values for a BATCH of states.
        
        Similar to generate_batch, but for value estimation.
        """
        print(f"  [ModelServer] estimate_values_batch called for {len(states)} states")
        self.value.value_head.eval()
        
        values = []
        batch_size = 16  # Value estimation is faster, can use larger batches
        
        print(f"  [ModelServer] Estimating values for {len(states)} states")
        
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            
            # For each state, get value estimate
            # Note: This could be further optimized by batching the forward pass
            for state in batch_states:
                value = self.value.estimate_value(state)
                values.append(value)
        
        return values
    
    def compute_log_probs_batch(
        self, 
        states: List[str], 
        actions: List[str]
    ) -> List[torch.Tensor]:
        """
        Compute log probabilities for state-action pairs.
        
        Used during PPO updates to compare new policy vs old policy.
        """
        print(f"  [ModelServer] compute_log_probs_batch called for {len(states)} state-action pairs")
        log_probs = []
        batch_size = 4  # Smaller batch for gradient computation
        
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            
            for state, action in zip(batch_states, batch_actions):
                log_prob = self.policy.compute_log_prob(state, action, requires_grad=True)
                log_probs.append(log_prob)
        
        return log_probs
    
    # ------------------------------------------------------------------------
    # MODE CONTROL
    # ------------------------------------------------------------------------
    
    def set_train_mode(self):
        """Set models to training mode."""
        print(f"  [ModelServer] Setting models to TRAIN mode")
        # Note: vLLM doesn't have train mode, this is for future HF integration
        pass
    
    def set_eval_mode(self):
        """Set models to eval mode."""
        print(f"  [ModelServer] Setting models to EVAL mode")
        # Note: vLLM is always in eval mode for inference
        pass
    
    # ------------------------------------------------------------------------
    # TRAINING UPDATE (PPO)
    # ------------------------------------------------------------------------
    
    def update_policy_value(
        self,
        states: List[str],
        actions: List[str],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        ppo_config: PPOConfig
    ) -> Dict[str, float]:
        """
        Perform PPO update on the model.
        
        This runs on the model server (where the actual model lives).
        Returns metrics about the update.
        """
        print(f"  [ModelServer] update_policy_value called for {len(states)} trajectories")
        self.set_train_mode()
        
        # Move tensors to device
        device = next(self.policy.model.parameters()).device
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        
        # PPO epochs
        for epoch in range(ppo_config.num_epochs):
            # Compute new log probs and values
            new_log_probs = self.compute_log_probs_batch(states, actions)
            new_values = self.estimate_values_batch(states)
            
            new_log_probs = torch.stack(new_log_probs).to(device)
            new_values = torch.stack(new_values).to(device)
            
            # PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - ppo_config.clip_epsilon,
                1.0 + ppo_config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # KL divergence (for early stopping)
            kl_div = (old_log_probs - new_log_probs).mean().item()
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)  # Keep graph for value update
            torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(),
                ppo_config.max_grad_norm
            )
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            # Recompute values for clean backward pass
            new_values_for_update = torch.stack(self.estimate_values_batch(states)).to(device)
            value_loss_update = F.mse_loss(new_values_for_update, returns)
            value_loss_update.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value.value_head.parameters(),
                ppo_config.max_grad_norm
            )
            self.value_optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += abs(kl_div)
            num_updates += 1
            
            # Early stopping based on KL divergence
            if abs(kl_div) > ppo_config.target_kl:
                print(f"    Early stopping at epoch {epoch} (KL={kl_div:.4f})")
                break
        
        # Return average metrics
        return {
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0.0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0.0,
            'kl_divergence': total_kl / num_updates if num_updates > 0 else 0.0,
        }
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def get_state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'policy': self.policy.model.state_dict(),
            'value': self.value.value_head.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.policy.model.load_state_dict(state_dict['policy'])
        self.value.value_head.load_state_dict(state_dict['value'])
        self.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        self.value_optimizer.load_state_dict(state_dict['value_optimizer'])


# ============================================================================
# TRAJECTORY COLLECTOR (CPU Workers)
# ============================================================================

@ray.remote
class TrajectoryCollector:
    """
    CPU-only worker that executes code and computes rewards.
    
    This is separate from the model server because:
    1. Code execution is CPU-bound (running Python code in sandbox)
    2. We want many workers (16+) but only 1 model server
    3. This allows us to parallelize execution while batching inference
    
    Key principle: No model inference happens here - we receive pre-generated
    actions from the model server and just execute them.
    """
    
    def __init__(self, worker_id: int):
        """Initialize trajectory collector."""
        self.worker_id = worker_id
        self.reward_calculator = RewardCalculator()
        self.sandbox = CodeSandbox(timeout=5.0)
        
        print(f"  [Worker {worker_id}] Initialized")
    
    def execute_and_build_trajectory(
        self,
        task: Task,
        state: str,
        action: str,
        log_prob: torch.Tensor,
        value: torch.Tensor
    ) -> Trajectory:
        """
        Execute code and build a trajectory.
        
        This is purely execution - no model inference.
        
        Args:
            task: The coding task
            state: The formatted prompt (for trajectory storage)
            action: The generated code (from model server)
            log_prob: Log probability of the action (from model server)
            value: Value estimate (from model server)
        
        Returns:
            Trajectory with reward computed
        """
        print(f"  [Worker] collect_trajectory called for task")
        trajectory = Trajectory()
        
        # Execute the code in sandbox
        print(f"  [Worker] Executing code in sandbox...")
        results = self.sandbox.execute(action, task)
        
        # Compute reward based on test results
        print(f"  [Worker] Computing reward...")
        reward = self.reward_calculator.compute_reward(
            action,
            results,
            task_difficulty=task.difficulty_score / 100.0 if hasattr(task, 'difficulty_score') else 0.5
        )
        
        # Build trajectory
        print(f"  [Worker] Reward: {reward:.3f}, building trajectory")
        trajectory.add_step(state, action, reward, log_prob, value)
        
        return trajectory


# ============================================================================
# DISTRIBUTED PPO TRAINER (Orchestrator)
# ============================================================================

class DistributedPPOTrainer:
    """
    Orchestrates distributed training with batched inference.
    
    Architecture:
    1. SharedModelServer (1 instance, 8 GPUs): Handles all model inference
    2. TrajectoryCollectors (16 instances, CPU): Execute code and compute rewards
    3. This class: Coordinates everything
    
    Training loop flow:
    1. Collect trajectories:
       a. Format all states (CPU)
       b. Batch inference on model server (8 GPUs working together!)
       c. Distribute execution across workers (16 CPUs in parallel)
    2. Compute advantages and returns (CPU)
    3. Update model on server (8 GPUs)
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        ppo_config: PPOConfig,
        num_workers: int = 16
    ):
        """Initialize distributed trainer."""
        self.model_config = model_config
        self.ppo_config = ppo_config
        self.num_workers = num_workers
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            print("\nInitializing Ray...")
            ray.init(num_gpus=8, num_cpus=num_workers + 4)  # +4 for main process
            print(f"✓ Ray initialized")
        
        # Create shared model server (tensor parallelism across 8 GPUs)
        print("\nCreating shared model server...")
        self.model_server = SharedModelServer.remote(model_config)
        
        # Wait for model to load
        print("Waiting for model to load...")
        ray.get(self.model_server.set_eval_mode.remote())
        
        # Create trajectory collectors (CPU workers)
        print(f"\nCreating {num_workers} trajectory collectors...")
        self.collectors = [
            TrajectoryCollector.remote(worker_id=i)
            for i in range(num_workers)
        ]
        
        # Task formatter
        self.formatter = TaskFormatter()
        
        # Advantage estimator
        self.advantage_estimator = AdvantageEstimator(
            gamma=ppo_config.gamma,
            lambda_=ppo_config.lambda_
        )
        
        self.step = 0
        
        print(f"\n{'='*70}")
        print("DISTRIBUTED TRAINER READY")
        print(f"{'='*70}")
        print(f"Model Server: 1 instance using 8 GPUs (tensor parallel)")
        print(f"Workers: {num_workers} CPU instances")
        print(f"Batch size: {ppo_config.batch_size}")
        print(f"{'='*70}\n")
    
    # ------------------------------------------------------------------------
    # TRAJECTORY COLLECTION (Optimized with batching!)
    # ------------------------------------------------------------------------
    
    def collect_trajectories_parallel(self, tasks: List[Task]) -> List[Trajectory]:
        """
        Collect trajectories with batched inference.
        
        This is THE KEY OPTIMIZATION. Instead of:
          for task in tasks:
              action = model.generate(task)  # 64 sequential calls!
        
        We do:
          actions = model.generate_batch(all_tasks)  # 1 batched call!
        
        This keeps all 8 GPUs busy instead of processing 1 request at a time.
        """
        print(f"\n{'='*70}")
        print(f"COLLECTING TRAJECTORIES (Batch size: {len(tasks)})")
        print(f"{'='*70}")
        
        # Set model to eval mode
        ray.get(self.model_server.set_eval_mode.remote())
        
        # ------------------------------------------------------------------------
        # PHASE 1: Format all states (CPU work)
        # ------------------------------------------------------------------------
        print("Phase 1: Formatting prompts...")
        states = [
            self.formatter.format_prompt(task, include_test=False)
            for task in tasks
        ]
        print(f"  ✓ Formatted {len(states)} prompts")
        
        # ------------------------------------------------------------------------
        # PHASE 2: Batched inference on model server (8 GPUs working together!)
        # ------------------------------------------------------------------------
        print("Phase 2: Batched inference on model server...")
        print(f"  All 8 GPUs will work together on this batch!")
        
        # This single call processes ALL states at once
        actions, log_probs = ray.get(
            self.model_server.generate_batch.remote(
                states,
                max_new_tokens=512,
                temperature=0.0  # Greedy decoding
            )
        )
        print(f"  ✓ Generated {len(actions)} actions")
        
        # Get value estimates (also batched)
        values = ray.get(
            self.model_server.estimate_values_batch.remote(states)
        )
        print(f"  ✓ Estimated {len(values)} values")
        
        # ------------------------------------------------------------------------
        # PHASE 3: Distributed execution (16 CPU workers in parallel)
        # ------------------------------------------------------------------------
        print("Phase 3: Executing code in parallel...")
        
        # Distribute execution across workers
        trajectory_futures = []
        for i, (task, state, action, log_prob, value) in enumerate(
            zip(tasks, states, actions, log_probs, values)
        ):
            # Round-robin assignment to workers
            worker = self.collectors[i % self.num_workers]
            
            # Execute asynchronously
            future = worker.execute_and_build_trajectory.remote(
                task, state, action, log_prob, value
            )
            trajectory_futures.append(future)
        
        # Wait for all executions to complete
        trajectories = ray.get(trajectory_futures)
        print(f"  ✓ Executed {len(trajectories)} trajectories")
        
        print(f"{'='*70}\n")
        
        return trajectories
    
    # ------------------------------------------------------------------------
    # TRAINING STEP
    # ------------------------------------------------------------------------
    
    def train_step(self, tasks: List[Task]) -> TrainingMetrics:
        """
        Perform one training step.
        
        Steps:
        1. Collect trajectories (with batched inference!)
        2. Compute advantages and returns
        3. Update policy and value networks
        4. Return metrics
        """
        print(f"\n[Trainer] train_step called with {len(tasks)} tasks")
        
        # Collect trajectories
        print(f"[Trainer] Collecting trajectories...")
        trajectories = self.collect_trajectories_parallel(tasks)
        
        # Filter out empty trajectories
        trajectories = [t for t in trajectories if not t.is_empty()]
        print(f"[Trainer] Collected {len(trajectories)} non-empty trajectories")
        
        if not trajectories:
            print(f"[Trainer] No valid trajectories, returning empty metrics")
            return self._empty_metrics()
        
        # Compute advantages and returns
        print(f"[Trainer] Computing advantages and returns...")
        for trajectory in trajectories:
            advantages, returns = self.advantage_estimator.compute_advantages_and_returns(
                trajectory,
                next_value=0.0
            )
            trajectory.advantages = advantages
            trajectory.returns = returns
        
        # Prepare data for update
        print(f"[Trainer] Preparing data for PPO update...")
        states, actions, old_log_probs, advantages, returns = self._prepare_update_data(trajectories)
        
        # Update networks on model server
        print(f"[Trainer] Updating policy and value networks on model server...")
        update_metrics = ray.get(
            self.model_server.update_policy_value.remote(
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
                self.ppo_config
            )
        )
        
        # Compute overall metrics
        metrics = self._compute_metrics(
            trajectories,
            update_metrics['policy_loss'],
            update_metrics['value_loss'],
            update_metrics['kl_divergence']
        )
        
        self.step += 1
        return metrics
    
    # ------------------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------------------
    
    def _prepare_update_data(self, trajectories: List[Trajectory]):
        """Prepare data for PPO update."""
        states = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []
        
        for traj in trajectories:
            states.extend(traj.states)
            actions.extend(traj.actions)
            old_log_probs.extend(traj.log_probs)
            
            if traj.advantages is not None:
                advantages.extend(traj.advantages)
            
            if traj.returns is not None:
                returns.extend(traj.returns)
        
        # Convert to tensors
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
    
    def _empty_metrics(self) -> TrainingMetrics:
        """Return empty metrics."""
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
    
    # ------------------------------------------------------------------------
    # CHECKPOINTING
    # ------------------------------------------------------------------------
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        state_dict = ray.get(self.model_server.get_state_dict.remote())
        torch.save(state_dict, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        state_dict = torch.load(path)
        ray.get(self.model_server.load_state_dict.remote(state_dict))


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimized distributed training with batched inference"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="codellama/CodeLlama-34b-Python-hf",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of CPU workers for code execution"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization method"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Total training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for trajectory collection"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="PPO epochs per update"
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        default="google-code-golf-2025",
        help="Directory containing tasks"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_distributed",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs_distributed",
        help="Logging directory"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="batched_distributed",
        help="Experiment name"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("OPTIMIZED DISTRIBUTED LLM-RL TRAINING")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Architecture: 1 model server (8 GPUs, tensor parallel)")
    print(f"Workers: {args.num_workers} CPU workers")
    print(f"Batch size: {args.batch_size}")
    print(f"Key optimization: BATCHED INFERENCE")
    print("="*70 + "\n")
    
    # ------------------------------------------------------------------------
    # Load and prepare tasks
    # ------------------------------------------------------------------------
    print("Loading tasks...")
    loader = TaskLoader(task_dir=args.task_dir)
    all_tasks = loader.load_all_tasks()
    print(f"✓ Loaded {len(all_tasks)} tasks")
    
    # Analyze difficulty
    print("\nAnalyzing task difficulty...")
    analyzer = DifficultyAnalyzer()
    for task in all_tasks:
        analyzer.compute_complexity_score(task)
    
    categories = analyzer.categorize_by_difficulty(all_tasks)
    print(f"✓ Difficulty categories:")
    print(f"  Easy: {len(categories['easy'])}")
    print(f"  Medium: {len(categories['medium'])}")
    print(f"  Hard: {len(categories['hard'])}")
    
    # Train/val split
    train_tasks = all_tasks[:int(0.8 * len(all_tasks))]
    val_tasks = all_tasks[int(0.8 * len(all_tasks)):]
    print(f"\n✓ Split: {len(train_tasks)} train, {len(val_tasks)} val")
    
    # ------------------------------------------------------------------------
    # Initialize configs
    # ------------------------------------------------------------------------
    model_config = ModelConfig(
        model_name=args.model_name,
        device="cuda",  # Will auto-shard across GPUs
        quantization=args.quantization if args.quantization != "none" else None,
        temperature=0.7
    )
    
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
    
    # ------------------------------------------------------------------------
    # Initialize trainer
    # ------------------------------------------------------------------------
    trainer = DistributedPPOTrainer(
        model_config=model_config,
        ppo_config=ppo_config,
        num_workers=args.num_workers
    )
    
    # ------------------------------------------------------------------------
    # Initialize logging and checkpointing
    # ------------------------------------------------------------------------
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=5
    )
    metrics_tracker = MetricsTracker(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    early_stop = EarlyStopping(patience=20, mode="max")
    
    # ------------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    best_val_score = 0.0
    import random
    
    for step in range(args.num_steps):
        # Sample batch of tasks
        batch_tasks = random.sample(
            train_tasks,
            min(args.batch_size, len(train_tasks))
        )
        
        # Training step (with batched inference!)
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
            print(f"Step {step:4d}/{args.num_steps} | "
                  f"Reward: {metrics.average_reward:6.3f} | "
                  f"Success: {metrics.success_rate:5.1%} | "
                  f"PL: {metrics.policy_loss:6.3f} | "
                  f"VL: {metrics.value_loss:6.3f} | "
                  f"KL: {metrics.kl_divergence:6.4f}")
        
        # Evaluate
        if step % args.eval_every == 0 and step > 0:
            print(f"\n{'='*70}")
            print(f"EVALUATION AT STEP {step}")
            print(f"{'='*70}")
            
            # TODO: Add proper evaluation
            # For now, use training metrics as proxy
            val_score = metrics.success_rate
            
            print(f"Validation success rate: {val_score:.2%}")
            metrics_tracker.log_evaluation(step, {"success_rate": val_score})
            
            # Check if best
            is_best = val_score > best_val_score
            if is_best:
                best_val_score = val_score
                print(f"✓ New best model! Success rate: {val_score:.2%}")
            
            # Early stopping check
            if early_stop(val_score):
                print(f"\nEarly stopping triggered at step {step}")
                break
            
            print(f"{'='*70}\n")
        
        # Save checkpoint
        if step % args.save_every == 0 and step > 0:
            print(f"\nSaving checkpoint at step {step}...")
            ckpt_path = Path(args.checkpoint_dir) / f"checkpoint_step_{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(ckpt_path))
            print(f"✓ Checkpoint saved: {ckpt_path}")
    
    # ------------------------------------------------------------------------
    # Training complete
    # ------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best validation score: {best_val_score:.2%}")
    print(f"Total steps: {trainer.step}")
    print(f"{'='*70}\n")
    
    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()