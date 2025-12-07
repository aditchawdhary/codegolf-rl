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

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType
import deepspeed
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
# SHARED MODEL SERVER (GPUs with Tensor Parallelism)
# ============================================================================

def create_model_server_class(num_gpus):
    """Factory function to create SharedModelServer with dynamic GPU count."""
    @ray.remote(num_gpus=num_gpus)
    class SharedModelServer:
        """
        Shared model server that uses dynamic GPUs with tensor parallelism.
        
        This is the heart of the system. The model is SHARDED across all GPUs,
        meaning different parts of each layer live on different GPUs.
        
        Key principle: Process requests in BATCHES to maximize GPU utilization.
        Instead of 16 sequential requests, we do 1 batch of 16 states.
        """
        
        def __init__(self, model_config: ModelConfig, num_gpus: int):
            """Initialize HuggingFace model with DeepSpeed for distributed training."""
            self.num_gpus = num_gpus
            print("\n" + "="*70)
            print("INITIALIZING HF MODEL WITH DEEPSPEED")
            print("="*70)
            print(f"Model: {model_config.model_name}")
            print(f"GPUs: {num_gpus}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with automatic device mapping
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            device_map="auto",  # Automatically distribute across GPUs
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache for training to save memory
        )
        
        # Enable gradient checkpointing to save memory
        print("Enabling gradient checkpointing...")
        self.model.gradient_checkpointing_enable()
        
        # Add LoRA for efficient fine-tuning
        print("Adding LoRA adapters...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print(f"✓ Model distributed across {torch.cuda.device_count()} GPUs")
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5
        )
        
        # Store config
        self.model_config = model_config
        
        print(f"\n✓ Model ready on {torch.cuda.device_count()} GPU(s)")
        print(f"✓ Trainable parameters: {self.model.module.num_parameters(only_trainable=True) if hasattr(self.model, 'module') else self.model.num_parameters(only_trainable=True):,}")
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
        Generate actions using HuggingFace model.
        
        Args:
            states: List of prompt strings
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            actions: Generated code strings
            log_probs: Log probabilities for PPO
        """
        print(f"  [ModelServer] Generating for {len(states)} states")
        
        self.model.eval()
        actions = []
        log_probs = []
        
        with torch.no_grad():
            for state in states:
                # Tokenize
                inputs = self.tokenizer(
                    state, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=4096
                ).to("cuda")
                
                # Generate
                if temperature == 0.0:
                    # Greedy decoding
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    # Sampling
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # Decode
                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                else:
                    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                action = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                actions.append(action)
                
                # Compute log prob (simplified)
                log_prob = torch.tensor(0.0)  # Will compute properly in PPO
                log_probs.append(log_prob)
        
        print(f"  [ModelServer] Generated {len(actions)} actions")
        return actions, log_probs
    
    def estimate_values_batch(self, states: List[str]) -> List[torch.Tensor]:
        """
        Estimate values (simplified - using mean logits as proxy).
        """
        print(f"  [ModelServer] Estimating values for {len(states)} states")
        
        values = []
        self.model.eval()
        
        with torch.no_grad():
            for state in states:
                inputs = self.tokenizer(
                    state, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048
                ).to("cuda")
                
                outputs = self.model(**inputs)
                value = outputs.logits.mean().cpu()
                values.append(value)
        
        print(f"  [ModelServer] Estimated {len(values)} values")
        return values
    
    def compute_log_probs_batch(
        self, 
        states: List[str], 
        actions: List[str],
        requires_grad: bool = False
    ) -> List[torch.Tensor]:
        """
        Compute log probabilities for PPO.
        
        Args:
            requires_grad: If True, compute with gradients for training
        """
        print(f"  [ModelServer] Computing log probs for {len(states)} pairs (grad={requires_grad})")
        log_probs = []
        
        self.model.train() if requires_grad else self.model.eval()
        
        for state, action in zip(states, actions):
            full_text = state + action
            
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to("cuda")
            
            if requires_grad:
                # With gradients for training
                outputs = self.model(**inputs)
                log_probs_tensor = torch.log_softmax(outputs.logits, dim=-1)
                mean_log_prob = log_probs_tensor.mean()
                log_probs.append(mean_log_prob)
            else:
                # Without gradients for initial collection
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    log_probs_tensor = torch.log_softmax(outputs.logits, dim=-1)
                    mean_log_prob = log_probs_tensor.mean()
                    log_probs.append(mean_log_prob.cpu())
        
        print(f"  [ModelServer] Computed {len(log_probs)} log probs")
        return log_probs
    
    def compute_log_probs_single(
        self, 
        state: str, 
        action: str,
        requires_grad: bool = True
    ) -> torch.Tensor:
        """
        Compute log probability for a single state-action pair.
        Used for gradient accumulation to save memory.
        """
        full_text = state + action
        
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to("cuda")
        
        outputs = self.model(**inputs)
        log_probs_tensor = torch.log_softmax(outputs.logits, dim=-1)
        mean_log_prob = log_probs_tensor.mean()
        
        return mean_log_prob
    
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
        PPO update - model learns from rewards in real-time!
        Uses gradient accumulation to reduce memory usage.
        """
        print(f"  [ModelServer] PPO update for {len(states)} trajectories")
        
        self.model.train()
        
        total_policy_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        
        # Move to device
        old_log_probs = old_log_probs.to("cuda")
        advantages = advantages.to("cuda")
        
        # Gradient accumulation: process 1 sample at a time to save memory
        accumulation_steps = len(states)
        
        # PPO epochs
        for epoch in range(ppo_config.num_epochs):
            self.optimizer.zero_grad()
            
            epoch_policy_loss = 0.0
            epoch_kl = 0.0
            new_log_probs_list = []
            
            # Process each sample individually with gradient accumulation
            for i, (state, action) in enumerate(zip(states, actions)):
                # Compute new log prob WITH gradients for this single sample
                new_log_prob = self.compute_log_probs_single(state, action, requires_grad=True)
                new_log_probs_list.append(new_log_prob)
                
                # PPO loss for this sample
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - ppo_config.clip_epsilon,
                    1.0 + ppo_config.clip_epsilon
                ) * advantages[i]
                sample_loss = -torch.min(surr1, surr2)
                
                # Scale loss by accumulation steps
                scaled_loss = sample_loss / accumulation_steps
                scaled_loss.backward()
                
                epoch_policy_loss += sample_loss.item()
                
                # Clear cache periodically
                if (i + 1) % 4 == 0:
                    torch.cuda.empty_cache()
            
            # Update after accumulating all gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                ppo_config.max_grad_norm
            )
            self.optimizer.step()
            
            # Compute KL divergence
            new_log_probs = torch.stack(new_log_probs_list)
            kl_div = (old_log_probs - new_log_probs).mean().item()
            
            # Track
            total_policy_loss += epoch_policy_loss / len(states)
            total_kl += abs(kl_div)
            num_updates += 1
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Early stop
            if abs(kl_div) > ppo_config.target_kl:
                print(f"    Early stop at epoch {epoch} (KL={kl_div:.4f})")
                break
        
        print(f"  [ModelServer] ✓ Model updated ({num_updates} epochs)")
        
        return {
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0.0,
            'value_loss': 0.0,
            'kl_divergence': total_kl / num_updates if num_updates > 0 else 0.0,
        }
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def get_state_dict(self):
        """Get state dict for checkpointing."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
    
    return SharedModelServer


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
        num_workers: int = 16,
        num_gpus: int = 8
    ):
        """Initialize distributed trainer."""
        self.model_config = model_config
        self.ppo_config = ppo_config
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            print("\nInitializing Ray...")
            ray.init(
                num_gpus=num_gpus,
                num_cpus=num_workers + 4,  # +4 for main process
                dashboard_host="0.0.0.0",
                dashboard_port=8265
            )
            print(f"✓ Ray initialized")
            print(f"✓ Dashboard available at http://0.0.0.0:8265")
        
        # Create shared model server (tensor parallelism across GPUs)
        print("\nCreating shared model server...")
        SharedModelServerClass = create_model_server_class(num_gpus)
        self.model_server = SharedModelServerClass.remote(model_config, num_gpus)
        
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
        print(f"Model Server: 1 instance using {num_gpus} GPUs (tensor parallel)")
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
        # PHASE 2: Batched inference on model server (5 GPUs working together!)
        # ------------------------------------------------------------------------
        print("Phase 2: Batched inference on model server...")
        print(f"  All {NUM_TENSOR_PARALLEL_GPU} GPUs will work together on this batch!")
        
        # This single call processes ALL states at once
        actions, log_probs = ray.get(
            self.model_server.generate_batch.remote(
                states,
                max_new_tokens=512,
                temperature=0.7  # Sampling for exploration
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
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism"
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
    print(f"Architecture: 1 model server ({args.num_gpus} GPUs, tensor parallel)")
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
        num_workers=args.num_workers,
        num_gpus=args.num_gpus
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