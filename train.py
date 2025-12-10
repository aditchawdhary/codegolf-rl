#!/usr/bin/env python
"""
Enhanced Training Script for LLM-RL Code Golf
Includes ALL features from discussion:
1. Uses ALL examples (train + test + arc_gen = 265 per task)
2. Full data augmentation (rotations + flips)
3. PPO training with proper reward calculation
4. Memory-efficient batching for L40S GPU
"""

import argparse
import random
from pathlib import Path
from typing import List
import torch

from src.models import ModelConfig, PolicyNetwork, ValueNetwork
from src.data import TaskLoader, DifficultyAnalyzer, Task, ProblemAugmenter
from src.training import (
    PPOTrainer,
    PPOConfig,
    RewardCalculator,
    CodeSandbox,
    AugmentedPPOTrainer,
    Trajectory
)
from src.experiments import CheckpointManager, MetricsTracker, EarlyStopping
from src.evaluation import InferencePipeline, PerformanceEvaluator


class EnhancedAugmentedPPOTrainer(AugmentedPPOTrainer):
    """
    Enhanced PPO trainer that uses ALL examples from each task.
    
    Key improvements:
    1. Collects trajectories from ALL 265 examples per augmented task
       (3 train + 1 test + 261 arc_gen)
    2. Memory-efficient batching to fit in L40S 48GB
    3. Proper handling of large example sets
    """
    
    def __init__(self, *args, use_all_examples: bool = True, 
                 examples_batch_size: int = 32, **kwargs):
        """
        Initialize enhanced trainer.
        
        Args:
            use_all_examples: If True, use all 265 examples per task
            examples_batch_size: Batch size for processing examples (memory management)
            *args, **kwargs: Passed to AugmentedPPOTrainer
        """
        super().__init__(*args, **kwargs)
        self.use_all_examples = use_all_examples
        self.examples_batch_size = examples_batch_size
        
        print(f"[EnhancedTrainer] Initialized")
        print(f"  Use all examples: {use_all_examples}")
        print(f"  Examples batch size: {examples_batch_size}")
    
    def collect_trajectories(self, tasks: List[Task]) -> List[Trajectory]:
        """
        Collect trajectories using ALL examples from each task.
        
        For each task:
        - 3 training examples
        - 1 test example  
        - 261 arc_gen examples
        = 265 total examples
        
        We generate code for EACH example individually.
        """
        if not self.use_all_examples:
            # Fall back to parent's implementation (only training examples)
            return super().collect_trajectories(tasks)
        
        print(f"\n[EnhancedTrainer] Collecting trajectories with ALL examples")
        print(f"  Tasks: {len(tasks)}")
        
        all_trajectories = []
        
        # Process tasks in batches to manage memory
        for task_idx, task in enumerate(tasks):
            print(f"  Processing task {task_idx + 1}/{len(tasks)} "
                  f"(ID: {task.task_id})...")
            
            # Get ALL examples for this task
            all_examples = (
                task.train_examples + 
                task.test_examples + 
                task.arc_gen_examples
            )
            
            print(f"    Total examples: {len(all_examples)} "
                  f"(train: {len(task.train_examples)}, "
                  f"test: {len(task.test_examples)}, "
                  f"arc_gen: {len(task.arc_gen_examples)})")
            
            # Process examples in batches to avoid OOM
            for batch_start in range(0, len(all_examples), self.examples_batch_size):
                batch_end = min(batch_start + self.examples_batch_size, len(all_examples))
                example_batch = all_examples[batch_start:batch_end]
                
                print(f"    Example batch {batch_start}-{batch_end}...")
                
                # Collect trajectory for each example
                for example_idx, example in enumerate(example_batch):
                    trajectory = Trajectory()
                    
                    # Format the example as a prompt
                    state = self._format_example_prompt(example, task)
                    
                    # Sample action from policy
                    try:
                        action, log_prob = self.policy.sample_action(
                            state,
                            max_new_tokens=512
                        )
                        
                        # Estimate value
                        value = self.value.estimate_value(state)
                        
                        # Create a mini-task with just this example for execution
                        mini_task = Task(
                            task_id=task.task_id,
                            train_examples=[example],
                            test_examples=[],
                            arc_gen_examples=[]
                        )
                        
                        # Execute code and get reward
                        results = self.sandbox.execute(action, mini_task)
                        reward = self.reward_calculator.compute_reward(
                            action,
                            results,
                            task_difficulty=task.difficulty_score / 100.0
                        )
                        
                        # Add to trajectory
                        trajectory.add_step(state, action, reward, log_prob, value)
                        all_trajectories.append(trajectory)
                        
                    except Exception as e:
                        print(f"      Warning: Failed to process example: {e}")
                        continue
                
                # Memory cleanup after each batch
                if (batch_end % 50 == 0) or (batch_end == len(all_examples)):
                    torch.cuda.empty_cache()
        
        print(f"  ✓ Collected {len(all_trajectories)} trajectories")
        return all_trajectories
    
    def _format_example_prompt(self, example, task: Task) -> str:
        """
        Format a single example as a prompt for the model.
        
        Uses the compact format from our discussion:
        # Example: [[input]] -> [[output]]
        """
        from src.data import TaskFormatter
        
        # Create a temporary task with just this example
        temp_task = Task(
            task_id=task.task_id,
            train_examples=[example],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        formatter = TaskFormatter(include_instructions=True)
        return formatter.format_prompt(temp_task, include_test=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced LLM-RL training with full data augmentation"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name (default: Qwen3-0.6B for L40S)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["4bit", "8bit", None],
        help="Model quantization (None recommended for 0.6B model)"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of tasks per training step"
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
    
    # Data augmentation arguments (NEW!)
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=4,
        help="Number of augmentations per task (rotations + flips)"
    )
    parser.add_argument(
        "--use-all-examples",
        action="store_true",
        default=True,
        help="Use ALL examples (train + test + arc_gen = 265 per task)"
    )
    parser.add_argument(
        "--examples-batch-size",
        type=int,
        default=32,
        help="Batch size for processing examples (memory management)"
    )
    
    # Data arguments
    parser.add_argument(
        "--task-dir",
        type=str,
        default="google-code-golf-2025",
        help="Directory with task JSON files"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=20,
        help="Number of tasks to use (default: 20 for testing)"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="enhanced_rl",
        help="Experiment name"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Evaluate every N steps"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*70)
    print("ENHANCED LLM-RL CODE GOLF TRAINING")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Quantization: {args.quantization}")
    print(f"Augmentations: {args.num_augmentations}x per task")
    print(f"Use all examples: {args.use_all_examples}")
    if args.use_all_examples:
        print(f"  → 265 examples per task (train + test + arc_gen)")
        print(f"  → {args.batch_size} tasks × {args.num_augmentations} aug × 265 examples")
        print(f"  → = {args.batch_size * args.num_augmentations * 265:,} total examples per step!")
    print("="*70 + "\n")
    
    # ------------------------------------------------------------------------
    # Load tasks
    # ------------------------------------------------------------------------
    print(f"Loading tasks from {args.task_dir}...")
    loader = TaskLoader(task_dir=args.task_dir)
    all_tasks = loader.load_all_tasks()
    
    # Limit to specified number of tasks
    if args.num_tasks:
        all_tasks = all_tasks[:args.num_tasks]
    
    print(f"✓ Loaded {len(all_tasks)} tasks")
    
    # Show example count breakdown
    if all_tasks:
        sample_task = all_tasks[0]
        print(f"\nExample task structure (Task {sample_task.task_id}):")
        print(f"  Train examples: {len(sample_task.train_examples)}")
        print(f"  Test examples: {len(sample_task.test_examples)}")
        print(f"  Arc-gen examples: {len(sample_task.arc_gen_examples)}")
        print(f"  Total: {sample_task.total_examples}")
    
    # ------------------------------------------------------------------------
    # Analyze difficulty
    # ------------------------------------------------------------------------
    print("\nAnalyzing task difficulty...")
    analyzer = DifficultyAnalyzer()
    for task in all_tasks:
        analyzer.compute_complexity_score(task)
    
    categories = analyzer.categorize_by_difficulty(all_tasks)
    print(f"✓ Difficulty distribution:")
    print(f"  Easy: {len(categories['easy'])} tasks")
    print(f"  Medium: {len(categories['medium'])} tasks")
    print(f"  Hard: {len(categories['hard'])} tasks")
    
    # ------------------------------------------------------------------------
    # Train/val split
    # ------------------------------------------------------------------------
    train_split = 0.8
    train_size = int(train_split * len(all_tasks))
    train_tasks = all_tasks[:train_size]
    val_tasks = all_tasks[train_size:]
    print(f"\n✓ Split: {len(train_tasks)} train, {len(val_tasks)} val")
    
    # ------------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------------
    print(f"\nInitializing model: {args.model_name}...")
    model_config = ModelConfig(
        model_name=args.model_name,
        device=args.device,
        quantization=args.quantization,
        temperature=0.7,
        torch_dtype="bfloat16"  # Better numerical stability
    )
    
    print("  Loading policy network...")
    policy = PolicyNetwork(model_config)
    
    print("  Creating value network...")
    value = ValueNetwork(policy, hidden_size=768)
    
    total_params = policy.get_total_parameters()
    trainable_params = policy.get_trainable_parameters()
    print(f"✓ Model loaded:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Estimate memory usage
    param_memory = total_params * 2 / (1024**3)  # 2 bytes per param (bfloat16)
    print(f"  Estimated memory: ~{param_memory:.1f} GB")
    
    # ------------------------------------------------------------------------
    # Initialize training components
    # ------------------------------------------------------------------------
    print("\nInitializing training components...")
    
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gamma=0.99,
        lambda_=0.95,
        clip_epsilon=0.2,
        max_grad_norm=1.0
    )
    
    reward_calc = RewardCalculator(
        max_reward=10.0,
        code_length_weight=0.01  # Small penalty for longer code
    )
    
    sandbox = CodeSandbox(timeout=5.0)
    
    # Use enhanced trainer
    trainer = EnhancedAugmentedPPOTrainer(
        policy, 
        value, 
        ppo_config, 
        reward_calc, 
        sandbox,
        num_augmentations=args.num_augmentations,
        use_all_examples=args.use_all_examples,
        examples_batch_size=args.examples_batch_size
    )
    
    print(f"✓ Enhanced PPO trainer initialized:")
    print(f"  Augmentations: {args.num_augmentations}x")
    print(f"  Use all examples: {args.use_all_examples}")
    print(f"  Examples batch size: {args.examples_batch_size}")
    
    # ------------------------------------------------------------------------
    # Initialize checkpointing and logging
    # ------------------------------------------------------------------------
    print("\nInitializing experiment tracking...")
    
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=5
    )
    
    metrics_tracker = MetricsTracker(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    early_stop = EarlyStopping(patience=20, mode="max")
    
    print(f"✓ Checkpoints: {args.checkpoint_dir}")
    print(f"✓ Logs: {args.log_dir}")
    print(f"✓ Experiment: {args.experiment_name}")
    
    # ------------------------------------------------------------------------
    # Resume from checkpoint if specified
    # ------------------------------------------------------------------------
    start_step = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        state = ckpt_manager.load_checkpoint(
            args.resume_from,
            policy,
            value,
            trainer.policy_optimizer,
            trainer.value_optimizer
        )
        start_step = state["step"]
        print(f"✓ Resumed from step {start_step}")
    
    # ------------------------------------------------------------------------
    # Initialize inference for evaluation
    # ------------------------------------------------------------------------
    print("\nInitializing evaluation...")
    inference = InferencePipeline(policy, sandbox)
    evaluator = PerformanceEvaluator(inference)
    print("✓ Evaluation pipeline ready")
    
    # ------------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Steps: {start_step} → {args.num_steps}")
    print(f"Batch size: {args.batch_size} tasks")
    print(f"Augmentations: {args.num_augmentations}x")
    print(f"Examples per task: {'ALL (265)' if args.use_all_examples else 'train only'}")
    print(f"{'='*70}\n")
    
    best_val_score = 0.0
    
    for step in range(start_step, args.num_steps):
        print(f"\n{'='*70}")
        print(f"STEP {step + 1}/{args.num_steps}")
        print(f"{'='*70}")
        
        # Sample tasks for this step
        batch_tasks = random.sample(
            train_tasks, 
            min(args.batch_size, len(train_tasks))
        )
        
        print(f"Sampled {len(batch_tasks)} tasks:")
        for i, task in enumerate(batch_tasks):
            print(f"  {i+1}. Task {task.task_id} "
                  f"(difficulty: {task.difficulty_score:.1f})")
        
        # Training step (with augmentation and all examples!)
        print("\nExecuting training step...")
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
        print(f"\nStep {step + 1}/{args.num_steps} Results:")
        print(f"  Average Reward: {metrics.average_reward:7.3f}")
        print(f"  Success Rate:   {metrics.success_rate:6.2%}")
        print(f"  Policy Loss:    {metrics.policy_loss:7.3f}")
        print(f"  Value Loss:     {metrics.value_loss:7.3f}")
        print(f"  KL Divergence:  {metrics.kl_divergence:7.4f}")
        
        # ------------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------------
        if step % args.eval_every == 0 and step > 0:
            print(f"\n{'='*70}")
            print(f"EVALUATION AT STEP {step + 1}")
            print(f"{'='*70}")
            
            # Evaluate on subset of validation tasks
            eval_tasks = val_tasks[:min(5, len(val_tasks))]
            print(f"Evaluating on {len(eval_tasks)} validation tasks...")
            
            eval_metrics = evaluator.evaluate(
                eval_tasks,
                num_samples=1,
                strategy="greedy"
            )
            
            print(f"\nValidation Results:")
            print(f"  Success Rate:    {eval_metrics['success_rate']:6.2%}")
            print(f"  Pass Rate:       {eval_metrics['average_pass_rate']:6.2%}")
            print(f"  Solved Tasks:    {eval_metrics['solved_tasks']}/{eval_metrics['total_tasks']}")
            print(f"  Avg Code Length: {eval_metrics.get('average_code_length', 0):.0f} chars")
            
            metrics_tracker.log_evaluation(step, eval_metrics)
            
            # Check for best model
            val_score = eval_metrics['success_rate']
            is_best = val_score > best_val_score
            
            if is_best:
                best_val_score = val_score
                print(f"\n  ✓ NEW BEST MODEL!")
                print(f"    Success rate: {val_score:.2%}")
            
            # Early stopping check
            if early_stop(val_score):
                print(f"\n⚠ Early stopping triggered at step {step + 1}")
                print(f"  Validation score hasn't improved for {early_stop.patience} evaluations")
                break
            
            print(f"{'='*70}")
        
        # ------------------------------------------------------------------------
        # Save checkpoint
        # ------------------------------------------------------------------------
        if step % args.save_every == 0 and step > 0:
            print(f"\nSaving checkpoint at step {step + 1}...")
            ckpt_path = ckpt_manager.save_checkpoint(
                policy,
                value,
                trainer.policy_optimizer,
                trainer.value_optimizer,
                step,
                metrics_dict,
                ppo_config.__dict__,
                is_best=False
            )
            print(f"✓ Checkpoint saved: {ckpt_path}")
    
    # ------------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    print("Evaluating on full validation set...")
    final_metrics = evaluator.evaluate(
        val_tasks, 
        num_samples=1, 
        strategy="greedy"
    )
    
    print(f"\nFinal Results:")
    print(f"  Success Rate:    {final_metrics['success_rate']:6.2%}")
    print(f"  Pass Rate:       {final_metrics['average_pass_rate']:6.2%}")
    print(f"  Solved Tasks:    {final_metrics['solved_tasks']}/{final_metrics['total_tasks']}")
    print(f"  Avg Code Length: {final_metrics.get('average_code_length', 0):.0f} chars")
    
    # Save final checkpoint
    print("\nSaving final checkpoint...")
    ckpt_manager.save_checkpoint(
        policy,
        value,
        trainer.policy_optimizer,
        trainer.value_optimizer,
        args.num_steps,
        final_metrics,
        ppo_config.__dict__,
        is_best=final_metrics['success_rate'] > best_val_score
    )
    
    # ------------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    
    metrics_tracker.print_summary()
    
    print(f"\nBest Validation Score: {best_val_score:.2%}")
    print(f"Final Validation Score: {final_metrics['success_rate']:.2%}")
    print(f"Total Training Steps: {args.num_steps}")
    
    if args.use_all_examples:
        total_examples = args.num_steps * args.batch_size * args.num_augmentations * 265
        print(f"Total Examples Processed: {total_examples:,}")
    
    # Export results
    results_path = Path(args.log_dir) / f"{args.experiment_name}_results.json"
    metrics_tracker.export_results(str(results_path))
    print(f"\n✓ Results exported to {results_path}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
