#!/usr/bin/env python
"""
Main training script for LLM-RL Code Golf system.
"""

import argparse
from pathlib import Path
from src.models import ModelConfig, PolicyNetwork, ValueNetwork
from src.data import TaskLoader, DifficultyAnalyzer
from src.training import (
    PPOTrainer,
    PPOConfig,
    RewardCalculator,
    CodeSandbox
)
from src.experiments import CheckpointManager, MetricsTracker, EarlyStopping
from src.evaluation import InferencePipeline, PerformanceEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LLM with RL for code golf tasks"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name"
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
        help="Model quantization"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="PPO epochs per update"
    )
    
    # Data arguments
    parser.add_argument(
        "--task-dir",
        type=str,
        default="google-code-golf-2025",
        help="Directory with task JSON files"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to use"
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
        default=100,
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
        default="llm_rl_codegolf",
        help="Experiment name"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N steps"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*60)
    print("LLM-RL Code Golf Training")
    print("="*60)
    
    # Load tasks
    print(f"\nLoading tasks from {args.task_dir}...")
    loader = TaskLoader(task_dir=args.task_dir)
    all_tasks = loader.load_all_tasks()
    
    if args.max_tasks:
        all_tasks = all_tasks[:args.max_tasks]
    
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
    
    # Initialize model
    print(f"\nInitializing model: {args.model_name}...")
    model_config = ModelConfig(
        model_name=args.model_name,
        device=args.device,
        quantization=args.quantization,
        temperature=0.7
    )
    
    policy = PolicyNetwork(model_config)
    value = ValueNetwork(policy, hidden_size=768)
    print(f"✓ Model loaded: {policy.get_total_parameters():,} parameters")
    
    # Initialize training components
    print("\nInitializing training components...")
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
    
    reward_calc = RewardCalculator()
    sandbox = CodeSandbox(timeout=5.0)
    
    trainer = PPOTrainer(policy, value, ppo_config, reward_calc, sandbox)
    print("✓ PPO trainer initialized")
    
    # Initialize checkpointing and logging
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=5
    )
    metrics_tracker = MetricsTracker(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    early_stop = EarlyStopping(patience=20, mode="max")
    print("✓ Checkpointing and logging initialized")
    
    # Resume from checkpoint if specified
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
    
    # Initialize inference for evaluation
    inference = InferencePipeline(policy, sandbox)
    evaluator = PerformanceEvaluator(inference)
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    best_val_score = 0.0
    
    for step in range(start_step, args.num_steps):
        # Sample tasks for this step
        import random
        batch_tasks = random.sample(train_tasks, min(args.batch_size, len(train_tasks)))
        
        # Training step
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
            eval_metrics = evaluator.evaluate(
                val_tasks[:10],  # Evaluate on subset
                num_samples=1,
                strategy="greedy"
            )
            
            print(f"  Val Success Rate: {eval_metrics['success_rate']:.2%}")
            print(f"  Val Pass Rate: {eval_metrics['average_pass_rate']:.2%}")
            
            metrics_tracker.log_evaluation(step, eval_metrics)
            
            # Check for best model
            val_score = eval_metrics['success_rate']
            is_best = val_score > best_val_score
            if is_best:
                best_val_score = val_score
                print(f"  ✓ New best model! Success rate: {val_score:.2%}")
            
            # Early stopping
            if early_stop(val_score):
                print(f"\nEarly stopping triggered at step {step}")
                break
        
        # Save checkpoint
        if step % args.save_every == 0 and step > 0:
            print(f"\nSaving checkpoint at step {step}...")
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
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}\n")
    
    final_metrics = evaluator.evaluate(val_tasks, num_samples=1, strategy="greedy")
    print(f"Final Success Rate: {final_metrics['success_rate']:.2%}")
    print(f"Final Pass Rate: {final_metrics['average_pass_rate']:.2%}")
    print(f"Solved Tasks: {final_metrics['solved_tasks']}/{final_metrics['total_tasks']}")
    
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
    
    # Print summary
    metrics_tracker.print_summary()
    
    # Export results
    results_path = Path(args.log_dir) / f"{args.experiment_name}_results.json"
    metrics_tracker.export_results(str(results_path))
    print(f"✓ Results exported to {results_path}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
