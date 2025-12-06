#!/usr/bin/env python
"""
Evaluate all saved checkpoints on validation tasks.
"""

import argparse
import json
from pathlib import Path
import torch
from src.models import ModelConfig, PolicyNetwork, ValueNetwork
from src.data import TaskLoader
from src.training import CodeSandbox
from src.evaluation import InferencePipeline, PerformanceEvaluator


def find_checkpoints(checkpoint_dir: str):
    """Find all checkpoint files."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []
    
    # Find all .pt files
    checkpoints = sorted(checkpoint_path.glob("*.pt"))
    return checkpoints


def evaluate_checkpoint(checkpoint_path, val_tasks, device="cuda"):
    """Evaluate a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"{'='*60}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model config from checkpoint
        model_config = ModelConfig(
            model_name=checkpoint.get("model_name", "codellama/CodeLlama-7b-Python-hf"),
            device=device,
            quantization=checkpoint.get("quantization", None)
        )
        
        # Initialize models
        policy = PolicyNetwork(model_config)
        value = ValueNetwork(policy, hidden_size=768)
        
        # Load state dicts
        policy.model.load_state_dict(checkpoint["policy_state_dict"])
        value.value_head.load_state_dict(checkpoint["value_state_dict"])
        
        # Initialize evaluation
        sandbox = CodeSandbox(timeout=5.0)
        inference = InferencePipeline(policy, sandbox)
        evaluator = PerformanceEvaluator(inference)
        
        # Evaluate on validation tasks
        print(f"Evaluating on {len(val_tasks)} validation tasks...")
        metrics = evaluator.evaluate(
            val_tasks,
            num_samples=1,
            strategy="greedy"
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  Step: {checkpoint.get('step', 'unknown')}")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Pass Rate: {metrics['average_pass_rate']:.2%}")
        print(f"  Solved Tasks: {metrics['solved_tasks']}/{metrics['total_tasks']}")
        print(f"  Avg Code Length: {metrics.get('average_code_length', 'N/A')}")
        
        return {
            "checkpoint": checkpoint_path.name,
            "step": checkpoint.get("step", None),
            "metrics": metrics
        }
        
    except Exception as e:
        print(f"Error evaluating {checkpoint_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        default="google-code-golf-2025",
        help="Directory with task JSON files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoint_evaluation_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of validation tasks to evaluate"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Checkpoint Evaluation")
    print("="*60)
    
    # Load validation tasks
    print(f"\nLoading tasks from {args.task_dir}...")
    loader = TaskLoader(task_dir=args.task_dir)
    all_tasks = loader.load_all_tasks()
    
    # Use last 20% as validation
    val_start = int(0.8 * len(all_tasks))
    val_tasks = all_tasks[val_start:]
    
    if args.max_tasks:
        val_tasks = val_tasks[:args.max_tasks]
    
    print(f"✓ Loaded {len(val_tasks)} validation tasks")
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir)
    
    if not checkpoints:
        print(f"\nNo checkpoints found in {args.checkpoint_dir}")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")
    
    # Evaluate each checkpoint
    results = []
    for checkpoint_path in checkpoints:
        result = evaluate_checkpoint(checkpoint_path, val_tasks, args.device)
        if result:
            results.append(result)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    if results:
        print("\nSummary:")
        print(f"{'Checkpoint':<30} {'Step':<10} {'Success Rate':<15} {'Solved':<10}")
        print("-" * 65)
        for result in results:
            checkpoint = result['checkpoint']
            step = result.get('step', 'N/A')
            success_rate = result['metrics']['success_rate']
            solved = f"{result['metrics']['solved_tasks']}/{result['metrics']['total_tasks']}"
            print(f"{checkpoint:<30} {str(step):<10} {success_rate:>6.2%}        {solved:<10}")
        
        # Find best checkpoint
        best_result = max(results, key=lambda x: x['metrics']['success_rate'])
        print(f"\n🏆 Best checkpoint: {best_result['checkpoint']}")
        print(f"   Success rate: {best_result['metrics']['success_rate']:.2%}")
        print(f"   Step: {best_result.get('step', 'N/A')}")


if __name__ == "__main__":
    main()
