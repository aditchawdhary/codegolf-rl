"""
Check memory availability and requirements for model tests.
"""

import psutil
import torch
from src.models.config import ModelConfig
from src.models.policy_network import PolicyNetwork


def get_memory_info():
    """Get current memory information."""
    memory = psutil.virtual_memory()
    print("=" * 60)
    print("SYSTEM MEMORY INFO")
    print("=" * 60)
    print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    print(f"Used Memory: {memory.used / (1024**3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%")
    print()
    return memory


def estimate_model_memory():
    """Estimate memory required for model."""
    print("=" * 60)
    print("MODEL MEMORY ESTIMATION")
    print("=" * 60)
    
    # Create a small config to estimate
    config = ModelConfig(
        model_name="gpt2",
        max_length=128,
        device="cpu",
        torch_dtype="float32",
    )
    
    print(f"Loading model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Dtype: {config.torch_dtype}")
    print()
    
    # Get memory before loading
    memory_before = psutil.Process().memory_info().rss / (1024**2)
    print(f"Process memory before loading: {memory_before:.2f} MB")
    
    # Load model
    policy = PolicyNetwork(config)
    
    # Get memory after loading
    memory_after = psutil.Process().memory_info().rss / (1024**2)
    print(f"Process memory after loading: {memory_after:.2f} MB")
    print(f"Model memory usage: {memory_after - memory_before:.2f} MB")
    print()
    
    # Get model parameter count and size
    total_params = policy.get_total_parameters()
    param_size_mb = sum(
        p.element_size() * p.nelement() 
        for p in policy.model.parameters()
    ) / (1024**2)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter memory: {param_size_mb:.2f} MB")
    print()
    
    return policy, memory_after - memory_before


def test_generation_memory():
    """Test memory usage during generation."""
    print("=" * 60)
    print("GENERATION MEMORY TEST")
    print("=" * 60)
    
    config = ModelConfig(
        model_name="gpt2",
        max_length=128,
        device="cpu",
        torch_dtype="float32",
    )
    policy = PolicyNetwork(config)
    
    prompt = "def solve():"
    num_samples = 5
    
    memory_readings = []
    
    for i in range(num_samples):
        memory_before = psutil.Process().memory_info().rss / (1024**2)
        
        # Generate
        code, _ = policy.sample_action(
            prompt,
            temperature=1.0,
            max_new_tokens=10
        )
        
        memory_after = psutil.Process().memory_info().rss / (1024**2)
        memory_used = memory_after - memory_before
        memory_readings.append(memory_after)
        
        print(f"Sample {i+1}:")
        print(f"  Memory before: {memory_before:.2f} MB")
        print(f"  Memory after: {memory_after:.2f} MB")
        print(f"  Memory delta: {memory_used:.2f} MB")
        print(f"  Generated: {code[:50]}...")
        print()
        
        # Try to clean up
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Peak memory: {max(memory_readings):.2f} MB")
    print(f"Memory growth: {memory_readings[-1] - memory_readings[0]:.2f} MB")
    print()


def main():
    """Run all checks."""
    # Get system memory
    memory = get_memory_info()
    
    # Estimate model memory
    try:
        policy, model_memory = estimate_model_memory()
        
        # Test generation memory
        test_generation_memory()
        
        # Final recommendations
        print("=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        available_gb = memory.available / (1024**3)
        required_gb = model_memory * 5 / 1024  # Rough estimate for 5 generations
        
        print(f"Available memory: {available_gb:.2f} GB")
        print(f"Estimated required for test: {required_gb:.2f} GB")
        
        if available_gb > required_gb * 2:
            print("✓ Sufficient memory available for test")
        elif available_gb > required_gb:
            print("⚠ Marginal memory - test may work but could be unstable")
        else:
            print("✗ Insufficient memory - test likely to fail")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
