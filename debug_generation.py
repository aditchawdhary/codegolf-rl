"""
Debug generation to find the exact issue.
"""

import torch
import gc
from src.models.config import ModelConfig
from src.models.policy_network import PolicyNetwork


def test_single_generation():
    """Test a single generation."""
    print("Testing single generation...")
    
    config = ModelConfig(
        model_name="gpt2",
        max_length=128,
        device="cpu",
        torch_dtype="float32",
    )
    policy = PolicyNetwork(config)
    
    prompt = "def solve():"
    
    try:
        code, log_prob = policy.sample_action(
            prompt,
            temperature=1.0,
            max_new_tokens=5
        )
        print(f"✓ Single generation successful")
        print(f"  Generated: {code}")
        print(f"  Log prob: {log_prob}")
        return True
    except Exception as e:
        print(f"✗ Single generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_generations_with_cleanup():
    """Test multiple generations with explicit cleanup."""
    print("\nTesting multiple generations with cleanup...")
    
    config = ModelConfig(
        model_name="gpt2",
        max_length=128,
        device="cpu",
        torch_dtype="float32",
    )
    policy = PolicyNetwork(config)
    
    prompt = "def solve():"
    samples = []
    
    for i in range(3):
        try:
            print(f"  Generation {i+1}...")
            
            # Generate
            with torch.no_grad():
                code, log_prob = policy.sample_action(
                    prompt,
                    temperature=1.0,
                    max_new_tokens=5
                )
            
            samples.append(code)
            print(f"    ✓ Success: {code[:30]}...")
            
            # Explicit cleanup
            gc.collect()
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"✓ All generations successful")
    print(f"  Unique samples: {len(set(samples))}/{len(samples)}")
    return True


def test_generation_without_logprob():
    """Test generation without computing log probabilities."""
    print("\nTesting generation without log prob computation...")
    
    config = ModelConfig(
        model_name="gpt2",
        max_length=128,
        device="cpu",
        torch_dtype="float32",
    )
    policy = PolicyNetwork(config)
    
    prompt = "def solve():"
    samples = []
    
    for i in range(3):
        try:
            print(f"  Generation {i+1}...")
            
            # Just generate, don't compute log prob
            code = policy.generate(
                prompt,
                temperature=1.0,
                max_new_tokens=5
            )
            
            samples.append(code)
            print(f"    ✓ Success: {code[:30]}...")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"✓ All generations successful")
    print(f"  Unique samples: {len(set(samples))}/{len(samples)}")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("GENERATION DEBUG TESTS")
    print("=" * 60)
    print()
    
    # Test 1: Single generation
    success1 = test_single_generation()
    
    if not success1:
        print("\n⚠ Single generation failed - stopping tests")
        return
    
    # Test 2: Multiple generations with cleanup
    success2 = test_multiple_generations_with_cleanup()
    
    # Test 3: Generation without log prob
    success3 = test_generation_without_logprob()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Single generation: {'✓' if success1 else '✗'}")
    print(f"Multiple with cleanup: {'✓' if success2 else '✗'}")
    print(f"Without log prob: {'✓' if success3 else '✗'}")


if __name__ == "__main__":
    main()
