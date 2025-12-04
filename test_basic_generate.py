"""
Test basic generation without our wrapper.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_direct_generation():
    """Test generation directly with transformers."""
    print("Testing direct transformers generation...")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32
        )
        model.eval()
        
        print("Tokenizing input...")
        prompt = "def solve():"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        print("Decoding...")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✓ Success!")
        print(f"  Generated: {generated_text}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_direct_generation()
