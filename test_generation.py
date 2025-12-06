#!/usr/bin/env python
"""
Quick test script to verify vLLM generation works with 6 GPUs.
Just loads the model and generates one sample.
"""

from vllm import LLM, SamplingParams
import json

print("\n" + "="*70)
print("TESTING VLLM GENERATION WITH 2 GPUs (1.5B model)")
print("="*70)

# Load a task
print("\nLoading a sample task...")
with open("google-code-golf-2025/task001.json", "r") as f:
    task = json.load(f)

print(f"Task loaded with {len(task.get('train', []))} training examples")
print(f"First example input: {task['train'][0]['input']}")
print(f"First example output: {task['train'][0]['output'][:3]}...")  # Show first 3 rows

# Create prompt
prompt = """You are an expert Python programmer. Write a function that transforms a 3x3 grid into a 9x9 grid.

Given input and output examples, write a Python function to solve this pattern:

Example:
Input: [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
Output: [[0, 0, 0, 0, 7, 7, 0, 7, 7], ...]

Write a concise Python function:

```python
"""

print("\n" + "="*70)
print("INITIALIZING VLLM WITH 2 GPUs (1.5B model for testing)")
print("="*70)

# Initialize vLLM with tensor parallelism
# Using smaller 1.5B model for testing (fits in disk space)
# For production, you'll need more disk space for 32B model
llm = LLM(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs for testing
    gpu_memory_utilization=0.85,
    max_model_len=2048,
    trust_remote_code=True,
    dtype="half"
)

print("✓ Model loaded and sharded across 2 GPUs")

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    top_p=0.95,
    stop=["```"]
)

print("\n" + "="*70)
print("GENERATING CODE")
print("="*70)

# Generate
outputs = llm.generate([prompt], sampling_params)

# Extract result
generated_code = outputs[0].outputs[0].text

print("\nGenerated Code:")
print("-" * 70)
print(generated_code)
print("-" * 70)

print("\n✓ Generation successful!")
print("="*70)
