"""
Property-based tests for task loading functionality.

Feature: llm-rl-code-golf
"""

import json
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings
import pytest


# Feature: llm-rl-code-golf, Property 1: Task loading completeness
@pytest.mark.property
@settings(max_examples=100, deadline=500)  # Increase deadline for file I/O
@given(task_id=st.integers(min_value=1, max_value=400))
def test_task_loading_completeness(task_id):
    """
    Property 1: Task loading completeness
    
    For any valid task JSON file, parsing should extract all required fields
    (train, test, arc-gen examples) without data loss.
    
    Validates: Requirements 1.1
    """
    # Construct the path to the task file
    task_file = Path(f"google-code-golf-2025/task{task_id:03d}.json")
    
    # Skip if file doesn't exist (some task IDs might not have files)
    if not task_file.exists():
        pytest.skip(f"Task file {task_file} does not exist")
    
    # Load the raw JSON
    with open(task_file, 'r') as f:
        raw_data = json.load(f)
    
    # Property: All required fields must be present
    assert "train" in raw_data, f"Task {task_id} missing 'train' field"
    assert "test" in raw_data, f"Task {task_id} missing 'test' field"
    assert "arc-gen" in raw_data, f"Task {task_id} missing 'arc-gen' field"
    
    # Property: Each field must be a list
    assert isinstance(raw_data["train"], list), f"Task {task_id} 'train' is not a list"
    assert isinstance(raw_data["test"], list), f"Task {task_id} 'test' is not a list"
    assert isinstance(raw_data["arc-gen"], list), f"Task {task_id} 'arc-gen' is not a list"
    
    # Property: Each example must have input and output fields
    for example_type in ["train", "test", "arc-gen"]:
        for idx, example in enumerate(raw_data[example_type]):
            assert "input" in example, \
                f"Task {task_id} {example_type}[{idx}] missing 'input' field"
            assert "output" in example, \
                f"Task {task_id} {example_type}[{idx}] missing 'output' field"
            
            # Property: Input and output must be 2D lists (grids)
            assert isinstance(example["input"], list), \
                f"Task {task_id} {example_type}[{idx}] 'input' is not a list"
            assert isinstance(example["output"], list), \
                f"Task {task_id} {example_type}[{idx}] 'output' is not a list"
            
            # Property: Grids must be non-empty
            assert len(example["input"]) > 0, \
                f"Task {task_id} {example_type}[{idx}] 'input' is empty"
            assert len(example["output"]) > 0, \
                f"Task {task_id} {example_type}[{idx}] 'output' is empty"
            
            # Property: Each row must be a list
            for row_idx, row in enumerate(example["input"]):
                assert isinstance(row, list), \
                    f"Task {task_id} {example_type}[{idx}] input row {row_idx} is not a list"
            
            for row_idx, row in enumerate(example["output"]):
                assert isinstance(row, list), \
                    f"Task {task_id} {example_type}[{idx}] output row {row_idx} is not a list"
    
    # Property: No data loss - if we serialize and deserialize, we get the same data
    serialized = json.dumps(raw_data)
    deserialized = json.loads(serialized)
    assert deserialized == raw_data, \
        f"Task {task_id} data loss during serialization round-trip"


@pytest.mark.unit
def test_all_400_tasks_exist():
    """
    Verify that all 400 task files can be found and loaded.
    
    This is a concrete test that complements the property-based test.
    """
    task_dir = Path("google-code-golf-2025")
    assert task_dir.exists(), "Task directory does not exist"
    
    missing_tasks = []
    invalid_tasks = []
    
    for task_id in range(1, 401):
        task_file = task_dir / f"task{task_id:03d}.json"
        
        if not task_file.exists():
            missing_tasks.append(task_id)
            continue
        
        try:
            with open(task_file, 'r') as f:
                data = json.load(f)
            
            # Verify required fields
            if not all(field in data for field in ["train", "test", "arc-gen"]):
                invalid_tasks.append(task_id)
        except (json.JSONDecodeError, IOError) as e:
            invalid_tasks.append(task_id)
    
    # Report findings
    if missing_tasks:
        pytest.fail(f"Missing {len(missing_tasks)} task files: {missing_tasks[:10]}...")
    
    if invalid_tasks:
        pytest.fail(f"Found {len(invalid_tasks)} invalid task files: {invalid_tasks[:10]}...")
