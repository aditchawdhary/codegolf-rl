"""
Property-based tests for task validation functionality.

Feature: llm-rl-code-golf
"""

import json
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings
import pytest
from src.data import TaskLoader


# Custom strategy for generating valid task data
@st.composite
def valid_task_data_strategy(draw):
    """Generate valid task JSON data."""
    num_train = draw(st.integers(min_value=1, max_value=5))
    num_test = draw(st.integers(min_value=1, max_value=3))
    num_arc_gen = draw(st.integers(min_value=1, max_value=3))
    
    def generate_example():
        height = draw(st.integers(min_value=1, max_value=5))
        width = draw(st.integers(min_value=1, max_value=5))
        
        input_grid = []
        output_grid = []
        for _ in range(height):
            input_row = draw(st.lists(
                st.integers(min_value=0, max_value=9),
                min_size=width,
                max_size=width
            ))
            output_row = draw(st.lists(
                st.integers(min_value=0, max_value=9),
                min_size=width,
                max_size=width
            ))
            input_grid.append(input_row)
            output_grid.append(output_row)
        
        return {"input": input_grid, "output": output_grid}
    
    return {
        "train": [generate_example() for _ in range(num_train)],
        "test": [generate_example() for _ in range(num_test)],
        "arc-gen": [generate_example() for _ in range(num_arc_gen)]
    }


# Feature: llm-rl-code-golf, Property 5: Task validation consistency
@pytest.mark.property
@settings(max_examples=100)
@given(task_data=valid_task_data_strategy())
def test_task_validation_consistency(task_data):
    """
    Property 5: Task validation consistency
    
    For any task file, validation should either pass (all required fields present)
    or fail (missing fields), never partially succeed.
    
    Validates: Requirements 1.5
    """
    # Create a temporary task file
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        task_file = task_dir / "task001.json"
        
        # Write valid task data
        with open(task_file, 'w') as f:
            json.dump(task_data, f)
        
        # Load the task
        loader = TaskLoader(task_dir=str(task_dir))
        
        # Property: Valid task should load successfully
        try:
            task = loader.load_task(1)
            
            # Property: Loaded task should have all required fields
            assert task.task_id == 1, "Task ID mismatch"
            assert len(task.train_examples) > 0, "No training examples loaded"
            assert len(task.test_examples) > 0, "No test examples loaded"
            assert len(task.arc_gen_examples) > 0, "No arc-gen examples loaded"
            
            # Property: Number of examples should match input data
            assert len(task.train_examples) == len(task_data["train"]), \
                "Training example count mismatch"
            assert len(task.test_examples) == len(task_data["test"]), \
                "Test example count mismatch"
            assert len(task.arc_gen_examples) == len(task_data["arc-gen"]), \
                "Arc-gen example count mismatch"
            
            # Property: All examples should have valid grids
            for example in task.train_examples + task.test_examples + task.arc_gen_examples:
                assert example.input_grid, "Example has empty input grid"
                assert example.output_grid, "Example has empty output grid"
                assert all(isinstance(row, list) for row in example.input_grid), \
                    "Input grid rows are not lists"
                assert all(isinstance(row, list) for row in example.output_grid), \
                    "Output grid rows are not lists"
            
            validation_passed = True
        except (FileNotFoundError, ValueError) as e:
            validation_passed = False
        
        # Property: Validation should be deterministic (binary: pass or fail)
        # If it passed once, it should pass again
        if validation_passed:
            task2 = loader.load_task(1)
            assert task2.task_id == task.task_id, "Inconsistent validation results"


@pytest.mark.property
@settings(max_examples=50)
@given(
    task_data=valid_task_data_strategy(),
    missing_field=st.sampled_from(["train", "test", "arc-gen"])
)
def test_missing_field_validation(task_data, missing_field):
    """
    Test that missing required fields cause validation to fail consistently.
    """
    # Remove a required field
    incomplete_data = task_data.copy()
    del incomplete_data[missing_field]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        task_file = task_dir / "task001.json"
        
        with open(task_file, 'w') as f:
            json.dump(incomplete_data, f)
        
        loader = TaskLoader(task_dir=str(task_dir))
        
        # Property: Loading should fail with ValueError
        with pytest.raises(ValueError) as exc_info:
            loader.load_task(1)
        
        # Property: Error message should mention the missing field
        assert missing_field in str(exc_info.value), \
            f"Error message doesn't mention missing field '{missing_field}'"


@pytest.mark.property
@settings(max_examples=50)
@given(task_data=valid_task_data_strategy())
def test_malformed_example_validation(task_data):
    """
    Test that malformed examples cause validation to fail.
    """
    # Create malformed data by removing input/output from first example
    malformed_data = task_data.copy()
    if malformed_data["train"]:
        malformed_data["train"][0] = {"input": [[1, 2]]}  # Missing output
    
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        task_file = task_dir / "task001.json"
        
        with open(task_file, 'w') as f:
            json.dump(malformed_data, f)
        
        loader = TaskLoader(task_dir=str(task_dir))
        
        # Property: Loading should fail
        with pytest.raises(ValueError):
            loader.load_task(1)


@pytest.mark.unit
def test_validation_with_empty_examples():
    """Test validation behavior with empty example lists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        task_file = task_dir / "task001.json"
        
        # Valid structure but empty lists
        data = {
            "train": [],
            "test": [],
            "arc-gen": []
        }
        
        with open(task_file, 'w') as f:
            json.dump(data, f)
        
        loader = TaskLoader(task_dir=str(task_dir))
        
        # Should load successfully (empty lists are valid)
        task = loader.load_task(1)
        assert len(task.train_examples) == 0
        assert len(task.test_examples) == 0
        assert len(task.arc_gen_examples) == 0


@pytest.mark.unit
def test_validation_with_invalid_json():
    """Test validation with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        task_file = task_dir / "task001.json"
        
        # Write invalid JSON
        with open(task_file, 'w') as f:
            f.write("{invalid json")
        
        loader = TaskLoader(task_dir=str(task_dir))
        
        # Should raise an error
        with pytest.raises(json.JSONDecodeError):
            loader.load_task(1)


@pytest.mark.unit
def test_validation_consistency_across_loads():
    """Test that validation is consistent across multiple loads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        task_file = task_dir / "task001.json"
        
        data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
            "arc-gen": [{"input": [[5]], "output": [[6]]}]
        }
        
        with open(task_file, 'w') as f:
            json.dump(data, f)
        
        loader = TaskLoader(task_dir=str(task_dir))
        
        # Load multiple times
        task1 = loader.load_task(1)
        task2 = loader.load_task(1)
        task3 = loader.load_task(1)
        
        # All should succeed and produce identical results
        assert task1.task_id == task2.task_id == task3.task_id
        assert len(task1.train_examples) == len(task2.train_examples) == len(task3.train_examples)
        
        # Grid data should be identical
        assert task1.train_examples[0].input_grid == task2.train_examples[0].input_grid
        assert task2.train_examples[0].input_grid == task3.train_examples[0].input_grid


@pytest.mark.unit
def test_nonexistent_file_validation():
    """Test validation behavior with nonexistent files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = TaskLoader(task_dir=str(tmpdir))
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            loader.load_task(999)
