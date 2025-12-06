"""
Property-based tests for task formatting functionality.

Feature: llm-rl-code-golf
"""

from hypothesis import given, strategies as st, settings
import pytest
from src.data import TaskFormatter


# Custom strategy for generating valid grids
@st.composite
def grid_strategy(draw):
    """Generate a valid 2D grid of integers."""
    height = draw(st.integers(min_value=1, max_value=10))
    width = draw(st.integers(min_value=1, max_value=10))
    
    # Generate grid with values 0-9
    grid = []
    for _ in range(height):
        row = draw(st.lists(
            st.integers(min_value=0, max_value=9),
            min_size=width,
            max_size=width
        ))
        grid.append(row)
    
    return grid


# Feature: llm-rl-code-golf, Property 2: Grid-to-text round trip preservation
@pytest.mark.property
@settings(max_examples=100)
@given(grid=grid_strategy())
def test_grid_to_text_round_trip(grid):
    """
    Property 2: Grid-to-text round trip preservation
    
    For any grid representation, converting to text format and back
    should preserve the grid structure and values.
    
    Validates: Requirements 1.2
    """
    formatter = TaskFormatter()
    
    # Convert grid to text
    text = formatter._grid_to_text(grid)
    
    # Property: Text should not be empty
    assert text.strip(), "Grid-to-text conversion produced empty string"
    
    # Convert text back to grid
    recovered_grid = formatter._text_to_grid(text)
    
    # Property: Round trip should preserve the grid exactly
    assert recovered_grid == grid, \
        f"Round trip failed:\nOriginal: {grid}\nRecovered: {recovered_grid}"
    
    # Property: Grid dimensions should be preserved
    assert len(recovered_grid) == len(grid), \
        f"Height mismatch: {len(recovered_grid)} != {len(grid)}"
    
    if grid:
        assert len(recovered_grid[0]) == len(grid[0]), \
            f"Width mismatch: {len(recovered_grid[0])} != {len(grid[0])}"
    
    # Property: All values should be preserved
    for i, (orig_row, recovered_row) in enumerate(zip(grid, recovered_grid)):
        assert orig_row == recovered_row, \
            f"Row {i} mismatch: {orig_row} != {recovered_row}"


@pytest.mark.property
@settings(max_examples=100)
@given(
    grid=grid_strategy(),
    include_spaces=st.booleans()
)
def test_grid_to_text_format_consistency(grid, include_spaces):
    """
    Test that grid-to-text conversion produces consistent format.
    
    The text format should:
    - Have one row per line
    - Have space-separated values
    - Be parseable back to the original grid
    """
    formatter = TaskFormatter()
    text = formatter._grid_to_text(grid)
    
    lines = text.strip().split('\n')
    
    # Property: Number of lines should equal grid height
    assert len(lines) == len(grid), \
        f"Line count mismatch: {len(lines)} != {len(grid)}"
    
    # Property: Each line should have the correct number of values
    for i, (line, row) in enumerate(zip(lines, grid)):
        values = line.strip().split()
        assert len(values) == len(row), \
            f"Row {i} value count mismatch: {len(values)} != {len(row)}"
        
        # Property: Each value should be an integer
        for val in values:
            assert val.isdigit() or (val.startswith('-') and val[1:].isdigit()), \
                f"Non-integer value found: {val}"


@pytest.mark.unit
def test_empty_grid_handling():
    """Test that empty grids are handled correctly."""
    formatter = TaskFormatter()
    
    # Empty grid should produce empty text
    empty_grid = []
    text = formatter._grid_to_text(empty_grid)
    assert text == "", "Empty grid should produce empty text"
    
    # Empty text should produce empty grid
    recovered = formatter._text_to_grid("")
    assert recovered == [], "Empty text should produce empty grid"


@pytest.mark.unit
def test_single_value_grid():
    """Test grid with a single value."""
    formatter = TaskFormatter()
    
    grid = [[5]]
    text = formatter._grid_to_text(grid)
    recovered = formatter._text_to_grid(text)
    
    assert recovered == grid, f"Single value grid failed: {recovered} != {grid}"


@pytest.mark.unit
def test_grid_with_zeros():
    """Test that grids with zeros are handled correctly."""
    formatter = TaskFormatter()
    
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    text = formatter._grid_to_text(grid)
    recovered = formatter._text_to_grid(text)
    
    assert recovered == grid, f"Grid with zeros failed: {recovered} != {grid}"


@pytest.mark.unit
def test_rectangular_grids():
    """Test grids of various rectangular shapes."""
    formatter = TaskFormatter()
    
    test_cases = [
        [[1, 2, 3]],  # 1x3
        [[1], [2], [3]],  # 3x1
        [[1, 2], [3, 4], [5, 6]],  # 3x2
        [[1, 2, 3], [4, 5, 6]],  # 2x3
    ]
    
    for grid in test_cases:
        text = formatter._grid_to_text(grid)
        recovered = formatter._text_to_grid(text)
        assert recovered == grid, \
            f"Rectangular grid failed: {recovered} != {grid}"



# Additional imports for prompt completeness tests
from src.data import Task, Example


# Custom strategy for generating tasks
@st.composite
def task_with_examples_strategy(draw):
    """Generate a task with examples for testing."""
    task_id = draw(st.integers(min_value=1, max_value=1000))
    num_train = draw(st.integers(min_value=1, max_value=5))
    num_test = draw(st.integers(min_value=0, max_value=2))
    
    # Generate examples
    train_examples = []
    for _ in range(num_train):
        grid = draw(grid_strategy())
        example = Example(input_grid=grid, output_grid=grid)
        train_examples.append(example)
    
    test_examples = []
    for _ in range(num_test):
        grid = draw(grid_strategy())
        example = Example(input_grid=grid, output_grid=grid)
        test_examples.append(example)
    
    return Task(
        task_id=task_id,
        train_examples=train_examples,
        test_examples=test_examples,
        arc_gen_examples=[]
    )


# Feature: llm-rl-code-golf, Property 4: Prompt completeness
@pytest.mark.property
@settings(max_examples=100)
@given(task=task_with_examples_strategy())
def test_prompt_completeness(task):
    """
    Property 4: Prompt completeness
    
    For any task, the formatted prompt should contain all task examples
    and generation instructions.
    
    Validates: Requirements 1.4
    """
    formatter = TaskFormatter(include_instructions=True)
    prompt = formatter.format_prompt(task, include_test=False)
    
    # Property: Prompt should not be empty
    assert prompt.strip(), "Prompt is empty"
    
    # Property: Prompt should contain instructions
    assert "# Code Golf Task" in prompt or "# Task:" in prompt, \
        "Prompt missing task header"
    
    # Property: Prompt should mention training examples
    assert "Training Examples" in prompt or "Example" in prompt, \
        "Prompt missing training examples section"
    
    # Property: Prompt should contain all training examples
    # Check that each example's data appears in the prompt
    for idx, example in enumerate(task.train_examples, 1):
        # Convert example to text and check if it's in the prompt
        example_text = formatter.format_examples([example])
        # At minimum, some values from the grid should appear
        for row in example.input_grid:
            for val in row:
                assert str(val) in prompt, \
                    f"Training example {idx} input value {val} not in prompt"
    
    # Property: Prompt should contain code generation instructions
    assert "function" in prompt.lower() or "code" in prompt.lower(), \
        "Prompt missing code generation instructions"
    
    # Property: Prompt should mention the transformation task
    assert "transform" in prompt.lower() or "input" in prompt.lower(), \
        "Prompt missing transformation description"


@pytest.mark.property
@settings(max_examples=50)
@given(task=task_with_examples_strategy(), include_test=st.booleans())
def test_prompt_test_inclusion(task, include_test):
    """
    Test that test examples are included/excluded based on parameter.
    """
    formatter = TaskFormatter(include_instructions=True)
    prompt = formatter.format_prompt(task, include_test=include_test)
    
    if include_test and task.test_examples:
        # Property: Test examples should be in prompt when requested
        assert "Test" in prompt, "Test examples not included when requested"
    elif not include_test:
        # Property: Test examples should not be in prompt when not requested
        # (unless they happen to be mentioned in instructions)
        test_count = prompt.count("# Test")
        assert test_count == 0, "Test examples included when not requested"


@pytest.mark.property
@settings(max_examples=50)
@given(task=task_with_examples_strategy())
def test_prompt_structure_consistency(task):
    """
    Test that prompts have consistent structure.
    """
    formatter = TaskFormatter(include_instructions=True)
    prompt = formatter.format_prompt(task, include_test=False)
    
    lines = prompt.split('\n')
    
    # Property: Prompt should have multiple lines
    assert len(lines) > 1, "Prompt should be multi-line"
    
    # Property: Prompt should have sections (marked with #)
    section_count = sum(1 for line in lines if line.strip().startswith('#'))
    assert section_count >= 2, f"Prompt should have at least 2 sections, got {section_count}"
    
    # Property: Each example should have Input and Output
    input_count = prompt.count("Input:")
    output_count = prompt.count("Output:")
    
    # Should have at least as many inputs/outputs as training examples
    assert input_count >= len(task.train_examples), \
        f"Not enough Input sections: {input_count} < {len(task.train_examples)}"
    assert output_count >= len(task.train_examples), \
        f"Not enough Output sections: {output_count} < {len(task.train_examples)}"


@pytest.mark.unit
def test_prompt_without_instructions():
    """Test prompt generation without instructions."""
    formatter = TaskFormatter(include_instructions=False)
    
    task = Task(
        task_id=1,
        train_examples=[Example([[1, 2]], [[3, 4]])],
        test_examples=[],
        arc_gen_examples=[]
    )
    
    prompt = formatter.format_prompt(task, include_test=False)
    
    # Should not contain the full instructions header
    assert "# Code Golf Task" not in prompt
    
    # But should still contain examples and task description
    assert "Example" in prompt or "Training" in prompt
    assert "1 2" in prompt  # Input values
    assert "3 4" in prompt  # Output values


@pytest.mark.unit
def test_prompt_with_multiple_examples():
    """Test prompt with multiple training examples."""
    formatter = TaskFormatter(include_instructions=True)
    
    task = Task(
        task_id=1,
        train_examples=[
            Example([[1]], [[2]]),
            Example([[3]], [[4]]),
            Example([[5]], [[6]]),
        ],
        test_examples=[],
        arc_gen_examples=[]
    )
    
    prompt = formatter.format_prompt(task, include_test=False)
    
    # All examples should be present
    assert "1" in prompt
    assert "2" in prompt
    assert "3" in prompt
    assert "4" in prompt
    assert "5" in prompt
    assert "6" in prompt
    
    # Should have multiple example sections
    example_count = prompt.count("Example")
    assert example_count >= 3, f"Expected at least 3 example mentions, got {example_count}"
