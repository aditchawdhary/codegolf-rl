"""
Property-based tests for difficulty analysis functionality.

Feature: llm-rl-code-golf
"""

from hypothesis import given, strategies as st, settings, assume
import pytest
from src.data import Task, Example, DifficultyAnalyzer


# Custom strategy for generating valid examples
@st.composite
def example_strategy(draw):
    """Generate a valid Example."""
    height = draw(st.integers(min_value=1, max_value=10))
    width = draw(st.integers(min_value=1, max_value=10))
    
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
    
    return Example(input_grid=input_grid, output_grid=output_grid)


# Custom strategy for generating valid tasks
@st.composite
def task_strategy(draw):
    """Generate a valid Task."""
    task_id = draw(st.integers(min_value=1, max_value=1000))
    num_train = draw(st.integers(min_value=1, max_value=5))
    
    train_examples = [draw(example_strategy()) for _ in range(num_train)]
    test_examples = [draw(example_strategy())]
    arc_gen_examples = [draw(example_strategy())]
    
    return Task(
        task_id=task_id,
        train_examples=train_examples,
        test_examples=test_examples,
        arc_gen_examples=arc_gen_examples
    )


# Feature: llm-rl-code-golf, Property 3: Batch difficulty ordering
@pytest.mark.property
@settings(max_examples=100)
@given(tasks=st.lists(task_strategy(), min_size=3, max_size=20))
def test_batch_difficulty_ordering(tasks):
    """
    Property 3: Batch difficulty ordering
    
    For any set of tasks with difficulty scores, batches should maintain
    monotonic ordering within difficulty groups.
    
    Validates: Requirements 1.3
    """
    analyzer = DifficultyAnalyzer()
    
    # Compute difficulty scores for all tasks
    for task in tasks:
        analyzer.compute_complexity_score(task)
    
    # Categorize tasks by difficulty
    categories = analyzer.categorize_by_difficulty(tasks)
    
    # Property: All tasks should be categorized
    total_categorized = sum(len(cat_tasks) for cat_tasks in categories.values())
    assert total_categorized == len(tasks), \
        f"Not all tasks categorized: {total_categorized} != {len(tasks)}"
    
    # Property: Categories should exist
    assert "easy" in categories, "Missing 'easy' category"
    assert "medium" in categories, "Missing 'medium' category"
    assert "hard" in categories, "Missing 'hard' category"
    
    # Property: Within each category, tasks should be ordered by difficulty
    for category_name, category_tasks in categories.items():
        if len(category_tasks) > 1:
            scores = [t.difficulty_score for t in category_tasks]
            # Check monotonic non-decreasing order
            for i in range(len(scores) - 1):
                assert scores[i] <= scores[i + 1], \
                    f"Category '{category_name}' not ordered: {scores[i]} > {scores[i + 1]}"
    
    # Property: Easy tasks should have lower scores than hard tasks
    if categories["easy"] and categories["hard"]:
        max_easy = max(t.difficulty_score for t in categories["easy"])
        min_hard = min(t.difficulty_score for t in categories["hard"])
        assert max_easy <= min_hard, \
            f"Easy tasks overlap with hard tasks: max_easy={max_easy}, min_hard={min_hard}"
    
    # Property: Medium tasks should be between easy and hard
    if categories["easy"] and categories["medium"] and categories["hard"]:
        max_easy = max(t.difficulty_score for t in categories["easy"])
        min_medium = min(t.difficulty_score for t in categories["medium"])
        max_medium = max(t.difficulty_score for t in categories["medium"])
        min_hard = min(t.difficulty_score for t in categories["hard"])
        
        assert max_easy <= min_medium, \
            f"Easy tasks overlap with medium: max_easy={max_easy}, min_medium={min_medium}"
        assert max_medium <= min_hard, \
            f"Medium tasks overlap with hard: max_medium={max_medium}, min_hard={min_hard}"


@pytest.mark.property
@settings(max_examples=50)
@given(tasks=st.lists(task_strategy(), min_size=1, max_size=10))
def test_difficulty_score_consistency(tasks):
    """
    Test that difficulty scores are consistent and deterministic.
    """
    analyzer = DifficultyAnalyzer()
    
    # Compute scores twice
    scores1 = []
    for task in tasks:
        score = analyzer.compute_complexity_score(task)
        scores1.append(score)
    
    scores2 = []
    for task in tasks:
        score = analyzer.compute_complexity_score(task)
        scores2.append(score)
    
    # Property: Scores should be identical on repeated computation
    assert scores1 == scores2, \
        f"Difficulty scores not consistent: {scores1} != {scores2}"
    
    # Property: All scores should be non-negative
    for score in scores1:
        assert score >= 0, f"Negative difficulty score: {score}"
    
    # Property: All scores should be bounded (0-100)
    for score in scores1:
        assert score <= 100, f"Difficulty score exceeds maximum: {score}"


@pytest.mark.property
@settings(max_examples=50)
@given(task=task_strategy())
def test_complexity_metrics_completeness(task):
    """
    Test that complexity metrics are computed and stored.
    """
    analyzer = DifficultyAnalyzer()
    score = analyzer.compute_complexity_score(task)
    
    # Property: Complexity metrics should be populated
    assert task.complexity_metrics, "Complexity metrics not populated"
    
    # Property: Required metrics should be present
    required_metrics = [
        "avg_grid_size",
        "avg_unique_values",
        "avg_size_ratio",
        "num_train_examples"
    ]
    
    for metric in required_metrics:
        assert metric in task.complexity_metrics, \
            f"Missing required metric: {metric}"
    
    # Property: Metrics should be numeric
    for metric, value in task.complexity_metrics.items():
        assert isinstance(value, (int, float)), \
            f"Metric '{metric}' is not numeric: {type(value)}"


@pytest.mark.unit
def test_difficulty_distribution():
    """Test that difficulty distribution is computed correctly."""
    analyzer = DifficultyAnalyzer()
    
    # Create tasks with known difficulty scores
    tasks = []
    for i in range(9):
        task = Task(
            task_id=i,
            train_examples=[Example([[1]], [[1]])],
            test_examples=[],
            arc_gen_examples=[]
        )
        task.difficulty_score = i * 10  # Scores: 0, 10, 20, ..., 80
        tasks.append(task)
    
    distribution = analyzer.get_difficulty_distribution(tasks)
    
    # Should have 3 tasks in each category (9 tasks / 3 categories)
    assert distribution["easy"] == 3, f"Expected 3 easy tasks, got {distribution['easy']}"
    assert distribution["medium"] == 3, f"Expected 3 medium tasks, got {distribution['medium']}"
    assert distribution["hard"] == 3, f"Expected 3 hard tasks, got {distribution['hard']}"


@pytest.mark.unit
def test_single_task_categorization():
    """Test categorization with a single task."""
    analyzer = DifficultyAnalyzer()
    
    task = Task(
        task_id=1,
        train_examples=[Example([[1, 2]], [[3, 4]])],
        test_examples=[],
        arc_gen_examples=[]
    )
    
    analyzer.compute_complexity_score(task)
    categories = analyzer.categorize_by_difficulty([task])
    
    # Single task goes to hard category (n//3 = 0, so easy and medium are empty)
    assert len(categories["easy"]) == 0
    assert len(categories["medium"]) == 0
    assert len(categories["hard"]) == 1
    
    # Total should still be 1
    total = sum(len(cat) for cat in categories.values())
    assert total == 1
