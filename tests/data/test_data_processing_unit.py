"""
Unit tests for data processing components.

Tests task loading, prompt formatting, and difficulty scoring edge cases.
"""

import json
import tempfile
from pathlib import Path
import pytest
from src.data import TaskLoader, TaskFormatter, DifficultyAnalyzer, Task, Example


class TestTaskLoader:
    """Unit tests for TaskLoader."""
    
    def test_load_task_with_valid_json(self):
        """Test task loading with valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            task_file = task_dir / "task001.json"
            
            data = {
                "train": [{"input": [[1, 2]], "output": [[3, 4]]}],
                "test": [{"input": [[5, 6]], "output": [[7, 8]]}],
                "arc-gen": [{"input": [[9, 0]], "output": [[1, 2]]}]
            }
            
            with open(task_file, 'w') as f:
                json.dump(data, f)
            
            loader = TaskLoader(task_dir=str(task_dir))
            task = loader.load_task(1)
            
            assert task.task_id == 1
            assert len(task.train_examples) == 1
            assert len(task.test_examples) == 1
            assert len(task.arc_gen_examples) == 1
    
    def test_load_task_with_invalid_json(self):
        """Test task loading with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            task_file = task_dir / "task001.json"
            
            with open(task_file, 'w') as f:
                f.write("{invalid")
            
            loader = TaskLoader(task_dir=str(task_dir))
            
            with pytest.raises(json.JSONDecodeError):
                loader.load_task(1)
    
    def test_load_task_missing_file(self):
        """Test loading nonexistent task file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = TaskLoader(task_dir=str(tmpdir))
            
            with pytest.raises(FileNotFoundError):
                loader.load_task(999)
    
    def test_load_all_tasks(self):
        """Test loading multiple tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            
            # Create 3 task files
            for i in range(1, 4):
                task_file = task_dir / f"task{i:03d}.json"
                data = {
                    "train": [{"input": [[i]], "output": [[i*2]]}],
                    "test": [{"input": [[i+1]], "output": [[i*3]]}],
                    "arc-gen": []
                }
                with open(task_file, 'w') as f:
                    json.dump(data, f)
            
            loader = TaskLoader(task_dir=str(task_dir))
            tasks = loader.load_all_tasks()
            
            assert len(tasks) == 3
            assert all(isinstance(t, Task) for t in tasks)
    
    def test_get_task_statistics(self):
        """Test task statistics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            
            # Create 2 tasks with different numbers of examples
            for i in range(1, 3):
                task_file = task_dir / f"task{i:03d}.json"
                data = {
                    "train": [{"input": [[1]], "output": [[2]]}] * i,
                    "test": [{"input": [[3]], "output": [[4]]}],
                    "arc-gen": [{"input": [[5]], "output": [[6]]}] * (i + 1)
                }
                with open(task_file, 'w') as f:
                    json.dump(data, f)
            
            loader = TaskLoader(task_dir=str(task_dir))
            stats = loader.get_task_statistics()
            
            assert stats["num_tasks"] == 2
            assert stats["total_train_examples"] == 3  # 1 + 2
            assert stats["total_test_examples"] == 2  # 1 + 1
            assert stats["total_arc_gen_examples"] == 5  # 2 + 3
    
    def test_invalid_task_directory(self):
        """Test initialization with invalid directory."""
        with pytest.raises(ValueError):
            TaskLoader(task_dir="/nonexistent/directory")


class TestTaskFormatter:
    """Unit tests for TaskFormatter."""
    
    def test_format_prompt_with_instructions(self):
        """Test prompt formatting with instructions."""
        formatter = TaskFormatter(include_instructions=True)
        task = Task(
            task_id=1,
            train_examples=[Example([[1, 2]], [[3, 4]])],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        prompt = formatter.format_prompt(task)
        
        assert "# Code Golf Task" in prompt
        assert "Training Examples" in prompt
        assert "1 2" in prompt
        assert "3 4" in prompt
    
    def test_format_prompt_without_instructions(self):
        """Test prompt formatting without instructions."""
        formatter = TaskFormatter(include_instructions=False)
        task = Task(
            task_id=1,
            train_examples=[Example([[1]], [[2]])],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        prompt = formatter.format_prompt(task)
        
        assert "# Code Golf Task" not in prompt
        assert "1" in prompt
        assert "2" in prompt
    
    def test_format_prompt_with_test_examples(self):
        """Test prompt formatting including test examples."""
        formatter = TaskFormatter(include_instructions=True)
        task = Task(
            task_id=1,
            train_examples=[Example([[1]], [[2]])],
            test_examples=[Example([[3]], [[4]])],
            arc_gen_examples=[]
        )
        
        prompt = formatter.format_prompt(task, include_test=True)
        
        assert "Test" in prompt
        assert "3" in prompt
        assert "4" in prompt
    
    def test_format_prompt_without_test_examples(self):
        """Test prompt formatting excluding test examples."""
        formatter = TaskFormatter(include_instructions=True)
        task = Task(
            task_id=1,
            train_examples=[Example([[1]], [[2]])],
            test_examples=[Example([[3]], [[4]])],
            arc_gen_examples=[]
        )
        
        prompt = formatter.format_prompt(task, include_test=False)
        
        # Test section should not be present
        assert "# Test" not in prompt
    
    def test_format_examples(self):
        """Test example formatting."""
        formatter = TaskFormatter()
        examples = [
            Example([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            Example([[9]], [[0]])
        ]
        
        text = formatter.format_examples(examples)
        
        assert "Input:" in text
        assert "Output:" in text
        assert "1 2" in text
        assert "5 6" in text
        assert "9" in text
        assert "0" in text
    
    def test_create_training_batch(self):
        """Test batch creation."""
        formatter = TaskFormatter()
        tasks = [
            Task(i, [], [], [])
            for i in range(10)
        ]
        
        batches = formatter.create_training_batch(tasks, batch_size=3)
        
        assert len(batches) == 4  # 10 tasks / 3 per batch = 4 batches
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1  # Last batch has remainder
    
    def test_grid_to_text_various_sizes(self):
        """Test grid-to-text conversion with various grid sizes."""
        formatter = TaskFormatter()
        
        # 1x1 grid
        grid = [[5]]
        text = formatter._grid_to_text(grid)
        assert text == "5"
        
        # 2x3 grid
        grid = [[1, 2, 3], [4, 5, 6]]
        text = formatter._grid_to_text(grid)
        lines = text.split('\n')
        assert len(lines) == 2
        assert lines[0] == "1 2 3"
        assert lines[1] == "4 5 6"
    
    def test_text_to_grid_various_formats(self):
        """Test text-to-grid conversion with various formats."""
        formatter = TaskFormatter()
        
        # Single value
        text = "5"
        grid = formatter._text_to_grid(text)
        assert grid == [[5]]
        
        # Multiple rows
        text = "1 2 3\n4 5 6"
        grid = formatter._text_to_grid(text)
        assert grid == [[1, 2, 3], [4, 5, 6]]
        
        # With extra whitespace
        text = "  1   2  \n  3   4  "
        grid = formatter._text_to_grid(text)
        assert grid == [[1, 2], [3, 4]]


class TestDifficultyAnalyzer:
    """Unit tests for DifficultyAnalyzer."""
    
    def test_compute_complexity_score_simple(self):
        """Test complexity score computation for simple task."""
        analyzer = DifficultyAnalyzer()
        task = Task(
            task_id=1,
            train_examples=[Example([[1]], [[2]])],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        score = analyzer.compute_complexity_score(task)
        
        assert 0 <= score <= 100
        assert task.difficulty_score == score
        assert "avg_grid_size" in task.complexity_metrics
    
    def test_compute_complexity_score_large_grid(self):
        """Test complexity score for large grids."""
        analyzer = DifficultyAnalyzer()
        
        # Large grid (10x10)
        large_grid = [[i for i in range(10)] for _ in range(10)]
        task = Task(
            task_id=1,
            train_examples=[Example(large_grid, large_grid)],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        score = analyzer.compute_complexity_score(task)
        
        # Large grids should have higher complexity
        assert score > 20
    
    def test_compute_complexity_score_many_unique_values(self):
        """Test complexity score with many unique values."""
        analyzer = DifficultyAnalyzer()
        
        # Grid with all unique values 0-9
        grid = [[i for i in range(10)]]
        task = Task(
            task_id=1,
            train_examples=[Example(grid, grid)],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        score = analyzer.compute_complexity_score(task)
        
        # More unique values should increase complexity
        assert score > 0
    
    def test_compute_complexity_score_size_change(self):
        """Test complexity score when output size differs from input."""
        analyzer = DifficultyAnalyzer()
        
        # Input 2x2, output 4x4 (size ratio = 4)
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 2, 3, 4]] * 4
        task = Task(
            task_id=1,
            train_examples=[Example(input_grid, output_grid)],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        score = analyzer.compute_complexity_score(task)
        
        # Size changes should increase complexity
        assert score > 10
    
    def test_categorize_by_difficulty_equal_distribution(self):
        """Test difficulty categorization with equal distribution."""
        analyzer = DifficultyAnalyzer()
        
        # Create 9 tasks with scores 10, 20, 30, ..., 90
        tasks = []
        for i in range(1, 10):
            task = Task(
                task_id=i,
                train_examples=[Example([[1]], [[1]])],
                test_examples=[],
                arc_gen_examples=[]
            )
            task.difficulty_score = i * 10
            tasks.append(task)
        
        categories = analyzer.categorize_by_difficulty(tasks)
        
        # Should have 3 in each category
        assert len(categories["easy"]) == 3
        assert len(categories["medium"]) == 3
        assert len(categories["hard"]) == 3
        
        # Easy should have lowest scores
        assert all(t.difficulty_score <= 30 for t in categories["easy"])
        # Hard should have highest scores
        assert all(t.difficulty_score >= 70 for t in categories["hard"])
    
    def test_categorize_by_difficulty_edge_cases(self):
        """Test difficulty categorization edge cases."""
        analyzer = DifficultyAnalyzer()
        
        # Single task
        task = Task(1, [Example([[1]], [[1]])], [], [])
        task.difficulty_score = 50
        categories = analyzer.categorize_by_difficulty([task])
        assert sum(len(cat) for cat in categories.values()) == 1
        
        # Two tasks
        tasks = [
            Task(1, [Example([[1]], [[1]])], [], []),
            Task(2, [Example([[2]], [[2]])], [], [])
        ]
        tasks[0].difficulty_score = 10
        tasks[1].difficulty_score = 90
        categories = analyzer.categorize_by_difficulty(tasks)
        assert sum(len(cat) for cat in categories.values()) == 2
    
    def test_get_difficulty_distribution(self):
        """Test difficulty distribution computation."""
        analyzer = DifficultyAnalyzer()
        
        tasks = []
        for i in range(12):
            task = Task(i, [Example([[1]], [[1]])], [], [])
            task.difficulty_score = i * 10
            tasks.append(task)
        
        distribution = analyzer.get_difficulty_distribution(tasks)
        
        assert "easy" in distribution
        assert "medium" in distribution
        assert "hard" in distribution
        assert distribution["easy"] == 4
        assert distribution["medium"] == 4
        assert distribution["hard"] == 4


class TestDataModels:
    """Unit tests for data models."""
    
    def test_example_grid_size(self):
        """Test Example grid_size property."""
        example = Example([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]])
        assert example.grid_size == (2, 3)
        
        # Empty grid
        example = Example([], [])
        assert example.grid_size == (0, 0)
    
    def test_task_total_examples(self):
        """Test Task total_examples property."""
        task = Task(
            task_id=1,
            train_examples=[Example([[1]], [[2]]), Example([[3]], [[4]])],
            test_examples=[Example([[5]], [[6]])],
            arc_gen_examples=[Example([[7]], [[8]]), Example([[9]], [[0]])]
        )
        
        assert task.total_examples == 5  # 2 + 1 + 2
    
    def test_task_initialization(self):
        """Test Task initialization with defaults."""
        task = Task(
            task_id=1,
            train_examples=[],
            test_examples=[],
            arc_gen_examples=[]
        )
        
        assert task.difficulty_score == 0.0
        assert task.complexity_metrics == {}
