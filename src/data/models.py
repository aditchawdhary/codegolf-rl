"""
Data models for code golf tasks.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class Example:
    """Represents a single input-output example."""
    input_grid: List[List[int]]
    output_grid: List[List[int]]
    
    @property
    def grid_size(self) -> Tuple[int, int]:
        """Returns (height, width) of the input grid."""
        if not self.input_grid:
            return (0, 0)
        return (len(self.input_grid), len(self.input_grid[0]) if self.input_grid else 0)


@dataclass
class Task:
    """Represents a complete code golf task with examples."""
    task_id: int
    train_examples: List[Example]
    test_examples: List[Example]
    arc_gen_examples: List[Example]
    difficulty_score: float = 0.0
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_examples(self) -> int:
        """Total number of examples across all sets."""
        return len(self.train_examples) + len(self.test_examples) + len(self.arc_gen_examples)
