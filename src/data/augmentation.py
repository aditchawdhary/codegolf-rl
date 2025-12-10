"""
Data augmentation for code golf tasks.

Implements geometric transformations (rotations, flips) that preserve
the transformation logic while providing diverse training examples.

Based on the ARC paper approach.
"""

import copy
from typing import List, Tuple
from .models import Task, Example


class ProblemAugmenter:
    """
    Augments code golf tasks using geometric transformations.
    
    Transformations:
    - Rotations: 0°, 90°, 180°, 270°
    - Flips: horizontal, vertical
    
    These preserve the transformation logic while creating diverse examples.
    """
    
    def __init__(self, augmentation_types: List[str] = None):
        """
        Initialize augmenter.
        
        Args:
            augmentation_types: List of augmentation types to use.
                               If None, uses all available types.
        """
        self.available_types = [
            'rotate_90',
            'rotate_180', 
            'rotate_270',
            'flip_horizontal',
            'flip_vertical',
            'identity'  # Original, no transformation
        ]
        
        if augmentation_types is None:
            self.augmentation_types = self.available_types
        else:
            # Validate types
            for aug_type in augmentation_types:
                if aug_type not in self.available_types:
                    raise ValueError(f"Unknown augmentation type: {aug_type}")
            self.augmentation_types = augmentation_types
        
        print(f"[ProblemAugmenter] Initialized with types: {self.augmentation_types}")
    
    def augment(self, task: Task, augmentation_type: str) -> Task:
        """
        Apply a single augmentation to a task.
        
        Args:
            task: Task to augment
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented task (new copy)
        """
        if augmentation_type not in self.augmentation_types:
            raise ValueError(f"Augmentation type not enabled: {augmentation_type}")
        
        # Create a deep copy
        augmented = copy.deepcopy(task)
        
        # Apply transformation to all examples
        if augmentation_type == 'identity':
            # No transformation
            pass
        elif augmentation_type == 'rotate_90':
            augmented.train_examples = [
                self._rotate_example(ex, 90) for ex in augmented.train_examples
            ]
            augmented.test_examples = [
                self._rotate_example(ex, 90) for ex in augmented.test_examples
            ]
            augmented.arc_gen_examples = [
                self._rotate_example(ex, 90) for ex in augmented.arc_gen_examples
            ]
        elif augmentation_type == 'rotate_180':
            augmented.train_examples = [
                self._rotate_example(ex, 180) for ex in augmented.train_examples
            ]
            augmented.test_examples = [
                self._rotate_example(ex, 180) for ex in augmented.test_examples
            ]
            augmented.arc_gen_examples = [
                self._rotate_example(ex, 180) for ex in augmented.arc_gen_examples
            ]
        elif augmentation_type == 'rotate_270':
            augmented.train_examples = [
                self._rotate_example(ex, 270) for ex in augmented.train_examples
            ]
            augmented.test_examples = [
                self._rotate_example(ex, 270) for ex in augmented.test_examples
            ]
            augmented.arc_gen_examples = [
                self._rotate_example(ex, 270) for ex in augmented.arc_gen_examples
            ]
        elif augmentation_type == 'flip_horizontal':
            augmented.train_examples = [
                self._flip_horizontal_example(ex) for ex in augmented.train_examples
            ]
            augmented.test_examples = [
                self._flip_horizontal_example(ex) for ex in augmented.test_examples
            ]
            augmented.arc_gen_examples = [
                self._flip_horizontal_example(ex) for ex in augmented.arc_gen_examples
            ]
        elif augmentation_type == 'flip_vertical':
            augmented.train_examples = [
                self._flip_vertical_example(ex) for ex in augmented.train_examples
            ]
            augmented.test_examples = [
                self._flip_vertical_example(ex) for ex in augmented.test_examples
            ]
            augmented.arc_gen_examples = [
                self._flip_vertical_example(ex) for ex in augmented.arc_gen_examples
            ]
        
        return augmented
    
    def augment_multiple(self, task: Task, n: int = 4) -> List[Task]:
        """
        Generate multiple augmented versions of a task.
        
        Args:
            task: Task to augment
            n: Number of augmentations to generate
            
        Returns:
            List of augmented tasks (includes original)
        """
        augmented_tasks = []
        
        # Always include original
        augmented_tasks.append(copy.deepcopy(task))
        
        # Generate n-1 augmented versions
        aug_count = 0
        for aug_type in self.augmentation_types:
            if aug_type == 'identity':
                continue  # Already included original
            
            if aug_count >= n - 1:
                break
            
            augmented = self.augment(task, aug_type)
            augmented_tasks.append(augmented)
            aug_count += 1
        
        return augmented_tasks[:n]
    
    # ------------------------------------------------------------------------
    # Transformation primitives
    # ------------------------------------------------------------------------
    
    def _rotate_example(self, example: Example, degrees: int) -> Example:
        """Rotate an example by specified degrees."""
        rotated_input = self._rotate_grid(example.input_grid, degrees)
        rotated_output = self._rotate_grid(example.output_grid, degrees)
        return Example(rotated_input, rotated_output)
    
    def _flip_horizontal_example(self, example: Example) -> Example:
        """Flip an example horizontally."""
        flipped_input = self._flip_horizontal_grid(example.input_grid)
        flipped_output = self._flip_horizontal_grid(example.output_grid)
        return Example(flipped_input, flipped_output)
    
    def _flip_vertical_example(self, example: Example) -> Example:
        """Flip an example vertically."""
        flipped_input = self._flip_vertical_grid(example.input_grid)
        flipped_output = self._flip_vertical_grid(example.output_grid)
        return Example(flipped_input, flipped_output)
    
    def _rotate_grid(self, grid: List[List[int]], degrees: int) -> List[List[int]]:
        """
        Rotate a grid by specified degrees (90, 180, 270).
        
        Args:
            grid: 2D grid to rotate
            degrees: Rotation angle (90, 180, 270)
            
        Returns:
            Rotated grid
        """
        if not grid or not grid[0]:
            return grid
        
        if degrees == 90:
            # Rotate 90° clockwise: transpose then reverse each row
            transposed = list(zip(*grid))
            return [list(reversed(row)) for row in transposed]
        
        elif degrees == 180:
            # Rotate 180°: reverse rows then reverse each row
            return [list(reversed(row)) for row in reversed(grid)]
        
        elif degrees == 270:
            # Rotate 270° clockwise (= 90° counter-clockwise)
            # Reverse each row then transpose
            reversed_rows = [list(reversed(row)) for row in grid]
            return [list(row) for row in zip(*reversed_rows)]
        
        else:
            raise ValueError(f"Invalid rotation angle: {degrees}")
    
    def _flip_horizontal_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Flip a grid horizontally (mirror left-right).
        
        Args:
            grid: 2D grid to flip
            
        Returns:
            Flipped grid
        """
        return [list(reversed(row)) for row in grid]
    
    def _flip_vertical_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Flip a grid vertically (mirror top-bottom).
        
        Args:
            grid: 2D grid to flip
            
        Returns:
            Flipped grid
        """
        return list(reversed(grid))


class TypedAugmenter(ProblemAugmenter):
    """
    Type-aware augmenter that only applies augmentations
    that preserve the problem semantics.
    
    For example, some problems are rotation-invariant but not flip-invariant.
    This augmenter can be configured to respect those constraints.
    """
    
    def __init__(
        self, 
        allow_rotations: bool = True,
        allow_flips: bool = True,
        rotation_angles: List[int] = None
    ):
        """
        Initialize typed augmenter.
        
        Args:
            allow_rotations: Whether to allow rotations
            allow_flips: Whether to allow flips
            rotation_angles: Specific rotation angles to allow (90, 180, 270)
        """
        self.allow_rotations = allow_rotations
        self.allow_flips = allow_flips
        
        # Build augmentation types based on constraints
        augmentation_types = ['identity']
        
        if allow_rotations:
            if rotation_angles is None:
                rotation_angles = [90, 180, 270]
            
            for angle in rotation_angles:
                if angle == 90:
                    augmentation_types.append('rotate_90')
                elif angle == 180:
                    augmentation_types.append('rotate_180')
                elif angle == 270:
                    augmentation_types.append('rotate_270')
        
        if allow_flips:
            augmentation_types.append('flip_horizontal')
            augmentation_types.append('flip_vertical')
        
        super().__init__(augmentation_types=augmentation_types)
        
        print(f"[TypedAugmenter] Initialized:")
        print(f"  Allow rotations: {allow_rotations}")
        print(f"  Allow flips: {allow_flips}")
        print(f"  Types: {self.augmentation_types}")
    
    def is_rotation_invariant(self, task: Task) -> bool:
        """
        Check if a task appears to be rotation-invariant.
        
        This is a heuristic check - not perfect but useful.
        """
        # Simple heuristic: if all grids are square, more likely rotation-invariant
        for example in task.train_examples:
            height = len(example.input_grid)
            width = len(example.input_grid[0]) if example.input_grid else 0
            
            if height != width:
                return False
        
        return True
    
    def auto_configure(self, task: Task):
        """
        Automatically configure augmentation types based on task properties.
        
        This analyzes the task and enables appropriate augmentations.
        """
        # Check if rotation-invariant
        if self.is_rotation_invariant(task):
            print(f"  Task {task.task_id} appears rotation-invariant")
            self.allow_rotations = True
        else:
            print(f"  Task {task.task_id} may not be rotation-invariant")
            self.allow_rotations = False
        
        # Rebuild augmentation types
        self.__init__(
            allow_rotations=self.allow_rotations,
            allow_flips=self.allow_flips
        )


# ------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------

def visualize_augmentation(task: Task, augmentation_type: str):
    """
    Visualize an augmentation by printing before/after grids.
    
    Useful for debugging and understanding transformations.
    """
    augmenter = ProblemAugmenter()
    augmented = augmenter.augment(task, augmentation_type)
    
    print(f"\nAugmentation: {augmentation_type}")
    print("=" * 60)
    
    # Show first training example
    if task.train_examples:
        original = task.train_examples[0]
        transformed = augmented.train_examples[0]
        
        print("\nOriginal Input:")
        for row in original.input_grid:
            print("  ", row)
        
        print("\nOriginal Output:")
        for row in original.output_grid:
            print("  ", row)
        
        print(f"\nTransformed Input ({augmentation_type}):")
        for row in transformed.input_grid:
            print("  ", row)
        
        print(f"\nTransformed Output ({augmentation_type}):")
        for row in transformed.output_grid:
            print("  ", row)
        
        print("=" * 60)


def test_augmentations():
    """Test augmentations with a simple example."""
    # Create a simple test task
    test_example = Example(
        input_grid=[[1, 2], [3, 4]],
        output_grid=[[5, 6], [7, 8]]
    )
    
    test_task = Task(
        task_id=999,
        train_examples=[test_example],
        test_examples=[],
        arc_gen_examples=[]
    )
    
    augmenter = ProblemAugmenter()
    
    print("\nTesting augmentations:")
    print("=" * 60)
    
    for aug_type in augmenter.augmentation_types:
        visualize_augmentation(test_task, aug_type)


if __name__ == "__main__":
    # Run tests if executed directly
    test_augmentations()
