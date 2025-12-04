# LLM-based Reinforcement Learning for Google Code Golf 2025

An LLM-based reinforcement learning system that learns to solve Google Code Golf 2025 challenges through PPO training.

## Project Structure

```
.
├── src/
│   ├── data/           # Data processing and task loading
│   ├── models/         # Model architectures (Policy, Value networks)
│   ├── training/       # PPO training loop and RL components
│   ├── evaluation/     # Performance evaluation and metrics
│   ├── experiments/    # Experiment management and analysis
│   └── utils/          # Utility functions and configuration
├── tests/              # Test suite (unit, integration, property-based)
├── configs/            # Configuration files
├── google-code-golf-2025/  # Task data
└── requirements.txt    # Python dependencies
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-rl-code-golf
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training

Run training with default configuration:
```bash
python -m src.training.train --config configs/default_config.yaml
```

### Evaluation

Evaluate a trained model:
```bash
python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt
```

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test categories:
```bash
pytest -m unit          # Unit tests only
pytest -m property      # Property-based tests only
pytest -m integration   # Integration tests only
```

## Configuration

Edit `configs/default_config.yaml` to customize:
- Model architecture and hyperparameters
- PPO training parameters
- Reward function weights
- Curriculum learning settings
- Logging and checkpointing

## Development

This project follows a spec-driven development approach. See `.kiro/specs/llm-rl-code-golf/` for:
- `requirements.md`: Detailed requirements
- `design.md`: System design and architecture
- `tasks.md`: Implementation task list

## License

MIT License
