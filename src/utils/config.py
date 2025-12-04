"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "codellama/CodeLlama-7b-Python-hf"
    max_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    quantization: Optional[str] = "4bit"
    lora: Dict[str, Any] = field(default_factory=dict)
    freeze_base: bool = True
    trainable_layers: list = field(default_factory=list)


@dataclass
class PPOConfig:
    """PPO training configuration."""
    learning_rate: float = 1.0e-5
    batch_size: int = 8
    num_epochs: int = 100
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    num_trajectories_per_update: int = 32


@dataclass
class RewardConfig:
    """Reward calculation configuration."""
    max_reward: float = 10.0
    test_pass_weight: float = 1.0
    code_length_weight: float = 0.1
    syntax_error_penalty: float = -5.0
    runtime_error_penalty: float = -2.0
    timeout_penalty: float = -3.0
    normalize: bool = True


@dataclass
class SandboxConfig:
    """Code execution sandbox configuration."""
    timeout: float = 5.0
    memory_limit: int = 512
    allow_imports: list = field(default_factory=list)


@dataclass
class DataConfig:
    """Data loading configuration."""
    task_dir: str = "google-code-golf-2025"
    num_tasks: int = 400
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = True
    initial_difficulty: str = "easy"
    progression_threshold: float = 0.7
    difficulty_levels: list = field(default_factory=lambda: ["easy", "medium", "hard"])


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    save_dir: str = "checkpoints"
    save_interval: int = 1000
    keep_best_n: int = 3
    early_stopping_patience: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "llm-rl-code-golf"
    log_interval: int = 10


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "baseline"
    seed: int = 42
    device: str = "cuda"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse nested configs
    model_config = ModelConfig(**config_dict.get('model', {}))
    ppo_config = PPOConfig(**config_dict.get('ppo', {}))
    reward_config = RewardConfig(**config_dict.get('reward', {}))
    sandbox_config = SandboxConfig(**config_dict.get('sandbox', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    curriculum_config = CurriculumConfig(**config_dict.get('curriculum', {}))
    checkpoint_config = CheckpointConfig(**config_dict.get('checkpoint', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))
    
    experiment_dict = config_dict.get('experiment', {})
    
    return ExperimentConfig(
        name=experiment_dict.get('name', 'baseline'),
        seed=experiment_dict.get('seed', 42),
        device=experiment_dict.get('device', 'cuda'),
        model=model_config,
        ppo=ppo_config,
        reward=reward_config,
        sandbox=sandbox_config,
        data=data_config,
        curriculum=curriculum_config,
        checkpoint=checkpoint_config,
        logging=logging_config
    )


def save_config(config: ExperimentConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: ExperimentConfig object
        save_path: Path to save YAML file
    """
    config_dict = {
        'experiment': {
            'name': config.name,
            'seed': config.seed,
            'device': config.device
        },
        'model': config.model.__dict__,
        'ppo': config.ppo.__dict__,
        'reward': config.reward.__dict__,
        'sandbox': config.sandbox.__dict__,
        'data': config.data.__dict__,
        'curriculum': config.curriculum.__dict__,
        'checkpoint': config.checkpoint.__dict__,
        'logging': config.logging.__dict__
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
