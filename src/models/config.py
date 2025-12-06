"""
Model configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 8  # Rank of the low-rank matrices
    lora_alpha: int = 16  # Scaling factor
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"  # "none", "all", or "lora_only"


@dataclass
class ModelConfig:
    """Configuration for the base LLM."""
    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    max_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    quantization: Optional[str] = "8bit"
    lora_config: Optional[LoRAConfig] = None
    trainable_layers: List[str] = field(default_factory=list)
    device: str = "cuda"
    torch_dtype: str = "float16"  # "float32", "float16", "bfloat16"
    trust_remote_code: bool = True
    use_cache: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "quantization": self.quantization,
            "trainable_layers": self.trainable_layers,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "use_cache": self.use_cache,
        }
        
        if self.lora_config:
            config_dict["lora_config"] = {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": self.lora_config.target_modules,
                "lora_dropout": self.lora_config.lora_dropout,
                "bias": self.lora_config.bias,
            }
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        lora_dict = config_dict.pop("lora_config", None)
        lora_config = None
        if lora_dict:
            lora_config = LoRAConfig(**lora_dict)
        
        return cls(lora_config=lora_config, **config_dict)
