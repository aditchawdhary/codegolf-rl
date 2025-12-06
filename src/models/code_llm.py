"""
Base LLM class for code generation.
"""

import torch
from typing import Optional, Dict, Any, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from .config import ModelConfig


class CodeLLM:
    """Base LLM for code generation."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the code LLM.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        self._configure_tokenizer()
    
    def _load_model(self):
        """Load the model with appropriate configuration."""
        # Determine device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.config.device = "cpu"
        elif self.config.device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.config.device = "cpu"
        
        # Configure quantization
        quantization_config = None
        if self.config.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "use_cache": self.config.use_cache,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = self._get_torch_dtype()
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Move to device if not using quantization
            if not quantization_config and self.config.device != "cpu":
                self.model = self.model.to(self.config.device)
            
            # Configure trainable layers
            self._configure_trainable_layers()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.config.model_name}: {e}")
    
    def _configure_tokenizer(self):
        """Configure the tokenizer for Python code."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure for code generation
            self.tokenizer.padding_side = "left"  # For batch generation
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.torch_dtype, torch.float16)
    
    def _configure_trainable_layers(self):
        """Configure which layers are trainable."""
        if not self.config.trainable_layers:
            # By default, freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Freeze all first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze specified layers
            for name, param in self.model.named_parameters():
                for trainable_layer in self.config.trainable_layers:
                    if trainable_layer in name:
                        param.requires_grad = True
                        break
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate code from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        max_new_tokens = max_new_tokens if max_new_tokens is not None else 512
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def get_logprobs(self, prompt: str, completion: str, requires_grad: bool = False) -> torch.Tensor:
        """
        Get log probabilities for a completion given a prompt.
        
        Args:
            prompt: Input prompt
            completion: Completion text
            requires_grad: Whether to track gradients (needed for training)
            
        Returns:
            Log probabilities tensor
        """
        # Tokenize prompt and completion
        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt")
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get model outputs (with or without gradients)
        if requires_grad:
            outputs = self.model(**inputs)
            logits = outputs.logits
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
        
        # Get log probabilities for completion tokens
        prompt_length = prompt_inputs["input_ids"].shape[1]
        completion_logits = logits[0, prompt_length-1:-1, :]
        completion_tokens = inputs["input_ids"][0, prompt_length:]
        
        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs[range(len(completion_tokens)), completion_tokens]
        
        return token_log_probs
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional forward pass arguments
            
        Returns:
            Model outputs dictionary
        """
        outputs = self.model(input_ids=input_ids, **kwargs)
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
        }
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config
        import json
        config_path = f"{path}/model_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> "CodeLLM":
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Loaded CodeLLM instance
        """
        import json
        
        # Load config
        config_path = f"{path}/model_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig.from_dict(config_dict)
        config.model_name = path  # Load from checkpoint path
        
        return cls(config)
    
    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Get count of total parameters."""
        return sum(p.numel() for p in self.model.parameters())
