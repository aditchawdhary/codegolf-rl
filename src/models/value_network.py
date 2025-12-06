"""
Value network for RL training.
"""

import torch
import torch.nn as nn
from typing import Optional
from .code_llm import CodeLLM
from .config import ModelConfig


class ValueNetwork(nn.Module):
    """Value function estimator for RL training."""
    
    def __init__(self, base_model: CodeLLM, hidden_size: int = 768):
        """
        Initialize value network.
        
        Args:
            base_model: Base LLM to extract features from
            hidden_size: Hidden layer size
        """
        print(f"[ValueNetwork.__init__] Starting initialization...")
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        
        # Value head - maps from model hidden size to scalar value
        # Get the model's hidden size
        print(f"[ValueNetwork.__init__] Getting model hidden size...")
        model_hidden_size = base_model.model.config.hidden_size
        print(f"[ValueNetwork.__init__] Model hidden size: {model_hidden_size}")
        
        print(f"[ValueNetwork.__init__] Creating value head layers...")
        self.value_head = nn.Sequential(
            nn.Linear(model_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Move value head to same device and dtype as base model
        print(f"[ValueNetwork.__init__] Moving value head to device...")
        device = base_model.model.device
        dtype = base_model.model.dtype
        print(f"[ValueNetwork.__init__] Device: {device}, dtype: {dtype}")
        self.value_head = self.value_head.to(device=device, dtype=dtype)
        print(f"[ValueNetwork.__init__] Value network initialization complete!")
    
    def estimate_value(self, state: str) -> torch.Tensor:
        """
        Estimate value of a state.
        
        Args:
            state: Input state (prompt)
            
        Returns:
            Estimated value (scalar tensor)
        """
        # Tokenize input
        inputs = self.base_model.tokenizer(
            state,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.base_model.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.base_model.model.device) for k, v in inputs.items()}
        
        # Get hidden states from base model
        with torch.no_grad():
            outputs = self.base_model.model(
                **inputs,
                output_hidden_states=True
            )
            
            # Use last hidden state of the last token
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        # Pass through value head
        value = self.value_head(last_hidden_state)
        
        return value.squeeze()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Value estimates
        """
        # Get hidden states from base model
        with torch.no_grad():
            outputs = self.base_model.model(
                input_ids=input_ids,
                output_hidden_states=True
            )
            
            # Use last hidden state of the last token
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        # Pass through value head
        value = self.value_head(last_hidden_state)
        
        return value.squeeze()
    
    def train_step(
        self,
        states: list,
        target_values: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Perform a training step on the value network.
        
        Args:
            states: List of state strings
            target_values: Target value estimates
            optimizer: Optimizer for value network
            
        Returns:
            Loss value
        """
        # Estimate values for all states
        predicted_values = []
        for state in states:
            value = self.estimate_value(state)
            predicted_values.append(value)
        
        predicted_values = torch.stack(predicted_values)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(predicted_values, target_values)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def save(self, path: str):
        """
        Save value network.
        
        Args:
            path: Path to save to
        """
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")
    
    def load(self, path: str):
        """
        Load value network.
        
        Args:
            path: Path to load from
        """
        self.value_head.load_state_dict(
            torch.load(f"{path}/value_head.pt", map_location=self.base_model.model.device)
        )
