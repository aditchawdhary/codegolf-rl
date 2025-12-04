"""
Checkpoint manager for saving and loading training state.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """Manages model checkpoints and training state."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", keep_last_n: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        policy,
        value,
        policy_optimizer,
        value_optimizer,
        step: int,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        is_best: bool = False
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            policy: Policy network
            value: Value network
            policy_optimizer: Policy optimizer
            value_optimizer: Value optimizer
            step: Training step
            metrics: Training metrics
            config: Training configuration
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save policy model
        policy.save_checkpoint(str(checkpoint_path / "policy"))
        
        # Save value network
        value.save(str(checkpoint_path))
        
        # Save optimizer states
        torch.save(
            policy_optimizer.state_dict(),
            checkpoint_path / "policy_optimizer.pt"
        )
        torch.save(
            value_optimizer.state_dict(),
            checkpoint_path / "value_optimizer.pt"
        )
        
        # Save training state
        training_state = {
            "step": step,
            "metrics": metrics,
            "config": config,
            "timestamp": timestamp,
            "is_best": is_best
        }
        
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Track checkpoint
        self.checkpoints.append({
            "path": str(checkpoint_path),
            "step": step,
            "timestamp": timestamp,
            "is_best": is_best
        })
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint"
            if best_path.exists():
                import shutil
                shutil.rmtree(best_path)
            import shutil
            shutil.copytree(checkpoint_path, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        policy,
        value,
        policy_optimizer=None,
        value_optimizer=None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            policy: Policy network to load into
            value: Value network to load into
            policy_optimizer: Optional policy optimizer to load state
            value_optimizer: Optional value optimizer to load state
            
        Returns:
            Training state dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load policy
        from src.models import CodeLLM
        policy_loaded = CodeLLM.load_checkpoint(str(checkpoint_path / "policy"))
        policy.model = policy_loaded.model
        policy.tokenizer = policy_loaded.tokenizer
        
        # Load value network
        value.load(str(checkpoint_path))
        
        # Load optimizer states if provided
        if policy_optimizer is not None:
            policy_optimizer.load_state_dict(
                torch.load(checkpoint_path / "policy_optimizer.pt")
            )
        
        if value_optimizer is not None:
            value_optimizer.load_state_dict(
                torch.load(checkpoint_path / "value_optimizer.pt")
            )
        
        # Load training state
        with open(checkpoint_path / "training_state.json", 'r') as f:
            training_state = json.load(f)
        
        return training_state
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        best_path = self.checkpoint_dir / "best_checkpoint"
        if best_path.exists():
            return str(best_path)
        
        # Find best from tracked checkpoints
        best_checkpoint = None
        for ckpt in self.checkpoints:
            if ckpt.get("is_best", False):
                best_checkpoint = ckpt["path"]
                break
        
        return best_checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoints:
            # Scan directory
            self._scan_checkpoints()
        
        if self.checkpoints:
            # Sort by step
            sorted_ckpts = sorted(self.checkpoints, key=lambda x: x["step"], reverse=True)
            return sorted_ckpts[0]["path"]
        
        return None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint info dictionaries
        """
        if not self.checkpoints:
            self._scan_checkpoints()
        
        return sorted(self.checkpoints, key=lambda x: x["step"], reverse=True)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self.checkpoints) <= self.keep_last_n:
            return
        
        # Sort by step
        sorted_ckpts = sorted(self.checkpoints, key=lambda x: x["step"], reverse=True)
        
        # Keep best checkpoint and last N
        to_keep = set()
        for ckpt in sorted_ckpts[:self.keep_last_n]:
            to_keep.add(ckpt["path"])
        
        # Always keep best
        for ckpt in self.checkpoints:
            if ckpt.get("is_best", False):
                to_keep.add(ckpt["path"])
        
        # Remove old checkpoints
        for ckpt in self.checkpoints[:]:
            if ckpt["path"] not in to_keep:
                checkpoint_path = Path(ckpt["path"])
                if checkpoint_path.exists():
                    import shutil
                    shutil.rmtree(checkpoint_path)
                self.checkpoints.remove(ckpt)
    
    def _scan_checkpoints(self):
        """Scan checkpoint directory for existing checkpoints."""
        if not self.checkpoint_dir.exists():
            return
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_"):
                state_file = item / "training_state.json"
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    
                    self.checkpoints.append({
                        "path": str(item),
                        "step": state.get("step", 0),
                        "timestamp": state.get("timestamp", ""),
                        "is_best": state.get("is_best", False)
                    })
