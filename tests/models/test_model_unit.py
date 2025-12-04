"""
Unit tests for model initialization and configuration.

Tests model loading with different configurations, quantization,
and checkpoint save/load functionality.
"""

import pytest
import torch
import tempfile
import os
import shutil
import json

from src.models.config import ModelConfig
from src.models.code_llm import CodeLLM


class TestModelInitialization:
    """Test model initialization with different configurations."""
    
    def test_model_loading_with_default_config(self):
        """Test loading model with default configuration."""
        config = ModelConfig(
            model_name="gpt2",
            max_length=256,
            device="cpu",
            torch_dtype="float32",
        )
        model = CodeLLM(config)
        
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.config == config
    
    def test_model_loading_with_custom_temperature(self):
        """Test model with custom temperature setting."""
        config = ModelConfig(
            model_name="gpt2",
            max_length=256,
            temperature=0.8,
            device="cpu",
        )
        model = CodeLLM(config)
        
        assert model.config.temperature == 0.8
    
    def test_model_loading_with_custom_top_p(self):
        """Test model with custom top_p setting."""
        config = ModelConfig(
            model_name="gpt2",
            max_length=256,
            top_p=0.95,
            device="cpu",
        )
        model = CodeLLM(config)
        
        assert model.config.top_p == 0.95
    
    def test_model_loading_with_custom_max_length(self):
        """Test model with custom max_length."""
        config = ModelConfig(
            model_name="gpt2",
            max_length=512,
            device="cpu",
        )
        model = CodeLLM(config)
        
        assert model.config.max_length == 512
    
    def test_tokenizer_configuration(self):
        """Test that tokenizer is properly configured."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
        )
        model = CodeLLM(config)
        
        # Check tokenizer has pad token
        assert model.tokenizer.pad_token is not None
        
        # Check padding side is set for batch generation
        assert model.tokenizer.padding_side == "left"
    
    def test_model_device_placement_cpu(self):
        """Test model is placed on CPU when specified."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
        )
        model = CodeLLM(config)
        
        # Check model is on CPU
        assert next(model.model.parameters()).device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_device_placement_cuda(self):
        """Test model is placed on CUDA when available."""
        config = ModelConfig(
            model_name="gpt2",
            device="cuda",
            quantization=None,
        )
        model = CodeLLM(config)
        
        # Check model is on CUDA
        assert next(model.model.parameters()).device.type == "cuda"
    
    def test_model_parameter_counts(self):
        """Test getting parameter counts."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
            trainable_layers=[],
        )
        model = CodeLLM(config)
        
        total_params = model.get_total_parameters()
        trainable_params = model.get_trainable_parameters()
        
        assert total_params > 0
        assert trainable_params == 0  # All frozen
    
    def test_model_with_trainable_layers(self):
        """Test model with specific trainable layers."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
            trainable_layers=["transformer.h.11"],  # Last transformer layer
        )
        model = CodeLLM(config)
        
        trainable_params = model.get_trainable_parameters()
        total_params = model.get_total_parameters()
        
        assert trainable_params > 0
        assert trainable_params < total_params


class TestQuantization:
    """Test quantization with various bit widths."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantization requires CUDA")
    def test_4bit_quantization(self):
        """Test loading model with 4-bit quantization."""
        config = ModelConfig(
            model_name="gpt2",
            device="cuda",
            quantization="4bit",
        )
        model = CodeLLM(config)
        
        assert model.model is not None
        assert model.config.quantization == "4bit"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantization requires CUDA")
    def test_8bit_quantization(self):
        """Test loading model with 8-bit quantization."""
        config = ModelConfig(
            model_name="gpt2",
            device="cuda",
            quantization="8bit",
        )
        model = CodeLLM(config)
        
        assert model.model is not None
        assert model.config.quantization == "8bit"
    
    def test_no_quantization(self):
        """Test loading model without quantization."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
            quantization=None,
        )
        model = CodeLLM(config)
        
        assert model.model is not None
        assert model.config.quantization is None


class TestCheckpointSaveLoad:
    """Test checkpoint save and load functionality."""
    
    def test_checkpoint_save_creates_files(self):
        """Test that saving checkpoint creates necessary files."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
        )
        model = CodeLLM(config)
        
        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(temp_dir, "checkpoint")
            model.save_checkpoint(checkpoint_path)
            
            # Check that checkpoint directory exists
            assert os.path.exists(checkpoint_path)
            
            # Check that config file exists
            config_file = os.path.join(checkpoint_path, "model_config.json")
            assert os.path.exists(config_file)
            
            # Check that model files exist
            assert os.path.exists(os.path.join(checkpoint_path, "config.json"))
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_checkpoint_save_includes_config(self):
        """Test that saved checkpoint includes model configuration."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
            temperature=0.7,
            max_length=256,
        )
        model = CodeLLM(config)
        
        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(temp_dir, "checkpoint")
            model.save_checkpoint(checkpoint_path)
            
            # Load config file
            config_file = os.path.join(checkpoint_path, "model_config.json")
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            
            # Check config values
            assert saved_config["temperature"] == 0.7
            assert saved_config["max_length"] == 256
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_checkpoint_load_restores_model(self):
        """Test that loading checkpoint restores model."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
        )
        model1 = CodeLLM(config)
        
        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(temp_dir, "checkpoint")
            model1.save_checkpoint(checkpoint_path)
            
            # Load checkpoint
            model2 = CodeLLM.load_checkpoint(checkpoint_path)
            
            assert model2.model is not None
            assert model2.tokenizer is not None
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_checkpoint_load_restores_config(self):
        """Test that loading checkpoint restores configuration."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
            temperature=0.9,
            top_p=0.85,
            max_length=384,
        )
        model1 = CodeLLM(config)
        
        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(temp_dir, "checkpoint")
            model1.save_checkpoint(checkpoint_path)
            
            # Load checkpoint
            model2 = CodeLLM.load_checkpoint(checkpoint_path)
            
            # Check config values
            assert model2.config.temperature == 0.9
            assert model2.config.top_p == 0.85
            assert model2.config.max_length == 384
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_checkpoint_with_different_model_sizes(self):
        """Test checkpoint save/load with different model configurations."""
        # Test with small model
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
        )
        model = CodeLLM(config)
        
        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(temp_dir, "checkpoint")
            model.save_checkpoint(checkpoint_path)
            
            # Load and verify
            loaded_model = CodeLLM.load_checkpoint(checkpoint_path)
            assert loaded_model.model is not None
            
            # Compare parameter counts
            assert (loaded_model.get_total_parameters() == 
                    model.get_total_parameters())
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestModelGeneration:
    """Test basic model generation functionality."""
    
    def test_model_can_generate_text(self):
        """Test that model can generate text from a prompt."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
            temperature=0.7,
        )
        model = CodeLLM(config)
        
        prompt = "def hello():"
        output = model.generate(prompt, max_new_tokens=10)
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_model_generation_with_temperature_zero(self):
        """Test deterministic generation with temperature=0."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
        )
        model = CodeLLM(config)
        
        prompt = "def add(a, b):"
        output1 = model.generate(prompt, temperature=0.0, max_new_tokens=10)
        output2 = model.generate(prompt, temperature=0.0, max_new_tokens=10)
        
        # With temperature=0, outputs should be identical
        assert output1 == output2
    
    def test_model_generation_respects_max_tokens(self):
        """Test that generation respects max_new_tokens parameter."""
        config = ModelConfig(
            model_name="gpt2",
            device="cpu",
        )
        model = CodeLLM(config)
        
        prompt = "def test():"
        output = model.generate(prompt, max_new_tokens=5)
        
        # Output should be relatively short
        tokens = model.tokenizer.encode(output)
        assert len(tokens) <= 10  # Allow some margin


class TestErrorHandling:
    """Test error handling in model initialization."""
    
    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises appropriate error."""
        config = ModelConfig(
            model_name="nonexistent-model-12345",
            device="cpu",
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            CodeLLM(config)
        
        assert "Failed to load model" in str(exc_info.value)
    
    def test_model_handles_cuda_unavailable(self):
        """Test that model falls back to CPU when CUDA unavailable."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test fallback")
        
        config = ModelConfig(
            model_name="gpt2",
            device="cuda",
        )
        model = CodeLLM(config)
        
        # Should fall back to CPU
        assert model.config.device == "cpu"
