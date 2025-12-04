"""
Property-based tests for model architecture components.

Feature: llm-rl-code-golf
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import tempfile
import os
import shutil

from src.models.config import ModelConfig
from src.models.code_llm import CodeLLM


# Custom strategies for generating test data
@st.composite
def python_code_strings(draw):
    """Generate valid Python code strings."""
    # Generate simple Python code patterns
    code_patterns = [
        "def solve(grid):\n    return grid",
        "x = 42\nprint(x)",
        "for i in range(10):\n    pass",
        "if True:\n    x = 1\nelse:\n    x = 2",
        "result = [x for x in range(5)]",
        "def func():\n    return None",
        "# Comment\nx = 1",
        "a, b = 1, 2",
        "import sys",
        "from typing import List",
    ]
    
    # Also generate random strings with Python-like characters
    if draw(st.booleans()):
        return draw(st.sampled_from(code_patterns))
    else:
        # Generate strings with common Python characters
        return draw(st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='_()[]{}:,.\n '
            ),
            min_size=1,
            max_size=100
        ))


@st.composite
def model_configs(draw, use_small_model=True):
    """Generate model configurations for testing."""
    # Use a small model for testing
    model_name = "gpt2" if use_small_model else draw(st.sampled_from([
        "gpt2",
        "distilgpt2"
    ]))
    
    return ModelConfig(
        model_name=model_name,
        max_length=draw(st.integers(min_value=128, max_value=512)),
        temperature=draw(st.floats(min_value=0.1, max_value=2.0)),
        top_p=draw(st.floats(min_value=0.1, max_value=1.0)),
        top_k=draw(st.integers(min_value=1, max_value=100)),
        quantization=None,  # Skip quantization for faster tests
        trainable_layers=[],
        device="cpu",  # Use CPU for tests
        torch_dtype="float32",
    )


class TestTokenizationProperties:
    """Test tokenization properties."""
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create a model instance for testing."""
        config = ModelConfig(
            model_name="gpt2",
            max_length=512,
            device="cpu",
            torch_dtype="float32",
            quantization=None,
        )
        return CodeLLM(config)
    
    @given(code=python_code_strings())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_6_tokenization_reversibility(self, model, code):
        """
        Property 6: Tokenization reversibility
        
        For any valid Python code string, encoding then decoding should 
        produce the original string (or a semantically equivalent version).
        
        **Feature: llm-rl-code-golf, Property 6: Tokenization reversibility**
        **Validates: Requirements 2.2**
        """
        # Skip empty strings
        assume(len(code.strip()) > 0)
        
        # Encode the code
        tokens = model.tokenizer.encode(code, add_special_tokens=False)
        
        # Decode back to string
        decoded = model.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # The decoded string should match the original
        # Note: Some whitespace normalization may occur
        assert decoded.strip() == code.strip() or decoded == code, \
            f"Tokenization not reversible:\nOriginal: {repr(code)}\nDecoded: {repr(decoded)}"


class TestQuantizationProperties:
    """Test quantization properties."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantization requires CUDA")
    @given(quantization=st.sampled_from(["4bit", "8bit"]))
    @settings(max_examples=10, deadline=None)
    def test_property_7_quantization_memory_reduction(self, quantization):
        """
        Property 7: Quantization memory reduction
        
        For any model, quantized versions should use strictly less memory 
        than full precision versions.
        
        **Feature: llm-rl-code-golf, Property 7: Quantization memory reduction**
        **Validates: Requirements 2.3**
        """
        # Create full precision model
        config_fp = ModelConfig(
            model_name="gpt2",
            max_length=256,
            device="cuda",
            torch_dtype="float32",
            quantization=None,
        )
        model_fp = CodeLLM(config_fp)
        
        # Get memory usage for full precision
        memory_fp = sum(
            p.element_size() * p.nelement() 
            for p in model_fp.model.parameters()
        )
        
        # Clean up
        del model_fp
        torch.cuda.empty_cache()
        
        # Create quantized model
        config_quant = ModelConfig(
            model_name="gpt2",
            max_length=256,
            device="cuda",
            torch_dtype="float32",
            quantization=quantization,
        )
        model_quant = CodeLLM(config_quant)
        
        # Get memory usage for quantized model
        memory_quant = sum(
            p.element_size() * p.nelement() 
            for p in model_quant.model.parameters()
        )
        
        # Quantized should use less memory
        assert memory_quant < memory_fp, \
            f"Quantized model ({memory_quant} bytes) should use less memory than full precision ({memory_fp} bytes)"
        
        # Clean up
        del model_quant
        torch.cuda.empty_cache()


class TestLayerFreezingProperties:
    """Test layer freezing properties."""
    
    @given(
        trainable_pattern=st.sampled_from([
            ["lm_head"],
            ["transformer.h.11"],
            ["transformer.h.10", "transformer.h.11"],
            [],
        ])
    )
    @settings(max_examples=20, deadline=None)
    def test_property_8_layer_freezing_correctness(self, trainable_pattern):
        """
        Property 8: Layer freezing correctness
        
        For any layer configuration, frozen layers should have requires_grad=False 
        and trainable layers should have requires_grad=True.
        
        **Feature: llm-rl-code-golf, Property 8: Layer freezing correctness**
        **Validates: Requirements 2.4**
        """
        config = ModelConfig(
            model_name="gpt2",
            max_length=256,
            device="cpu",
            torch_dtype="float32",
            quantization=None,
            trainable_layers=trainable_pattern,
        )
        model = CodeLLM(config)
        
        # Check that trainable layers have requires_grad=True
        for name, param in model.model.named_parameters():
            should_be_trainable = any(
                pattern in name for pattern in trainable_pattern
            )
            
            if should_be_trainable:
                assert param.requires_grad, \
                    f"Layer {name} should be trainable but requires_grad={param.requires_grad}"
            else:
                assert not param.requires_grad, \
                    f"Layer {name} should be frozen but requires_grad={param.requires_grad}"


class TestCheckpointProperties:
    """Test checkpoint save/load properties."""
    
    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=None)
    def test_property_9_checkpoint_round_trip(self, seed):
        """
        Property 9: Checkpoint round trip
        
        For any model state, saving to checkpoint and loading should 
        restore identical model parameters.
        
        **Feature: llm-rl-code-golf, Property 9: Checkpoint round trip**
        **Validates: Requirements 2.5**
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Create a model
        config = ModelConfig(
            model_name="gpt2",
            max_length=256,
            device="cpu",
            torch_dtype="float32",
            quantization=None,
        )
        model1 = CodeLLM(config)
        
        # Get original parameters
        original_params = {
            name: param.clone().detach()
            for name, param in model1.model.named_parameters()
        }
        
        # Save checkpoint
        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(temp_dir, "checkpoint")
            model1.save_checkpoint(checkpoint_path)
            
            # Load checkpoint
            model2 = CodeLLM.load_checkpoint(checkpoint_path)
            
            # Compare parameters
            for name, param in model2.model.named_parameters():
                assert name in original_params, f"Parameter {name} not found in original model"
                
                original = original_params[name]
                loaded = param.detach()
                
                # Check shapes match
                assert original.shape == loaded.shape, \
                    f"Shape mismatch for {name}: {original.shape} vs {loaded.shape}"
                
                # Check values match (with small tolerance for numerical precision)
                assert torch.allclose(original, loaded, rtol=1e-5, atol=1e-7), \
                    f"Parameter {name} not restored correctly after checkpoint round trip"
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
