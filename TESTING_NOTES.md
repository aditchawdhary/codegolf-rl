# Testing Notes

## macOS PyTorch Generation Issue

### Problem
Tests that involve model generation (`model.generate()`) fail with a **bus error** on macOS. This is a known compatibility issue between PyTorch, Transformers, and macOS (especially Apple Silicon).

### Root Cause
- **Not a memory issue**: System has 32GB RAM with 15GB available
- **Not our code**: Direct calls to `transformers.generate()` also fail
- **Platform-specific**: Bus error occurs in PyTorch's C++ backend during generation

### Affected Tests
The following tests are skipped on macOS due to this issue:
- `test_property_11_sampling_diversity` - Tests sampling diversity with temperature > 0
- `test_property_14_network_update_consistency` - Tests policy/value network updates
- `test_policy_can_sample_action` - Tests basic action sampling
- `test_policy_get_entropy` - Tests entropy estimation

### Workarounds
1. **Run on Linux/CUDA**: These tests should work on Linux systems or with CUDA
2. **Validate during training**: The properties are validated during actual training runs
3. **Use mocked generation**: For CI/CD, consider mocking the generation calls

### Tests That Work
The following tests pass successfully:
- Model initialization and configuration
- Tokenization (encode/decode without generation)
- Log probability computation
- Value network operations
- Parameter freezing and checkpointing

### Recommendations
- Run full test suite on a Linux machine or CI/CD pipeline
- The skipped tests validate important properties that should be checked before deployment
- Consider using Docker with Linux for local testing if needed

### Memory Analysis Results
```
Total Memory: 32.00 GB
Available Memory: 15.30 GB
Model Memory Usage: ~113 MB
Parameter Memory: ~475 MB
```

The system has sufficient resources; the issue is a PyTorch/macOS compatibility problem, not resource constraints.
