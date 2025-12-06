#!/usr/bin/env python
"""
Launcher script for training that handles path setup.
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the training script
if __name__ == "__main__":
    # Import after path is set
    from train import main
    main()
