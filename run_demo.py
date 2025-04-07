"""
Standalone script to run the custom_einops demo.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the demo
from custom_einops.demo import run_examples

if __name__ == "__main__":
    run_examples() 