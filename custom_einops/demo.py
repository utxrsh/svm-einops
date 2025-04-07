"""
Demonstration of the custom einops implementation.

This script showcases the core functionality of our einops implementation
by running examples from the assignment and comparing with the original einops.
"""

import numpy as np
import time
from einops import rearrange as original_rearrange, repeat as original_repeat
from .einops import rearrange, repeat


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def compare_outputs(name, custom_result, original_result):
    """Compare the outputs from custom and original einops."""
    print(f"\n{name}:")
    print(f"  Custom shape:   {custom_result.shape}")
    print(f"  Original shape: {original_result.shape}")
    print(f"  Shapes match:   {custom_result.shape == original_result.shape}")
    print(f"  Values match:   {np.allclose(custom_result, original_result)}")


def compare_speed(name, custom_fn, original_fn, iterations=100):
    """Compare the speed of custom and original implementations."""
    # Warm-up
    for _ in range(5):
        custom_fn()
        original_fn()
    
    # Time custom implementation
    start = time.time()
    for _ in range(iterations):
        custom_fn()
    custom_time = time.time() - start
    
    # Time original implementation
    start = time.time()
    for _ in range(iterations):
        original_fn()
    original_time = time.time() - start
    
    print(f"\n{name} Speed Comparison ({iterations} iterations):")
    print(f"  Custom:   {custom_time:.4f} seconds")
    print(f"  Original: {original_time:.4f} seconds")
    print(f"  Ratio:    {custom_time / original_time:.2f}x")


def run_examples():
    """Run examples from the assignment."""
    print_header("Basic Einops Examples")
    
    # Example 1: Transpose
    x = np.random.rand(3, 4)
    custom_result = rearrange(x, 'h w -> w h')
    original_result = original_rearrange(x, 'h w -> w h')
    compare_outputs("Transpose", custom_result, original_result)
    compare_speed("Transpose", 
                  lambda: rearrange(x, 'h w -> w h'),
                  lambda: original_rearrange(x, 'h w -> w h'))
    
    # Example 2: Split an axis
    x = np.random.rand(12, 10)
    custom_result = rearrange(x, '(h w) c -> h w c', h=3)
    original_result = original_rearrange(x, '(h w) c -> h w c', h=3)
    compare_outputs("Split Axis", custom_result, original_result)
    compare_speed("Split Axis", 
                  lambda: rearrange(x, '(h w) c -> h w c', h=3),
                  lambda: original_rearrange(x, '(h w) c -> h w c', h=3))
    
    # Example 3: Merge axes
    x = np.random.rand(3, 4, 5)
    custom_result = rearrange(x, 'a b c -> (a b) c')
    original_result = original_rearrange(x, 'a b c -> (a b) c')
    compare_outputs("Merge Axes", custom_result, original_result)
    compare_speed("Merge Axes", 
                  lambda: rearrange(x, 'a b c -> (a b) c'),
                  lambda: original_rearrange(x, 'a b c -> (a b) c'))
    
    # Example 4: Repeat an axis
    x = np.random.rand(3, 1, 5)
    custom_result = repeat(x, 'a 1 c -> a b c', b=4)
    original_result = original_repeat(x, 'a 1 c -> a b c', b=4)
    compare_outputs("Repeat Axis", custom_result, original_result)
    compare_speed("Repeat Axis", 
                  lambda: repeat(x, 'a 1 c -> a b c', b=4),
                  lambda: original_repeat(x, 'a 1 c -> a b c', b=4))
    
    # Example 5: Handle batch dimensions
    x = np.random.rand(2, 3, 4, 5)
    custom_result = rearrange(x, '... h w -> ... (h w)')
    original_result = original_rearrange(x, '... h w -> ... (h w)')
    compare_outputs("Batch Dimensions", custom_result, original_result)
    compare_speed("Batch Dimensions", 
                  lambda: rearrange(x, '... h w -> ... (h w)'),
                  lambda: original_rearrange(x, '... h w -> ... (h w)'))
    
    print_header("Complex Einops Examples")
    
    # Example 6: Complex reshape with multiple operations
    x = np.random.rand(10, 20, 30, 3)
    custom_result = rearrange(x, '(a b) (c d) e f -> a b (c d e) f', a=2, c=5)
    original_result = original_rearrange(x, '(a b) (c d) e f -> a b (c d e) f', a=2, c=5)
    compare_outputs("Complex Pattern", custom_result, original_result)
    compare_speed("Complex Pattern", 
                  lambda: rearrange(x, '(a b) (c d) e f -> a b (c d e) f', a=2, c=5),
                  lambda: original_rearrange(x, '(a b) (c d) e f -> a b (c d e) f', a=2, c=5))


if __name__ == "__main__":
    run_examples() 