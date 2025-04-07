# Custom Einops Implementation

A concise, functional implementation of the core einops operations, focusing on `rearrange` and `repeat` functionality.

## Overview

This implementation provides a clean, readable version of the einops pattern-based tensor manipulation system. It works with NumPy arrays and supports the following operations:

- Reshaping (merging and splitting dimensions)
- Transposition (reordering dimensions)
- Adding new dimensions
- Repeating elements along dimensions
- Complex pattern handling with anonymous dimensions and numeric literals

## Usage

### Basic Operations

```python
import numpy as np
from custom_einops import rearrange, repeat

# Transpose
x = np.random.rand(3, 4)
result = rearrange(x, 'h w -> w h')

# Split an axis
x = np.random.rand(12, 10)
result = rearrange(x, '(h w) c -> h w c', h=3)

# Merge axes
x = np.random.rand(3, 4, 5)
result = rearrange(x, 'a b c -> (a b) c')

# Repeat an axis
x = np.random.rand(3, 1, 5)
result = repeat(x, 'a 1 c -> a b c', b=4)

# Handle batch dimensions
x = np.random.rand(2, 3, 4, 5)
result = rearrange(x, '... h w -> ... (h w)')

# Multiple repetitions
x = np.array([1, 2])
result = rearrange(x, 'a -> 2 a 3')  # Shape: (2, 2, 3)

# Advanced pattern with composition and repetition
x = np.array([1, 2])
result = rearrange(x, 'a -> (a 3)')  # Shape: (6,)
```

### Pattern Syntax

The pattern syntax follows these rules:

- Dimensions are named with identifiers (e.g., `batch`, `h`, `w`)
- The arrow `->` separates input and output patterns
- Parentheses `(...)` are used for composing/decomposing dimensions
- Ellipsis `...` represents any number of dimensions
- Anonymous dimensions `1` represent singleton dimensions
- Numeric literals in the output represent repetition (e.g., `a b -> a b 3`)

## Implementation Details

This implementation focuses on clarity and correctness while maintaining a functional approach. Below are key design decisions and implementation details:

### Design Decisions

1. **Functional Over Object-Oriented**: We chose a functional approach rather than class-based implementation for simplicity and to make the code flow easier to follow.

2. **Strong Typing**: Comprehensive type hints are used throughout the code for better documentation and to catch type-related errors early.

3. **Staged Operation Pipeline**: Rather than trying to perform all operations in one step, we break the process into clear stages (reshape, transpose, repeat, reshape again).

4. **Separation of Concerns**: Parsing, validation, and execution are cleanly separated, making the code easier to understand and maintain.

5. **Prioritizing Readability**: While performance is important, we prioritized code clarity to make the implementation educational and maintainable.

6. **Comprehensive Error Handling**: Detailed error messages help users understand issues with their patterns or tensor shapes.

### Flow of Pattern Processing and Execution

The process of parsing and executing an einops pattern follows these main steps:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                  │     │                  │     │                  │     │                  │
│   Parse Pattern  │────►│ Resolve Axes and │────►│ Create Execution │────►│ Apply Operations │
│                  │     │   Dimensions     │     │       Plan       │     │                  │
│                  │     │                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

#### 1. Parse Pattern

- Split the pattern into LHS (input) and RHS (output) parts
- Tokenize each side and identify axis names, parentheses, ellipses, etc.
- Handle compositions (parenthesized groups) of axes
- Validate syntax and check for errors (e.g., unbalanced parentheses)

#### 2. Resolve Axes and Dimensions

- Map input tensor dimensions to named axes
- Handle ellipsis by determining the number of dimensions it represents
- Infer unknown axis lengths (e.g., in compositions where some lengths are known)
- Validate that tensor shape is compatible with the pattern

#### 3. Create Execution Plan

- Determine what operations are needed:
  - Initial reshape (for LHS compositions)
  - Transposition (for reordering existing dimensions)
  - Repeat/expand operations (for new dimensions)
  - Final reshape (for RHS compositions)
- Calculate the expected final shape

#### 4. Apply Operations

Apply the necessary NumPy operations in sequence:
1. Initial reshape if LHS has compositions
2. Transpose to reorder dimensions
3. Expand dimensions and repeat for new axes
4. Final reshape to handle RHS compositions

### Key Components

- **EinopsError**: Custom exception class for informative error messages
- **_parse_expression**: Parses one side of the pattern into a structured representation
- **_resolve_ellipsis**: Handles ellipsis notation by generating appropriate axis names
- **_process_pattern**: Main parser that analyzes the pattern and returns execution information
- **rearrange**: Public function that orchestrates the tensor transformation
- **repeat**: Specialized version of rearrange for repeating tensor elements

## Running Tests

The implementation includes a comprehensive test suite covering various use cases:

```bash
pytest custom_einops/test_einops.py -v
```

The tests include comparisons with the original einops library to verify correctness.

## Performance Considerations

This implementation prioritizes clarity and correctness over performance optimization. It may be slower than the original einops library, particularly for complex patterns or large tensors.

Potential optimization strategies include:

1. **Caching parsed patterns**: Avoid re-parsing patterns used repeatedly
2. **Vectorized operations**: Optimize the repeat operations for better performance
3. **Memory management**: Reduce unnecessary tensor copies
4. **Just-in-time compilation**: Explore using libraries like Numba for performance-critical sections

These optimizations could be implemented in future iterations while maintaining the clean architecture of the current implementation. 