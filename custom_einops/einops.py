"""
Custom implementation of einops functionality, focusing on the rearrange operation.
This module provides a concise way to manipulate tensors using Einstein notation-inspired syntax,
supporting operations like reshaping, transposition, splitting, merging, and repeating axes.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Union, Optional
import math


class EinopsError(ValueError):
    """Custom exception for errors in einops operations."""
    pass


def _parse_expression(expression: str) -> Tuple[List[Union[str, List[str]]], Set[str], bool, bool]:
    """
    Parse one side of the pattern (LHS or RHS) into a structured representation.
    
    Args:
        expression: String representing either LHS or RHS of the pattern
        
    Returns:
        Tuple containing:
            - List of parsed axes or axis groups
            - Set of unique named axes
            - Flag indicating if ellipsis is present
            - Flag indicating if composition/decomposition is present
            
    Raises:
        EinopsError: For syntax errors in the pattern
    """
    raw_axes = []
    identifiers = set()
    has_ellipsis = False
    has_composition = False
    
    # Process expression, handling parentheses
    current_composition = None
    paren_level = 0
    
    # Preprocess for easier tokenization
    processed_expr = expression.replace('(', ' ( ').replace(')', ' ) ')
    tokens = processed_expr.split()
    
    for token in tokens:
        if not token:
            continue
            
        if token == '...':
            if has_ellipsis:
                raise EinopsError(f"Multiple ellipses (...) found in '{expression}'")
            if paren_level > 0:
                current_composition.append('...')
            else:
                raw_axes.append('...')
            has_ellipsis = True
            
        elif token == '(':
            if paren_level > 0:
                raise EinopsError(f"Nested parentheses not allowed in '{expression}'")
            paren_level += 1
            current_composition = []
            has_composition = True
            
        elif token == ')':
            if paren_level == 0:
                raise EinopsError(f"Unbalanced parentheses in '{expression}'")
            paren_level -= 1
            raw_axes.append(list(current_composition))
            current_composition = None
            
        elif token.isidentifier():
            if token in identifiers and paren_level == 0:
                raise EinopsError(f"Duplicate identifier '{token}' in '{expression}'")
            
            if paren_level > 0:
                current_composition.append(token)
            else:
                raw_axes.append(token)
                
            identifiers.add(token)
            
        elif token.isdigit():
            num_val = int(token)
            if num_val <= 0:
                raise EinopsError(f"Numeric axis must be positive, found '{token}' in '{expression}'")
                
            if paren_level > 0:
                current_composition.append(str(num_val))
            else:
                raw_axes.append(str(num_val))
                
        else:
            raise EinopsError(f"Invalid token '{token}' in '{expression}'")
    
    if paren_level != 0:
        raise EinopsError(f"Unbalanced parentheses in '{expression}'")
        
    return raw_axes, identifiers, has_ellipsis, has_composition


def _resolve_ellipsis(
    pattern_axes: List[Union[str, List[str]]],
    has_ellipsis: bool,
    tensor_shape: Tuple[int, ...],
    explicit_axes_count: int
) -> List[str]:
    """
    Resolve ellipsis in pattern by generating appropriate axis names.
    
    Args:
        pattern_axes: List of parsed axes from the pattern
        has_ellipsis: Whether the pattern contains ellipsis
        tensor_shape: Shape of the input tensor
        explicit_axes_count: Number of explicit (non-ellipsis) axes
        
    Returns:
        List of ellipsis axis names if ellipsis present, empty list otherwise
        
    Raises:
        EinopsError: If tensor rank doesn't match pattern requirements
    """
    if not has_ellipsis:
        return []
        
    ellipsis_dims = len(tensor_shape) - explicit_axes_count
    if ellipsis_dims < 0:
        raise EinopsError(f"Input tensor has {len(tensor_shape)} dimensions, but pattern requires at least {explicit_axes_count}")
        
    return [f"_ellipsis_{i}" for i in range(ellipsis_dims)]


def _process_pattern(
    pattern: str, 
    tensor_shape: Tuple[int, ...], 
    axes_lengths: Dict[str, int]
) -> Tuple[List[str], List[str], Dict[str, int], Tuple[int, ...], Dict[str, int]]:
    """
    Process the pattern string and extract information needed for rearrangement.
    
    Args:
        pattern: The einops pattern string (e.g. 'b h w -> b (h w)')
        tensor_shape: Shape of the input tensor
        axes_lengths: Dictionary of provided axis lengths
        
    Returns:
        Tuple containing:
            - List of decomposed LHS axes
            - List of decomposed RHS axes
            - Dictionary of resolved axis lengths
            - Tuple representing the final output shape
            - Dictionary mapping repeat axes to their lengths
            
    Raises:
        EinopsError: For invalid patterns or shape mismatches
    """
    # Split and parse both sides of the pattern
    if pattern.count('->') != 1:
        raise EinopsError(f"Pattern must contain exactly one '->' separator.")
        
    lhs_str, rhs_str = pattern.split('->')
    lhs_str = lhs_str.strip()
    rhs_str = rhs_str.strip()
    
    lhs_raw_axes, lhs_identifiers, lhs_has_ellipsis, lhs_has_composition = _parse_expression(lhs_str)
    rhs_raw_axes, rhs_identifiers, rhs_has_ellipsis, rhs_has_composition = _parse_expression(rhs_str)
    
    # Validate ellipsis usage
    if not lhs_has_ellipsis and rhs_has_ellipsis:
        raise EinopsError(f"Ellipsis found in right side, but not left side of pattern '{pattern}'")
    
    # Count non-ellipsis dimensions
    lhs_explicit_dims = sum(1 for ax in lhs_raw_axes if ax != '...')
    
    # Resolve ellipsis axes
    ellipsis_axes = _resolve_ellipsis(lhs_raw_axes, lhs_has_ellipsis, tensor_shape, lhs_explicit_dims)
    
    # Initialize axis-length mapping
    resolved_axes_lengths = axes_lengths.copy()
    
    # Store ellipsis axes dimensions in the resolved_axes_lengths dictionary
    if lhs_has_ellipsis:
        ellipsis_idx = lhs_raw_axes.index('...')
        for i, axis_name in enumerate(ellipsis_axes):
            dim_idx = ellipsis_idx + i
            if dim_idx < len(tensor_shape):
                resolved_axes_lengths[axis_name] = tensor_shape[dim_idx]
    
    # Process LHS axes and map to tensor dimensions
    decomposed_lhs_axes = []
    tensor_dim_idx = 0
    
    for axis_group in lhs_raw_axes:
        if axis_group == '...':
            decomposed_lhs_axes.extend(ellipsis_axes)
            tensor_dim_idx += len(ellipsis_axes)
            continue
            
        if tensor_dim_idx >= len(tensor_shape):
            raise EinopsError(f"Pattern '{lhs_str}' requires more dimensions than tensor shape {tensor_shape}")
            
        dim_size = tensor_shape[tensor_dim_idx]
        
        if isinstance(axis_group, list):
            # Handle composition (h w) -> compute individual axis lengths
            comp_axes = []
            known_product = 1
            unknown_axes = []
            
            for ax in axis_group:
                if ax == '1':
                    continue
                elif ax.isdigit() and int(ax) > 1:
                    raise EinopsError(f"Numeric literal {ax} > 1 not allowed in LHS composition")
                elif ax in resolved_axes_lengths:
                    known_product *= resolved_axes_lengths[ax]
                    comp_axes.append(ax)
                else:
                    unknown_axes.append(ax)
                    comp_axes.append(ax)
            
            # Validate and infer unknown axis lengths
            if len(unknown_axes) > 1:
                raise EinopsError(f"Multiple unknown axes {unknown_axes} in composition {axis_group}")
                
            if len(unknown_axes) == 1:
                unknown_axis = unknown_axes[0]
                if dim_size % known_product != 0:
                    raise EinopsError(f"Dimension size {dim_size} not divisible by known product {known_product}")
                    
                inferred_size = dim_size // known_product
                resolved_axes_lengths[unknown_axis] = inferred_size
            elif known_product != dim_size:
                raise EinopsError(f"Composition size mismatch: expected {known_product}, got {dim_size}")
                
            decomposed_lhs_axes.extend(comp_axes)
            
        elif axis_group == '1':
            if dim_size != 1:
                raise EinopsError(f"Expected dimension of size 1, got {dim_size}")
            # Skip adding '1' to decomposed axes
        elif axis_group.isdigit() and int(axis_group) > 1:
            raise EinopsError(f"Numeric literal {axis_group} > 1 not allowed on LHS")
        else:
            # Regular named axis
            if axis_group in resolved_axes_lengths and resolved_axes_lengths[axis_group] != dim_size:
                raise EinopsError(f"Axis '{axis_group}' length mismatch: got {dim_size}, expected {resolved_axes_lengths[axis_group]}")
                
            resolved_axes_lengths[axis_group] = dim_size
            decomposed_lhs_axes.append(axis_group)
            
        tensor_dim_idx += 1
    
    # Validate tensor rank matches pattern
    if tensor_dim_idx != len(tensor_shape):
        raise EinopsError(f"Pattern '{lhs_str}' doesn't match tensor shape {tensor_shape}")
    
    # Process RHS axes and build final shape
    decomposed_rhs_axes = []
    repeat_axes_info = {}
    final_shape = []
    
    for axis_group in rhs_raw_axes:
        if axis_group == '...':
            decomposed_rhs_axes.extend(ellipsis_axes)
            for ax in ellipsis_axes:
                final_shape.append(resolved_axes_lengths[ax])
            continue
            
        if isinstance(axis_group, list):
            # Handle composition on RHS
            comp_axes = []
            group_size_product = 1
            
            for ax in axis_group:
                if ax == '1':
                    comp_axes.append('_anon_1')
                    # Size remains 1
                elif ax.isdigit():
                    # Handle repeat via numeric literal
                    repeat_len = int(ax)
                    repeat_name = f"_repeat_{len(repeat_axes_info)}"
                    repeat_axes_info[repeat_name] = repeat_len
                    comp_axes.append(repeat_name)
                    group_size_product *= repeat_len
                elif ax in resolved_axes_lengths:
                    # Known axis from LHS or kwargs
                    comp_axes.append(ax)
                    group_size_product *= resolved_axes_lengths[ax]
                elif ax in axes_lengths:
                    # New axis specified in kwargs
                    resolved_axes_lengths[ax] = axes_lengths[ax]
                    repeat_axes_info[ax] = axes_lengths[ax]
                    comp_axes.append(ax)
                    group_size_product *= axes_lengths[ax]
                else:
                    raise EinopsError(f"Unknown axis '{ax}' in RHS composition")
            
            decomposed_rhs_axes.extend(comp_axes)
            final_shape.append(group_size_product)
            
        elif axis_group == '1':
            decomposed_rhs_axes.append('_anon_1')
            final_shape.append(1)
        elif axis_group.isdigit():
            # Handle repeat via numeric literal
            repeat_len = int(axis_group)
            repeat_name = f"_repeat_{len(repeat_axes_info)}"
            repeat_axes_info[repeat_name] = repeat_len
            decomposed_rhs_axes.append(repeat_name)
            final_shape.append(repeat_len)
        elif axis_group in decomposed_lhs_axes:
            # Existing axis from LHS
            decomposed_rhs_axes.append(axis_group)
            final_shape.append(resolved_axes_lengths[axis_group])
        elif axis_group in axes_lengths:
            # New axis for repetition
            resolved_axes_lengths[axis_group] = axes_lengths[axis_group]
            repeat_axes_info[axis_group] = axes_lengths[axis_group]
            decomposed_rhs_axes.append(axis_group)
            final_shape.append(axes_lengths[axis_group])
        else:
            raise EinopsError(f"Unknown axis '{axis_group}' on RHS")
    
    # Check for missing axes (reduction not supported in rearrange)
    lhs_axis_set = set(decomposed_lhs_axes)
    rhs_axis_set = {ax for ax in decomposed_rhs_axes if not (ax.startswith('_repeat_') or ax == '_anon_1')}
    
    missing_on_rhs = lhs_axis_set - rhs_axis_set
    if missing_on_rhs:
        raise EinopsError(f"Axes {missing_on_rhs} present on LHS but missing on RHS. Reduction not supported.")
    
    return decomposed_lhs_axes, decomposed_rhs_axes, resolved_axes_lengths, tuple(final_shape), repeat_axes_info


def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths: int) -> np.ndarray:
    """
    Rearrange a tensor according to the pattern.
    
    This function supports:
    - Reshaping (merging/splitting dimensions)
    - Transposition (reordering dimensions)
    - Repeating elements along new dimensions
    
    Args:
        tensor: Input tensor (numpy ndarray)
        pattern: String pattern like 'b h w -> b (h w)' 
        **axes_lengths: Lengths for axes introduced in the pattern
        
    Returns:
        Rearranged tensor
        
    Raises:
        EinopsError: For invalid patterns or shape mismatches
    """
    # Validate inputs
    if not isinstance(tensor, np.ndarray):
        raise EinopsError("Input tensor must be a NumPy ndarray")
    if not isinstance(pattern, str):
        raise EinopsError("Pattern must be a string")
    if '->' not in pattern:
        raise EinopsError("Pattern must contain '->' separator")
    if pattern.count('->') > 1:
        raise EinopsError("Pattern must contain exactly one '->' separator")
        
    # Parse and process the pattern
    decomposed_lhs, decomposed_rhs, axes_lengths_dict, final_shape, repeat_info = _process_pattern(
        pattern, tensor.shape, axes_lengths
    )
    
    # Extract operation flags and prepare shapes
    lhs_shape = tuple(axes_lengths_dict[ax] for ax in decomposed_lhs)
    needs_initial_reshape = tensor.shape != lhs_shape
    
    # Identifying axes for transpose
    transpose_indices = []
    repeat_axes = []
    
    # Build execution plan
    result = tensor
    
    # Step 1: Initial reshape if composition on LHS
    if needs_initial_reshape:
        try:
            result = result.reshape(lhs_shape)
        except ValueError as e:
            raise EinopsError(f"Reshape error: {e}, cannot reshape {tensor.shape} to {lhs_shape}")
    
    # Step 2: Prepare transpose (non-repeat axes)
    lhs_axes_set = set(decomposed_lhs)
    for i, ax in enumerate(decomposed_rhs):
        if ax in lhs_axes_set:
            transpose_indices.append(decomposed_lhs.index(ax))
        elif ax in repeat_info or ax == '_anon_1':
            repeat_axes.append((i, ax))
    
    # Step 3: Transpose if needed
    if transpose_indices and transpose_indices != list(range(len(transpose_indices))):
        result = np.transpose(result, transpose_indices)
    
    # Step 4: Handle repeats by expanding dims and repeating
    if repeat_axes:
        # Sort by insertion index to maintain proper order
        repeat_axes.sort(key=lambda x: x[0])
        
        # For complex patterns with multiple repeats, handle each one separately
        for i, (idx, ax) in enumerate(repeat_axes):
            # Calculate the current tensor shape for correct axis index adjustment
            current_ndim = result.ndim
            adjusted_idx = min(idx, current_ndim)
            
            # Expand dimension
            result = np.expand_dims(result, axis=adjusted_idx)
            
            # Repeat if needed
            if ax in repeat_info:
                repeat_count = repeat_info[ax]
                result = np.repeat(result, repeat_count, axis=adjusted_idx)
    
    # Step 5: Final reshape to target shape if needed
    if result.shape != final_shape:
        try:
            result = result.reshape(final_shape)
        except ValueError as e:
            raise EinopsError(f"Final reshape error: {e}, cannot reshape {result.shape} to {final_shape}")
    
    return result


def repeat(tensor: np.ndarray, pattern: str, **axes_lengths: int) -> np.ndarray:
    """
    Repeat elements of a tensor according to the pattern.
    
    This is a specialized case of rearrange where new dimensions are added
    with repeated data.
    
    Args:
        tensor: Input tensor (numpy ndarray)
        pattern: String pattern like 'b h w -> b h w c' where 'c' is new
        **axes_lengths: Lengths for axes introduced in the pattern
        
    Returns:
        Tensor with repeated elements
        
    Raises:
        EinopsError: For invalid patterns or shape mismatches
    """
    # For repeat, all axes from the left side must be on the right side
    lhs_str, rhs_str = pattern.split('->')
    lhs_tokens = set()
    
    # Extract tokens from LHS
    for token in lhs_str.strip().replace('(', ' ').replace(')', ' ').split():
        if token not in ['...', ''] and not token.isdigit():
            lhs_tokens.add(token)
    
    # Check if all LHS tokens appear in RHS
    for token in lhs_tokens:
        if token not in rhs_str:
            raise EinopsError(f"All axes from LHS must appear in RHS for repeat, missing: {token}")
    
    # Delegate to rearrange
    return rearrange(tensor, pattern, **axes_lengths) 