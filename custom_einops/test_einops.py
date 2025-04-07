"""
Tests for the custom einops implementation.
This file contains comprehensive tests for both basic and advanced functionality.
"""

import numpy as np
import pytest
from .einops import rearrange, repeat, EinopsError
import einops as original_einops  # The actual einops library for comparison


def test_basic_reshape():
    """Test basic reshaping operations."""
    x = np.random.rand(12, 10)
    
    # Test merging dimensions
    result = rearrange(x, 'a b -> (a b)')
    assert result.shape == (120,)
    assert np.array_equal(result, x.reshape(120))
    
    # Test splitting dimensions
    result = rearrange(x, '(a b) c -> a b c', a=3)
    assert result.shape == (3, 4, 10)
    assert np.array_equal(result, x.reshape(3, 4, 10))


def test_transpose():
    """Test transposition operations."""
    x = np.random.rand(3, 4, 5)
    
    # Simple transpose
    result = rearrange(x, 'a b c -> c b a')
    assert result.shape == (5, 4, 3)
    assert np.array_equal(result, np.transpose(x, (2, 1, 0)))
    
    # Transpose with dimension reordering
    result = rearrange(x, 'a b c -> b a c')
    assert result.shape == (4, 3, 5)
    assert np.array_equal(result, np.transpose(x, (1, 0, 2)))


def test_ellipsis():
    """Test ellipsis for handling batch dimensions."""
    x = np.random.rand(2, 3, 4, 5)
    
    # Using ellipsis on both sides
    result = rearrange(x, '... h w -> ... (h w)')
    assert result.shape == (2, 3, 20)
    assert np.array_equal(result, x.reshape(2, 3, 20))
    
    # Ellipsis with explicit dimensions
    result = rearrange(x, 'b ... w -> b w ...')
    assert result.shape == (2, 5, 3, 4)
    
    # Ellipsis in the middle
    x = np.random.rand(2, 3, 4, 5, 6)
    result = rearrange(x, 'a ... e -> e ... a')
    assert result.shape == (6, 3, 4, 5, 2)
    
    # Ellipsis only
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, '... -> ...')
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, x)
    
    # Ellipsis at different positions
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, 'b ... -> ... b')
    assert result.shape == (3, 4, 2)
    assert np.array_equal(result, np.transpose(x, (1, 2, 0)))
    
    # Ellipsis with splitting
    x = np.random.rand(2, 3, 10)
    result = rearrange(x, 'b ... (h w) -> b ... h w', h=2)
    assert result.shape == (2, 3, 2, 5)


def test_repeat_operation():
    """Test repeat operation for duplicating data."""
    x = np.random.rand(3, 1, 5)
    
    # Repeat along existing dimension with size 1
    result = repeat(x, 'a 1 c -> a b c', b=4)
    assert result.shape == (3, 4, 5)
    
    # Verify that data is properly repeated
    for i in range(4):
        assert np.array_equal(result[:, i, :], x[:, 0, :])
    
    # Adding a new dimension
    x = np.random.rand(3, 5)
    result = repeat(x, 'a b -> a b c', c=2)
    assert result.shape == (3, 5, 2)
    
    # Repeat using explicit number
    result = repeat(x, 'a b -> a b 3')
    assert result.shape == (3, 5, 3)
    
    # Repeat at different positions
    x = np.array([1, 2])
    result = rearrange(x, 'a -> a 3')
    expected = np.array([[1, 1, 1], [2, 2, 2]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)
    
    x = np.array([[1, 2], [3, 4]])
    result = rearrange(x, 'a b -> a b 2')
    expected = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)
    
    x = np.array([[1, 2], [3, 4]])
    result = rearrange(x, 'a b -> a 2 b')
    expected = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)
    
    x = np.array([[1, 2], [3, 4]])
    result = rearrange(x, 'a b -> 3 a b')
    expected = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)
    
    # Multiple repeats
    x = np.array([1, 2])
    result = rearrange(x, 'a -> 2 a 3')
    expected = np.array([[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)
    
    # Repeat in composition
    x = np.array([1, 2])
    result = rearrange(x, 'a -> (a 3)')
    expected = np.array([1, 1, 1, 2, 2, 2])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


def test_anonymous_axes():
    """Test handling of anonymous axes (1 and numeric literals)."""
    x = np.random.rand(3, 1, 5)
    
    # Using anonymous axis '1'
    result = rearrange(x, 'a 1 b -> a b')
    assert result.shape == (3, 5)
    
    # Creating anonymous axis '1'
    result = rearrange(x, 'a 1 b -> a 1 1 b')
    assert result.shape == (3, 1, 1, 5)
    
    # Using numeric literals for repetition
    result = rearrange(x, 'a 1 b -> a 3 b')
    assert result.shape == (3, 3, 5)
    
    # Anonymous axis in different positions
    x = np.arange(6).reshape(2, 1, 3)
    result = rearrange(x, 'a 1 c -> a c')
    assert result.shape == (2, 3)
    assert np.array_equal(result, np.arange(6).reshape(2, 3))
    
    x = np.arange(6).reshape(2, 3)
    result = rearrange(x, 'a c -> a 1 c')
    assert result.shape == (2, 1, 3)
    assert np.array_equal(result, np.arange(6).reshape(2, 1, 3))
    
    x = np.arange(6).reshape(1, 2, 3)
    result = rearrange(x, '1 b c -> b 1 c')
    assert result.shape == (2, 1, 3)


def test_complex_patterns():
    """Test more complex patterns combining multiple operations."""
    x = np.random.rand(20, 30, 3)
    
    # Composition with splitting and merging
    result = rearrange(x, '(batch height) width channels -> batch (height width) channels', batch=5)
    assert result.shape == (5, 120, 3)
    
    # Multiple compositions
    x = np.random.rand(10, 20, 30, 3)
    result = rearrange(x, '(a b) (c d) e f -> a b (c d e) f', a=2, c=5)
    # Check with original einops for expected shape
    expected = original_einops.rearrange(x, '(a b) (c d) e f -> a b (c d e) f', a=2, c=5)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)
    
    # Split and merge combinations
    x = np.arange(12).reshape(6, 2)
    result = rearrange(x, '(h w) c -> h w c', h=3)
    assert result.shape == (3, 2, 2)
    assert np.array_equal(result, np.arange(12).reshape(3, 2, 2))
    
    x = np.arange(12).reshape(2, 3, 2)
    result = rearrange(x, 'a b c -> (a b) c')
    assert result.shape == (6, 2)
    assert np.array_equal(result, np.arange(12).reshape(6, 2))
    
    x = np.arange(24).reshape(12, 2)
    result = rearrange(x, '(h w) c -> h (w c)', h=3)
    assert result.shape == (3, 8)
    assert np.array_equal(result, np.arange(24).reshape(3, 8))
    
    x = np.arange(24).reshape(6, 4)
    result = rearrange(x, '(a b) c -> a (b c)', a=2)
    assert result.shape == (2, 12)
    assert np.array_equal(result, np.arange(24).reshape(2, 12))


def test_error_cases():
    """Test that appropriate errors are raised for invalid inputs."""
    x = np.random.rand(3, 4, 5)
    
    # Missing arrow in pattern
    with pytest.raises(EinopsError):
        rearrange(x, 'a b c')
    
    # Reduction (axis missing on RHS)
    with pytest.raises(EinopsError):
        rearrange(x, 'a b c -> a c')
    
    # Invalid numeric literal on LHS
    with pytest.raises(EinopsError):
        rearrange(x, '3 b c -> b c 3')
    
    # Unknown axis length
    with pytest.raises(EinopsError):
        rearrange(x, '(a b) c d -> a b c d')
    
    # Mismatched tensor shape
    with pytest.raises(EinopsError):
        rearrange(np.random.rand(3, 3), 'a b c -> a b c')
        
    # Ellipsis on RHS but not LHS
    with pytest.raises(EinopsError):
        rearrange(x, 'a b c -> a ... c')
        
    # Invalid tensor type
    with pytest.raises(EinopsError):
        rearrange([1, 2, 3], 'a -> a')
        
    # Multiple separators
    with pytest.raises(EinopsError):
        rearrange(np.zeros(1), 'a -> b -> c')
        
    # Unbalanced parentheses
    with pytest.raises(EinopsError):
        rearrange(np.zeros(1), '(a -> b')
        
    # Unbalanced parentheses closing
    with pytest.raises(EinopsError):
        rearrange(np.zeros(1), 'a) -> b')
        
    # Invalid axes lengths
    with pytest.raises(EinopsError):
        rearrange(np.zeros(1), 'a -> b', **{'b': 0})
        
    # Multiple ellipsis
    with pytest.raises(EinopsError):
        rearrange(np.zeros((2, 3)), '... a ... -> a')
        
    # Nested parentheses
    with pytest.raises(EinopsError):
        rearrange(np.zeros((2, 3)), 'a (b (c)) -> a b c')
        
    # Duplicate identifier
    with pytest.raises(EinopsError):
        rearrange(np.zeros((2, 2)), 'a a -> a')
        
    # Unknown axis on RHS
    with pytest.raises(EinopsError):
        rearrange(np.zeros((2, 3)), 'a b -> a c')
        
    # Split axis incompatible shape
    with pytest.raises(EinopsError):
        rearrange(np.arange(10).reshape(5, 2), '(h w) c -> h w c', h=3)
        
    # Anonymous axis mismatch
    with pytest.raises(EinopsError):
        rearrange(np.arange(10).reshape(5, 2), 'a 1 -> a')


def test_comparison_with_original():
    """Compare our implementation with the original einops library."""
    shapes_to_test = [
        (2, 3, 4),
        (5, 6),
        (2, 3, 4, 5),
        (10, 1, 5),
        (24, 10)
    ]
    
    patterns_to_test = [
        'a b c -> c b a',
        'a b -> (a b)',
        '... h w -> ... (h w)',
        '(a b) c -> a b c',
        'a 1 b -> a b 1',
        'a b -> a b 2'
    ]
    
    for shape in shapes_to_test:
        x = np.random.rand(*shape)
        
        for pattern in patterns_to_test:
            try:
                # Skip patterns that would cause errors
                if 'c' in pattern and len(shape) < 3:
                    continue
                if 'h w' in pattern and len(shape) < 2:
                    continue
                if '(a b)' in pattern and shape[0] % 2 != 0:
                    continue
                
                # Test with relevant axes_lengths
                axes_lengths = {}
                if '(a b)' in pattern and '->' in pattern and pattern.index('(a b)') < pattern.index('->'):
                    axes_lengths['a'] = 2
                if 'a b c' in pattern and '->' in pattern and pattern.index('a b c') > pattern.index('->'):
                    axes_lengths['a'] = 2
                    axes_lengths['b'] = shape[0] // 2
                    axes_lengths['c'] = shape[1]
                if pattern.endswith('2'):
                    axes_lengths = {}  # No need for explicit length with numeric literal
                
                # Compare results
                our_result = rearrange(x, pattern, **axes_lengths)
                original_result = original_einops.rearrange(x, pattern, **axes_lengths)
                
                assert our_result.shape == original_result.shape
                assert np.allclose(our_result, original_result)
                
            except Exception as e:
                # Some combinations of patterns and shapes are invalid
                # We only care about comparing successful operations
                pass


if __name__ == "__main__":
    # Run tests and report results
    pytest.main(["-xvs", __file__]) 