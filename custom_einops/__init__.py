"""
Custom implementation of einops functionality.

This module provides a clean, functional implementation of the core
einops operations for tensor manipulation.
"""

from .einops import rearrange, repeat, EinopsError

__all__ = ['rearrange', 'repeat', 'EinopsError'] 