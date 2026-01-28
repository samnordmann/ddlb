"""
Utility functions and classes for TP Row-wise implementations with Sequence Parallelism
"""

import os
from typing import Dict, Optional, Any, List, Union


class EnvVarGuard:
    """
    Class for managing environment variables.
    Sets up environment variables in __init__ and cleans them up in __del__.
    """
    
    def __init__(self, env_vars: Dict[str, str]):
        self.env_vars = env_vars
        self._original_values = {}
        
        # Store original values and set new ones
        for key, value in self.env_vars.items():
            self._original_values[key] = os.environ.get(key)
            os.environ[key] = value
    
    def __del__(self):
        # Restore original values or remove if they didn't exist
        for key in self.env_vars:
            if key in self._original_values:
                if self._original_values[key] is None:
                    del os.environ[key]
                else:
                    os.environ[key] = self._original_values[key]


class OptionsManager:
    """
    Manages options for TP Row-wise implementations with Sequence Parallelism.
    Handles parsing, validation, and storage of options.
    """
    
    # Options that are used by the benchmark runner but not by implementations
    BENCHMARK_OPTIONS = {'implementation'}
    
    def __init__(self, default_options: Dict[str, Any], allowed_values: Dict[str, List[Any]] = None):
        """
        Initialize the options manager.
        
        Args:
            default_options: Dictionary of default option values
            allowed_values: Dictionary mapping option names to lists of allowed values or range tuples
        """
        self.default_options = default_options.copy()
        self.allowed_values = allowed_values or {}
        self.options = self.default_options.copy()
    
    def parse(self, kwargs: Dict[str, Any]) -> None:
        """
        Parse and validate options from kwargs.
        
        Args:
            kwargs: Dictionary of options to parse
            
        Raises:
            ValueError: If an unknown option is provided or if a value is not allowed
        """
        # Filter out benchmark-specific options
        implementation_options = {
            k: v for k, v in kwargs.items() 
            if k not in self.BENCHMARK_OPTIONS
        }
        
        # Check for unknown options
        unknown_options = set(implementation_options.keys()) - set(self.default_options.keys())
        if unknown_options:
            raise ValueError(
                f"Unknown options provided: {unknown_options}. "
                f"Valid options are: {list(self.default_options.keys())}"
            )
        
        # Update options with provided values
        self.options.update(implementation_options)
        
        # Validate options against allowed values
        for option, allowed in self.allowed_values.items():
            if option in self.options:
                value = self.options[option]
                
                # Handle numeric range validation
                if isinstance(allowed, tuple) and len(allowed) == 2:
                    min_val, max_val = allowed
                    if not (isinstance(value, (int, float)) and min_val <= value <= max_val):
                        raise ValueError(
                            f"Invalid value for {option}: {value}. "
                            f"Must be a number between {min_val} and {max_val}"
                        )
                # Handle list of allowed values
                elif value not in allowed:
                    raise ValueError(
                        f"Invalid value for {option}: {value}. "
                        f"Must be one of {allowed}"
                    )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an option value."""
        return self.options.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get an option value using dictionary syntax."""
        return self.options[key]


def setup_ucc_env_vars(backend: str) -> Dict[str, str]:
    """
    Set up environment variables for UCC backend.
    
    Args:
        backend: Backend string (e.g., 'ucc/tl/nccl', 'ucc/tl/cuda')
        
    Returns:
        Dictionary of environment variables to set
    """
    env_vars = {}
    
    if backend.startswith('ucc/tl/'):
        tl = backend.split('/')[-1]
        env_vars["UCC_CL_BASIC_TLS"] = tl
        # env_vars[f"UCC_TL_{tl.upper()}_TUNE"] = "inf"
        # to avoid deadlock, disable ucx cuda transport
        if tl == "ucp":
          env_vars["UCX_RNDV_THRESH"] = "0"
          env_vars["UCX_TLS"] = "ib,cuda_copy"
    
    return env_vars

