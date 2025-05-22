"""
Utility functions and classes for TP Column-wise implementations
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
    Manages options for TP Column-wise implementations.
    Handles parsing, validation, and storage of options.
    """
    
    def __init__(self, default_options: Dict[str, Any], allowed_values: Dict[str, List[Any]] = None):
        """
        Initialize the options manager.
        
        Args:
            default_options: Dictionary of default option values
            allowed_values: Dictionary mapping option names to lists of allowed values
        """
        self.default_options = default_options.copy()
        self.allowed_values = allowed_values or {}
        self.options = self.default_options.copy()
    
    def parse(self, kwargs: Dict[str, Any]) -> None:
        """
        Parse and validate options from kwargs.
        
        Args:
            kwargs: Dictionary of options to parse
        """
        # Update options with provided values
        self.options.update(kwargs)
        
        # Validate options against allowed values
        for option, allowed in self.allowed_values.items():
            if option in self.options:
                value = self.options[option]
                if value not in allowed:
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
        # to avoid deadlock, disable ucx cuda transport
        if tl == "ucp":
          env_vars["UCX_RNDV_THRESH"] = "0"
          env_vars["UCX_TLS"] = "ib,cuda_copy"
    
    return env_vars 