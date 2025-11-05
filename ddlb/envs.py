"""
Centralized environment variable management for DDLB.

This module provides a unified interface for accessing environment variables
across different distributed computing frameworks (OpenMPI, SLURM, PMI, etc.).
"""

import os
from typing import Optional, Union


def get_env(
    key: str, 
    default: Optional[str] = None, 
    fallback_keys: Optional[list[str]] = None
) -> str:
    """
    Get environment variable with optional fallback keys.
    
    Args:
        key: Primary environment variable name
        default: Default value if no environment variables are found
        fallback_keys: List of fallback environment variable names to try
        
    Returns:
        String value from environment variable or default
        
    Example:
        >>> get_env("OMPI_COMM_WORLD_RANK", "0", ["SLURM_PROCID", "PMI_RANK"])
        
    This will try OMPI_COMM_WORLD_RANK first, then SLURM_PROCID, then PMI_RANK,
    and finally return "0" if none are set.
    """
    # Try primary key first
    value = os.getenv(key)
    if value is not None:
        return value
    
    # Try fallback keys in order
    if fallback_keys:
        for fallback_key in fallback_keys:
            value = os.getenv(fallback_key)
            if value is not None:
                return value
    
    # Return default if nothing found
    return default if default is not None else ""


def get_rank() -> int:
    """Get global process rank with fallbacks for different MPI implementations."""
    return int(get_env("OMPI_COMM_WORLD_RANK", "0", ["SLURM_PROCID", "PMI_RANK"]))


def get_local_rank() -> int:
    """Get local process rank (within node) with fallbacks."""
    return int(get_env("OMPI_COMM_WORLD_LOCAL_RANK", "0", ["SLURM_LOCALID"]))


def get_world_size() -> int:
    """Get total number of processes with fallbacks."""
    return int(get_env("OMPI_COMM_WORLD_SIZE", "1", ["SLURM_NTASKS", "PMI_SIZE"]))


def get_local_size() -> int:
    """Get number of processes per node with fallbacks."""
    return int(get_env("OMPI_COMM_WORLD_LOCAL_SIZE", "1", ["SLURM_NTASKS_PER_NODE"]))


def get_master_addr() -> str:
    """Get master node address for distributed training."""
    return get_env("DDLB_MASTER_ADDR", "localhost")


def get_master_port() -> str:
    """Get master node port for distributed training."""
    return get_env("DDLB_MASTER_PORT", "12345")


def get_jax_coord_addr() -> str:
    """Get JAX coordinator address."""
    return get_env("JAX_COORD_ADDR", "127.0.0.1:12355")
