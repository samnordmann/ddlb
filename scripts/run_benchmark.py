#!/usr/bin/env python3
"""Command-line script to run DDLB benchmarks."""

import json
import sys
from ddlb.cli import run_benchmark

def main():
    """Main entry point for the benchmark script."""
    config_path = 'scripts/config.json'

    # Load configuration from JSON file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    # Run benchmark without initializing communicator in parent
    run_benchmark(config)

if __name__ == '__main__':
    main() 