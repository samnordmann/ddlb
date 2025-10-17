#!/usr/bin/env python3
"""Command-line script to run DDLB benchmarks."""

import json
import sys
import argparse
from ddlb.cli import run_benchmark

def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Command-line script to run DDLB benchmarks.")
    parser.add_argument(
        'config_path',
        nargs='?',
        default='scripts/config.json',
        help='Path to the JSON configuration file (default: scripts/config.json)'
    )
    args = parser.parse_args()
    config_path = args.config_path

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