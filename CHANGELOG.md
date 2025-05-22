# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Distributed matrix multiplication primitives
- Support for PyTorch Distributed with various backends
- Compute-only reference implementations
- Configurable benchmark parameters via JSON
- Automatic validation of results
- Performance visualization

### Changed
- Restructured project to follow Python best practices
- Moved CLI functionality to dedicated module
- Updated configuration format

### Fixed
- UCX transport configuration issues
- MPI initialization and cleanup

## [0.1.0] - 2024-03-12
### Added
- Initial release 