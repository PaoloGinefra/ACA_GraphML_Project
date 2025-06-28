"""
Test runner for GDLPipeline test suite.

This script runs the comprehensive test suite for GDLPipeline, including:
- Unit tests
- Integration tests  
- Benchmark tests
- ZINC dataset tests (if available)

Usage:
    python run_gdl_pipeline_tests.py [test_type]
    
Test types:
    - all: Run all tests
    - unit: Run unit tests only
    - integration: Run integration tests only
    - benchmark: Run benchmark tests only
    - zinc: Run ZINC-specific tests only
    - quick: Run quick tests only (no benchmarks)
"""

import sys
import os
import pytest
import argparse
from typing import List, Optional

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

# Test file mappings
TEST_FILES = {
    'unit': 'test_GDLPipeline.py',
    'integration': 'test_GDLPipeline_integration.py',
    'benchmark': 'test_GDLPipeline_benchmark.py'
}


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import torch
        import torch_geometric
    except ImportError as e:
        missing_deps.append(f"PyTorch/PyG: {e}")

    try:
        from ACAgraphML.Pipeline.Models.GDLPipeline import GDLPipeline
    except ImportError as e:
        missing_deps.append(f"GDLPipeline: {e}")

    try:
        from ACAgraphML.Dataset import ZINC_Dataset
        zinc_available = True
    except ImportError:
        zinc_available = False

    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        return False, False

    return True, zinc_available


def run_tests(test_type: str = 'all', verbose: bool = True, exit_on_fail: bool = True) -> int:
    """Run specified tests and return exit code."""

    # Check dependencies
    deps_ok, zinc_available = check_dependencies()
    if not deps_ok:
        print("Cannot run tests due to missing dependencies.")
        return 1

    # Build pytest arguments
    pytest_args = []

    if test_type == 'all':
        pytest_args.extend([
            TEST_FILES['unit'],
            TEST_FILES['integration'],
            TEST_FILES['benchmark']
        ])
    elif test_type == 'quick':
        pytest_args.extend([
            TEST_FILES['unit'],
            TEST_FILES['integration'],
            '-m', 'not benchmark and not slow'
        ])
    elif test_type == 'zinc':
        if zinc_available:
            pytest_args.extend([
                TEST_FILES['integration'] + '::TestGDLPipelineZINCIntegration',
                '-v'
            ])
        else:
            pytest_args.extend([
                TEST_FILES['integration'] + '::TestGDLPipelineMockZINC',
                '-v'
            ])
    elif test_type in TEST_FILES:
        pytest_args.append(TEST_FILES[test_type])
    else:
        print(f"Unknown test type: {test_type}")
        return 1

    # Add common arguments
    if verbose:
        pytest_args.append('-v')

    pytest_args.extend([
        '--tb=short',
        '--strict-markers',
        '-x' if exit_on_fail else '--continue-on-collection-errors'
    ])

    # Print test configuration
    print("GDLPipeline Test Suite")
    print("=" * 50)
    print(f"Test type: {test_type}")
    print(f"ZINC dataset available: {zinc_available}")
    print(f"Verbose output: {verbose}")
    print(f"Exit on failure: {exit_on_fail}")
    print()

    # Run tests
    exit_code = pytest.main(pytest_args)

    # Print summary
    print("\n" + "=" * 50)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ Tests failed (exit code: {exit_code})")

    return exit_code


def run_specific_test_class(test_file: str, test_class: str, verbose: bool = True) -> int:
    """Run a specific test class."""
    pytest_args = [
        f"{test_file}::{test_class}",
        '-v' if verbose else '',
        '--tb=short'
    ]

    print(f"Running {test_class} from {test_file}")
    return pytest.main([arg for arg in pytest_args if arg])


def run_test_discovery():
    """Discover and list all available tests."""
    print("Discovering GDLPipeline tests...")
    print()

    for test_type, test_file in TEST_FILES.items():
        if os.path.exists(test_file):
            print(f"{test_type.upper()} TESTS ({test_file}):")

            # Use pytest to collect test items
            pytest_args = [test_file, '--collect-only', '-q']
            pytest.main(pytest_args)
            print()
        else:
            print(f"{test_type.upper()} TESTS: {test_file} not found")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run GDLPipeline test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_gdl_pipeline_tests.py                    # Run all tests
    python run_gdl_pipeline_tests.py unit              # Run unit tests only
    python run_gdl_pipeline_tests.py benchmark         # Run benchmark tests
    python run_gdl_pipeline_tests.py --discover        # List available tests
    python run_gdl_pipeline_tests.py quick --no-exit   # Quick tests, don't exit on fail
        """
    )

    parser.add_argument(
        'test_type',
        nargs='?',
        default='all',
        choices=['all', 'unit', 'integration', 'benchmark', 'zinc', 'quick'],
        help='Type of tests to run (default: all)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Run tests in quiet mode'
    )

    parser.add_argument(
        '--no-exit',
        action='store_true',
        help='Continue on test failures'
    )

    parser.add_argument(
        '--discover',
        action='store_true',
        help='Discover and list available tests'
    )

    parser.add_argument(
        '--class',
        dest='test_class',
        help='Run specific test class (requires test_type)'
    )

    args = parser.parse_args()

    # Change to tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tests_dir)

    if args.discover:
        run_test_discovery()
        return 0

    if args.test_class:
        if args.test_type not in TEST_FILES:
            print(f"Invalid test type for class selection: {args.test_type}")
            return 1
        return run_specific_test_class(
            TEST_FILES[args.test_type],
            args.test_class,
            verbose=not args.quiet
        )

    return run_tests(
        test_type=args.test_type,
        verbose=not args.quiet,
        exit_on_fail=not args.no_exit
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
