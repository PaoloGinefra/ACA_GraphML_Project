"""
Test runner for GDLPipelineLightningModule tests.

This script runs the comprehensive test suite for the PyTorch Lightning wrapper
of the GDLPipeline with various test configurations and reporting options.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def run_lightning_module_tests():
    """Run all GDLPipelineLightningModule tests."""

    test_file = str(Path(__file__).parent /
                    "test_GDLPipelineLightningModule.py")

    # Basic test run
    print("🚀 Running GDLPipelineLightningModule Test Suite")
    print("=" * 60)

    # Run with verbose output
    exit_code = pytest.main([
        test_file,
        "-v",
        "--tb=short",
        "--durations=10",
        "-x"  # Stop on first failure for faster debugging
    ])

    if exit_code == 0:
        print("\n✅ All GDLPipelineLightningModule tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")

    return exit_code


def run_quick_tests():
    """Run a subset of quick tests for development."""

    test_file = str(Path(__file__).parent /
                    "test_GDLPipelineLightningModule.py")

    print("⚡ Running Quick GDLPipelineLightningModule Tests")
    print("=" * 60)

    # Run only basic functionality tests (skip integration tests)
    exit_code = pytest.main([
        test_file + "::TestGDLPipelineLightningModuleBasics",
        test_file + "::TestGDLPipelineLightningModuleLossFunctions",
        test_file + "::TestGDLPipelineLightningModuleOptimizers",
        test_file + "::TestGDLPipelineLightningModuleConvenienceFunctions",
        "-v",
        "--tb=short"
    ])

    if exit_code == 0:
        print("\n✅ Quick tests passed!")
    else:
        print(f"\n❌ Some quick tests failed (exit code: {exit_code})")

    return exit_code


def run_integration_tests():
    """Run integration tests with PyTorch Lightning."""

    test_file = str(Path(__file__).parent /
                    "test_GDLPipelineLightningModule.py")

    print("🔗 Running GDLPipelineLightningModule Integration Tests")
    print("=" * 60)

    # Run only integration tests
    exit_code = pytest.main([
        test_file + "::TestGDLPipelineLightningModuleIntegration",
        "-v",
        "--tb=short",
        "-s"  # Don't capture output for integration tests
    ])

    if exit_code == 0:
        print("\n✅ Integration tests passed!")
    else:
        print(f"\n❌ Some integration tests failed (exit code: {exit_code})")

    return exit_code


def run_specific_test_class(test_class_name: str):
    """Run tests for a specific test class."""

    test_file = str(Path(__file__).parent /
                    "test_GDLPipelineLightningModule.py")

    print(f"🎯 Running {test_class_name} Tests")
    print("=" * 60)

    exit_code = pytest.main([
        f"{test_file}::{test_class_name}",
        "-v",
        "--tb=short"
    ])

    if exit_code == 0:
        print(f"\n✅ {test_class_name} tests passed!")
    else:
        print(
            f"\n❌ Some {test_class_name} tests failed (exit code: {exit_code})")

    return exit_code


def run_with_coverage():
    """Run tests with coverage reporting."""

    test_file = str(Path(__file__).parent /
                    "test_GDLPipelineLightningModule.py")

    print("📊 Running GDLPipelineLightningModule Tests with Coverage")
    print("=" * 60)

    # Run with coverage (requires pytest-cov)
    exit_code = pytest.main([
        test_file,
        "--cov=ACAgraphML.Pipeline.LightningModules",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ])

    if exit_code == 0:
        print("\n✅ All tests passed with coverage analysis!")
        print("📁 Coverage report generated in htmlcov/")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")

    return exit_code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GDLPipelineLightningModule tests")
    parser.add_argument(
        "--mode",
        choices=["all", "quick", "integration", "coverage"],
        default="all",
        help="Test mode to run"
    )
    parser.add_argument(
        "--class",
        dest="test_class",
        help="Run specific test class (e.g., TestGDLPipelineLightningModuleBasics)"
    )

    args = parser.parse_args()

    # Set PyTorch to use CPU for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.test_class:
        exit_code = run_specific_test_class(args.test_class)
    elif args.mode == "quick":
        exit_code = run_quick_tests()
    elif args.mode == "integration":
        exit_code = run_integration_tests()
    elif args.mode == "coverage":
        exit_code = run_with_coverage()
    else:  # "all"
        exit_code = run_lightning_module_tests()

    sys.exit(exit_code)
