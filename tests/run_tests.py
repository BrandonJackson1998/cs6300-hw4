"""
Test runner for the nursing scheduler test suite.
This file provides utilities for running all tests and generating reports.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_all_tests():
    """Run all tests in the test suite."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


def run_specific_test(test_module):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run only integration tests."""
    return run_specific_test('test_agent')


def run_unit_tests():
    """Run unit tests (everything except integration)."""
    test_modules = [
        'test_validator',
        'test_scorer', 
        'test_generator',
        'test_analyzer'
    ]
    
    all_passed = True
    for module in test_modules:
        print(f"\n{'='*50}")
        print(f"Running {module}")
        print('='*50)
        
        passed = run_specific_test(module)
        all_passed = all_passed and passed
    
    return all_passed


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run nursing scheduler tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--unit', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--module', type=str,
                       help='Run specific test module (e.g., test_validator)')
    
    args = parser.parse_args()
    
    if args.integration:
        print("Running integration tests...")
        success = run_integration_tests()
    elif args.unit:
        print("Running unit tests...")
        success = run_unit_tests()
    elif args.module:
        print(f"Running {args.module}...")
        success = run_specific_test(args.module)
    else:
        print("Running all tests...")
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)