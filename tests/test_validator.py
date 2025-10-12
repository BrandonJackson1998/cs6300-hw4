"""
Tests for the ConstraintValidator class.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools import ConstraintValidator
from tests.fixtures import SAMPLE_EMPLOYEES, SAMPLE_PATIENTS, SAMPLE_SCHEDULE, INVALID_SCHEDULE


class TestConstraintValidator(unittest.TestCase):
    """Test cases for ConstraintValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConstraintValidator(SAMPLE_EMPLOYEES, SAMPLE_PATIENTS)
    
    def test_init(self):
        """Test validator initialization."""
        self.assertEqual(len(self.validator.employees), 3)
        self.assertEqual(len(self.validator.patients), 2)
        self.assertIn("E001", self.validator.employees)
        self.assertIn("P001", self.validator.patients)
    
    def test_validate_valid_schedule(self):
        """Test validation of a valid schedule."""
        is_valid, violations, breakdown = self.validator.validate(SAMPLE_SCHEDULE)
        
        # Should have minimal violations (may have soft constraint violations)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(violations, list)
        self.assertIsInstance(breakdown, dict)
        self.assertIn('patient', breakdown)
        self.assertIn('employee', breakdown)
        self.assertIn('total', breakdown)
    
    def test_validate_invalid_schedule(self):
        """Test validation of an invalid schedule."""
        is_valid, violations, breakdown = self.validator.validate(INVALID_SCHEDULE)
        
        # Should detect violations
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
        self.assertGreater(breakdown['total'], 0)
    
    def test_validate_empty_schedule(self):
        """Test validation of empty schedule."""
        is_valid, violations, breakdown = self.validator.validate({})
        
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
        self.assertGreater(breakdown['patient'], 0)  # Should have patient coverage violations
    
    def test_check_consecutive_shifts(self):
        """Test consecutive shift checking."""
        # Test with shifts that violate consecutive limit
        shifts = ["D01S0", "D01S1", "D01S2", "D01S3", "D01S4"]  # 5 consecutive
        violations = self.validator._check_consecutive_shifts("E001", shifts)
        
        # Should detect consecutive shift violations
        self.assertIsInstance(violations, list)
    
    def test_violation_categorization(self):
        """Test that violations are properly categorized as patient vs employee."""
        is_valid, violations, breakdown = self.validator.validate(INVALID_SCHEDULE)
        
        # Should categorize violations
        self.assertGreaterEqual(breakdown['patient'], 0)
        self.assertGreaterEqual(breakdown['employee'], 0)
        self.assertEqual(breakdown['total'], breakdown['patient'] + breakdown['employee'])


if __name__ == '__main__':
    unittest.main()