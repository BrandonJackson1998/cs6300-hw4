"""
Nursing Shift Scheduler Agent

A LangChain-based agent for optimizing nursing shift schedules
using Claude 3.5 Sonnet and specialized scheduling tools.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .agent import NursingSchedulerAgent
from .tools import (
    ConstraintValidator,
    ScheduleScorer,
    ScheduleGenerator,
    StaffingAnalyzer
)

__all__ = [
    "NursingSchedulerAgent",
    "ConstraintValidator",
    "ScheduleScorer",
    "ScheduleGenerator",
    "StaffingAnalyzer",
]