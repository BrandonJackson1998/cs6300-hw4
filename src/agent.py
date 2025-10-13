"""
Nursing Shift Scheduler Agent

Uses LangChain + Google Gemini with 4 tools to optimize nursing schedules.
Integrates with LangSmith for tracing and optimization.

Run: python src/agent.py
"""

import json
import os
import sys
import time
import copy
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict, deque
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# Import our tool implementations
from .tools import (
    ConstraintValidator, ScheduleScorer, 
    ScheduleGenerator, StaffingAnalyzer
)

# Load environment variables
load_dotenv()


class RateLimiter:
    """Global rate limiter for API calls"""
    def __init__(self, calls_per_minute=9):
        self.calls_per_minute = calls_per_minute
        self.call_times = deque()
    
    def wait_if_needed(self):
        """Wait if we're approaching rate limit"""
        now = time.time()
        
        # Remove calls older than 60 seconds
        while self.call_times and now - self.call_times[0] > 60:
            self.call_times.popleft()
        
        # If we've made too many calls, wait
        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.call_times[0]) + 1
            if sleep_time > 0:
                print(f"⏳ Rate limit: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.call_times.clear()
        
        # Record this call
        self.call_times.append(time.time())


# Logging utility
class TeeLogger:
    """Logs to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def generate_staffing_narrative(violations, employees, patients, schedule):
    """
    Generate a comprehensive natural language staffing analysis to save with the schedule.
    This will be displayed in the visualization.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠ GEMINI_API_KEY not found - skipping AI narrative")
            return None
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        
        # ===== SHIFT TIMING ANALYSIS =====
        shift_violations = defaultdict(int)
        shift_names = {
            0: "12am-4am (Night)",
            1: "4am-8am (Early Morning)", 
            2: "8am-12pm (Morning)",
            3: "12pm-4pm (Afternoon)",
            4: "4pm-8pm (Evening)",
            5: "8pm-12am (Night)"
        }
        
        for v in violations:
            match = re.search(r'D\d{2}S(\d)', v)
            if match:
                shift_num = int(match.group(1))
                shift_violations[shift_num] += 1
        
        # ===== EMPLOYEE AVAILABILITY BY SHIFT =====
        employee_availability_by_shift = defaultdict(int)
        for emp in employees:
            for shift_id in emp.get('available_shifts', []):
                if len(shift_id) >= 5:
                    shift_num = int(shift_id[4:])
                    employee_availability_by_shift[shift_num] += 1
        
        # ===== PATIENT NEEDS BY SHIFT =====
        patient_needs_by_shift = defaultdict(int)
        for patient in patients:
            for shift_id in patient.get('care_shifts', []):
                if len(shift_id) >= 5:
                    shift_num = int(shift_id[4:])
                    patient_needs_by_shift[shift_num] += patient.get('nurses_needed', 1)
        
        # ===== SHIFT SUPPLY/DEMAND ANALYSIS =====
        shift_analysis = []
        for shift_num in range(6):
            available = employee_availability_by_shift[shift_num]
            needed = patient_needs_by_shift[shift_num]
            violations_count = shift_violations[shift_num]
            
            # Calculate actual scheduled nurses for this shift
            scheduled = 0
            shift_prefix = f"S{shift_num}"
            for shift_id, patient_assignments in schedule.items():
                if shift_id.endswith(shift_prefix):
                    for patient_id, employee_ids in patient_assignments.items():
                        scheduled += len(employee_ids)
            
            ratio = (available / needed * 100) if needed > 0 else 0
            coverage_ratio = (scheduled / needed * 100) if needed > 0 else 0
            
            status = "✓ ADEQUATE" if coverage_ratio >= 100 else "⚠ UNDERSTAFFED" if coverage_ratio >= 70 else "🚨 CRITICAL"
            shift_analysis.append(
                f"  {shift_names[shift_num]}: {scheduled} scheduled / {needed} needed ({coverage_ratio:.0f}%) {status} | {available} available"
            )
            if violations_count > 0:
                shift_analysis[-1] += f" - {violations_count} violations"
        
        shift_analysis_text = '\n'.join(shift_analysis)
        
        # ===== SKILL ANALYSIS =====
        # Count available shifts by skill (preferred availability)
        employee_skills_preferred = defaultdict(int)
        employee_skills_maximum = defaultdict(int)
        
        for emp in employees:
            # Preferred availability (when they want to work)
            preferred_shifts_count = len(emp.get('available_shifts', []))
            # Maximum availability (assuming 24/7 availability - 6 shifts per day * 28 days)
            maximum_shifts_count = 6 * 28  # 168 total shifts per month
            
            for skill in emp.get('skills', []):
                employee_skills_preferred[skill] += preferred_shifts_count
                employee_skills_maximum[skill] += maximum_shifts_count
        
        # Count patient skill requirements (shifts needed)
        patient_skill_needs = defaultdict(int)
        for patient in patients:
            for skill in patient.get('required_skills', []):
                patient_skill_needs[skill] += len(patient.get('care_shifts', []))
        
        # Analyze skill gaps from violations
        skill_gaps_from_violations = defaultdict(int)
        for v in violations:
            if 'requires skills' in v.lower():
                match = re.search(r"requires skills \{([^}]+)\}", v)
                if match:
                    skills_str = match.group(1)
                    skills = [s.strip().strip("'\"") for s in skills_str.split(',')]
                    for skill in skills:
                        if skill:
                            skill_gaps_from_violations[skill] += 1
        
        # Build skill analysis
        skill_analysis_lines = []
        all_skills = set(employee_skills_preferred.keys()) | set(patient_skill_needs.keys())
        
        for skill in sorted(all_skills):
            preferred_shifts = employee_skills_preferred[skill]
            maximum_shifts = employee_skills_maximum[skill]
            need_count = patient_skill_needs[skill]
            gap_count = skill_gaps_from_violations[skill]
            
            if need_count > 0:
                preferred_coverage = (preferred_shifts / need_count * 100) if need_count > 0 else 0
                maximum_coverage = (maximum_shifts / need_count * 100) if need_count > 0 else 0
                
                if gap_count > 0:
                    skill_analysis_lines.append(
                        f"  🚨 {skill}: {preferred_shifts} preferred available-shifts / {need_count} shift-needs ({preferred_coverage:.0f}% coverage) | Max capacity: {maximum_shifts} shifts ({maximum_coverage:.0f}%) - {gap_count} gaps in schedule"
                    )
                elif preferred_coverage < 50:
                    skill_analysis_lines.append(
                        f"  ⚠ {skill}: {preferred_shifts} preferred available-shifts / {need_count} shift-needs ({preferred_coverage:.0f}% coverage) | Max capacity: {maximum_shifts} shifts ({maximum_coverage:.0f}%) - LOW"
                    )
                elif preferred_coverage >= 150:
                    skill_analysis_lines.append(
                        f"  ✓ {skill}: {preferred_shifts} preferred available-shifts / {need_count} shift-needs ({preferred_coverage:.0f}% coverage) | Max capacity: {maximum_shifts} shifts ({maximum_coverage:.0f}%) - OVERSUPPLIED"
                    )
                else:
                    skill_analysis_lines.append(
                        f"  ✓ {skill}: {preferred_shifts} preferred available-shifts / {need_count} shift-needs ({preferred_coverage:.0f}% coverage) | Max capacity: {maximum_shifts} shifts ({maximum_coverage:.0f}%) - ADEQUATE"
                    )
            elif preferred_shifts > 0:
                skill_analysis_lines.append(
                    f"  ℹ {skill}: {preferred_shifts} preferred available-shifts but NOT REQUIRED by any patient - consider reassignment"
                )
        
        skill_analysis_text = '\n'.join(skill_analysis_lines) if skill_analysis_lines else "  - No skill data available"
        
        # ===== LEVEL ANALYSIS =====
        # Count available shifts by level (not just employees)
        employee_level_shifts = defaultdict(int)
        for emp in employees:
            level = emp.get('level', 1)
            available_shifts = len(emp.get('available_shifts', []))
            employee_level_shifts[level] += available_shifts
        
        # Count patient level requirements
        patient_level_needs = defaultdict(int)
        for patient in patients:
            min_level = patient.get('min_level', 1)
            shift_count = len(patient.get('care_shifts', []))
            patient_level_needs[min_level] += shift_count
        
        # Analyze level gaps from violations
        level_gaps_from_violations = defaultdict(int)
        for v in violations:
            if 'requires level' in v.lower():
                match = re.search(r'requires level (\d)', v)
                if match:
                    level = int(match.group(1))
                    level_gaps_from_violations[level] += 1
        
        # Build level analysis
        level_analysis_lines = []
        for level in sorted(set(employee_level_shifts.keys()) | set(patient_level_needs.keys())):
            level_shifts = employee_level_shifts[level]
            need_count = patient_level_needs.get(level, 0)
            gap_count = level_gaps_from_violations[level]
            
            # For level requirements, Level 3 can cover Level 2 and 1, etc.
            # So we need to count "qualified" nurse available-shifts
            qualified_shifts = sum(employee_level_shifts[l] for l in employee_level_shifts if l >= level)
            
            if need_count > 0:
                coverage = (qualified_shifts / need_count * 100) if need_count > 0 else 0
                if gap_count > 0:
                    level_analysis_lines.append(
                        f"  🚨 Level {level}+: {qualified_shifts} qualified available-shifts / {need_count} shift-needs ({coverage:.0f}% coverage) - {gap_count} gaps"
                    )
                elif coverage < 50:
                    level_analysis_lines.append(
                        f"  ⚠ Level {level}+: {qualified_shifts} qualified available-shifts / {need_count} shift-needs ({coverage:.0f}% coverage) - INSUFFICIENT"
                    )
                elif coverage >= 200:
                    level_analysis_lines.append(
                        f"  ✓ Level {level}+: {qualified_shifts} qualified available-shifts / {need_count} shift-needs ({coverage:.0f}% coverage) - OVERQUALIFIED"
                    )
                else:
                    level_analysis_lines.append(
                        f"  ✓ Level {level}+: {qualified_shifts} qualified available-shifts / {need_count} shift-needs ({coverage:.0f}% coverage) - ADEQUATE"
                    )
            else:
                level_analysis_lines.append(
                    f"  ℹ Level {level}: {level_shifts} available-shifts but no patients require this level"
                )
        
        level_analysis_text = '\n'.join(level_analysis_lines) if level_analysis_lines else "  - No level data available"
        
        # ===== VIOLATION BREAKDOWN =====
        availability_violations = sum(1 for v in violations if 'not available' in v.lower())
        hour_violations = sum(1 for v in violations if 'scheduled for' in v.lower() and 'h/week' in v.lower())
        conflict_violations = sum(1 for v in violations if 'cannot work together' in v.lower())
        skill_violations = sum(1 for v in violations if 'requires skills' in v.lower())
        level_violations = sum(1 for v in violations if 'requires level' in v.lower())
        coverage_violations = sum(1 for v in violations if 'needs care' in v.lower() or ('needs' in v.lower() and 'nurses but' in v.lower()))
        
        # ===== TOTAL SHIFTS CONTEXT =====
        total_shifts_available = sum(len(e.get('available_shifts', [])) for e in employees)
        availability_violation_rate = (availability_violations / total_shifts_available * 100) if total_shifts_available > 0 else 0
        
        # ===== OVERALL STATS =====
        total_patient_hours = sum(len(p.get('care_shifts', [])) * p.get('nurses_needed', 1) * 4 for p in patients) / 4
        total_employee_hours = sum(len(e.get('available_shifts', [])) * 4 for e in employees) / 4
        
        # ===== EMPLOYEE COMPOSITION SUMMARY =====
        total_employees = len(employees)
        full_time = sum(1 for e in employees if e.get('max_hours_per_week', 0) >= 40)
        part_time = total_employees - full_time
        
        avg_pay_rate = sum(e.get('hourly_pay_rate', 0) for e in employees) / total_employees if total_employees > 0 else 0
        
        context = f"""You are a healthcare staffing consultant analyzing a nursing facility schedule.

FACILITY OVERVIEW:
- Total Employees: {total_employees} ({full_time} full-time, {part_time} part-time)
- Average pay rate: ${avg_pay_rate:.2f}/hr
- Total Patients: {len(patients)}
- Total Violations: {len(violations)} ({coverage_violations} coverage, {skill_violations} skill, {level_violations} level, {availability_violations} availability, {hour_violations} hours, {conflict_violations} conflicts)

VIOLATION SEVERITY GUIDE:
🔴 CRITICAL (Patient Safety/Legal Issues - Immediate Risk):
  - Coverage violations ({coverage_violations}): Patients without required nurses - IMMEDIATE PATIENT SAFETY RISK
  - Skill violations ({skill_violations}): Nurses lack required clinical skills - MEDICAL CARE COMPROMISED
  - Level violations ({level_violations}): Nurses below required experience level - CARE QUALITY RISK
  - Unknown patient/employee: System errors with invalid IDs - DATA INTEGRITY FAILURE
  - Patient scheduled for unneeded care: Wrong care timing - OPERATIONAL FAILURE
  - System errors (too many nurses per patient): Over maximum limits - RESOURCE WASTE

🟡 MODERATE (Operational Issues - Business Impact):
  - MAX HOURS EXCEEDED: Employees over legal weekly limits - LABOR LAW VIOLATION  
  - Hour violations ({hour_violations}): Employees working over preferred but under max hours - BURNOUT RISK
  - Minimum hours not met: Employees below preferred minimum - CONTRACT VIOLATION
  - Conflict violations ({conflict_violations}): Incompatible employees working together - WORKPLACE DISRUPTION
  - Load imbalance: Some employees idle while others overworked - INEFFICIENT RESOURCE USE
  - Consecutive shift violations: More than 3 shifts in a row - EMPLOYEE WELLBEING RISK

🟢 MINOR (Preference Issues - Non-urgent):  
  - Availability violations ({availability_violations}): Nurses scheduled when not available - PREFERENCE ONLY
    (Rate: {availability_violation_rate:.1f}% of {total_shifts_available} total shifts - {'HIGH' if availability_violation_rate > 20 else 'MODERATE' if availability_violation_rate > 10 else 'LOW'} impact)

HOURS ANALYSIS:
- Weekly patient care hours needed: {total_patient_hours:.0f}h
- Weekly employee hours available: {total_employee_hours:.0f}h
- Coverage ratio: {(total_employee_hours/total_patient_hours)*100:.0f}%

SHIFT-BY-SHIFT STAFFING (Supply vs Demand):
{shift_analysis_text}

SKILL INVENTORY & GAPS (Dual Metrics Analysis):
IMPORTANT: Each skill shows TWO metrics - preferred availability vs maximum theoretical capacity:
- PREFERRED: When employees actually want to work (reduces availability violations)
- MAX CAPACITY: Theoretical 24/7 availability (emergency capacity only)
Focus on PREFERRED coverage for operational planning. Max capacity is strategic backup only.

{skill_analysis_text}

EXPERIENCE LEVEL COMPOSITION:
{level_analysis_text}

TASK: Write a comprehensive 4-paragraph analysis for facility leadership:

Paragraph 1 - EXECUTIVE SUMMARY (3-4 sentences):
Using the violation severity guide above, assess schedule viability. Focus on CRITICAL violations (coverage/skill/level) as these affect patient safety. MODERATE violations create operational challenges. MINOR violations (hour preferences) are not urgent. Is this schedule operationally viable for patient care? What's the most serious issue?

Paragraph 2 - SHIFT TIMING & AVAILABILITY ANALYSIS (4-5 sentences):
Based on the shift-by-shift analysis above, which specific time blocks (morning/afternoon/evening/night) are adequately staffed vs understaffed? Focus on "scheduled vs needed" ratios for actual coverage. Also analyze the "available" numbers to identify strategic opportunities: which shifts have high employee availability (easy to fill) vs low availability (need hiring incentives)? For understaffed shifts, is the issue lack of available employees or poor scheduling efficiency? What operational changes could help (shift incentives for low-availability periods, better utilization of high-availability periods)?

Paragraph 3 - SKILLS & EXPERIENCE COMPOSITION (4-5 sentences):
Analyze the skill inventory with specific dual metrics data. Quote exact coverage percentages from the analysis above using the format "Skill: X preferred available-shifts / Y shift-needs (Z% coverage) | Max capacity: W shifts (V%)". The PREFERRED availability metrics are more important because they represent when employees actually want to work, leading to fewer availability violations. The MAX capacity shows theoretical 24/7 availability. For each skill, explain whether the preferred coverage is adequate (>100%) or if there are gaps requiring attention. For experience levels, use the qualified nurse counts (Level 3+ can work Level 1-2 assignments) and provide specific coverage ratios. Be precise about which skills have adequate preferred coverage vs those needing strategic hiring to match employee preferences with operational needs.

Paragraph 4 - SPECIFIC HIRING RECOMMENDATIONS (4-6 action items):
Provide concrete, prioritized hiring recommendations in this format:
"1. [CRITICAL/HIGH/MEDIUM] Hire [NUMBER] [FULL-TIME/PART-TIME] Level [X] nurses with [SKILLS] for [SPECIFIC SHIFTS]"
Base recommendations on the gaps identified above. Include rationale for each recommendation (which specific gap it addresses).

Be direct, data-driven, and action-oriented. Use specific numbers and shift times from the analysis provided above. Quote exact percentages and coverage ratios rather than vague terms like "significant shortages." If a skill shows 150% coverage, call it "oversupplied," not "short." Prioritize patient safety and operational efficiency.

FORMAT: Return your analysis as clean HTML with proper structure:
- Use <h3> tags for paragraph headings
- Use <p> tags for paragraph content  
- Use <ul> and <li> for recommendation lists
- Use <strong> for emphasis on key metrics
- Use color coding: <span style="color: #d73502"> for critical issues, <span style="color: #f57c00"> for moderate issues, <span style="color: #388e3c"> for positive findings
- Include the robot emoji 🤖 in the title"""

        print("\n🤖 Generating comprehensive AI staffing narrative...")
        response = llm.invoke(context)
        print("✓ AI narrative generated successfully")
        
        # Clean up the response content
        content = response.content.strip()
        
        # Convert markdown-style headers to HTML with better styling
        content = content.replace('**Executive Summary**', '<h3 style="color: #2d3748; margin-top: 25px; margin-bottom: 15px;">📊 Executive Summary</h3>')
        content = content.replace('**Shift Timing & Availability Analysis**', '<h3 style="color: #2d3748; margin-top: 25px; margin-bottom: 15px;">⏰ Shift Timing & Availability Analysis</h3>')
        content = content.replace('**Skills & Experience Composition**', '<h3 style="color: #2d3748; margin-top: 25px; margin-bottom: 15px;">🎯 Skills & Experience Composition</h3>')
        content = content.replace('**Specific Hiring Recommendations**', '<h3 style="color: #2d3748; margin-top: 25px; margin-bottom: 15px;">📋 Specific Hiring Recommendations</h3>')
        
        # Also handle any other bold text patterns
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        
        # Convert numbered lists to proper HTML
        lines = content.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    formatted_lines.append('</ol>')
                    in_list = False
                # Skip adding empty lines to reduce spacing
                continue
                
            # Check if line starts with a number (numbered list) or bullet point
            if re.match(r'^\d+\.', line) or re.match(r'^\*\*\w+:\*\*', line):
                if not in_list:
                    formatted_lines.append('<ol style="margin: 15px 0; padding-left: 20px;">')
                    in_list = True
                # Extract content after number or format bold priorities
                if re.match(r'^\d+\.', line):
                    list_content = line[line.find(".")+1:].strip()
                else:
                    list_content = line
                formatted_lines.append(f'<li style="margin: 8px 0; line-height: 1.5;">{list_content}</li>')
            else:
                if in_list:
                    formatted_lines.append('</ol>')
                    in_list = False
                if line.startswith('<h3'):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(f'<p style="margin: 12px 0; line-height: 1.6;">{line}</p>')
        
        if in_list:
            formatted_lines.append('</ol>')
        
        content = '\n'.join(formatted_lines)
        
        # Simple wrapper with clean styling (no duplicate header since it's handled by the visualization)
        html_content = f"""<div style="background: #f8f9fa; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;">
{content}
</div>"""
        
        return html_content
        
    except Exception as e:
        print(f"⚠ Warning: Could not generate staffing narrative: {e}")
        import traceback
        traceback.print_exc()
        return None

# Pydantic models for tool inputs (required by LangChain)

class ValidateScheduleInput(BaseModel):
    """Input for schedule validation - leave empty to validate last generated schedule"""
    schedule: Optional[Union[Dict, str]] = Field(default=None, description="Schedule to validate (optional - if not provided, validates the last generated schedule)")


class ScoreScheduleInput(BaseModel):
    """Input for schedule scoring"""
    schedule: Optional[Union[Dict, str]] = Field(default=None, description="Schedule to score (optional - if not provided, scores the last generated schedule)")
    cost_weight: float = Field(default=1.0, description="Weight for cost objective")
    continuity_weight: float = Field(default=2.0, description="Weight for continuity objective")
    fairness_weight: float = Field(default=1.0, description="Weight for fairness objective")
    overtime_weight: float = Field(default=1.5, description="Weight for overtime objective")


class GenerateScheduleInput(BaseModel):
    """Input for schedule generation"""
    strategy: str = Field(default="greedy", description="Strategy: 'greedy' | 'iterative' | 'random'")
    seed_schedule: Optional[Union[Dict, str]] = Field(default=None, description="Existing schedule to improve (for iterative)")
    previous_violations: Optional[List[str]] = Field(default=None, description="Violations to fix (for iterative)")


class AnalyzeStaffingInput(BaseModel):
    """Input for staffing analysis"""
    schedule: Optional[Union[Dict, str]] = Field(default=None, description="Schedule to analyze (optional - if not provided, analyzes last generated schedule)")
    violations: List[str] = Field(description="List of constraint violations from validation")


class CompareSchedulesInput(BaseModel):
    """Input for comparing two schedules"""
    old_patient_violations: int = Field(description="Patient violations from old schedule")
    old_employee_violations: int = Field(description="Employee violations from old schedule")
    new_patient_violations: int = Field(description="Patient violations from new schedule")
    new_employee_violations: int = Field(description="Employee violations from new schedule")


class NursingSchedulerAgent:
    """
    Intelligent agent for nursing shift scheduling optimization.
    
    Uses Google Gemini with 5 specialized tools:
    1. Constraint Validator
    2. Schedule Scorer  
    3. Schedule Generator
    4. Staffing Analyzer
    5. Schedule Comparator
    """
    
    def __init__(self, employees_file: str = "data/employees.json", 
                 patients_file: str = "data/patients.json",
                 config=None):
        """
        Initialize agent with employee and patient data.
        
        Args:
            employees_file: Path to employees.json
            patients_file: Path to patients.json
            config: Configuration object (optional)
        """
        # Load configuration if not provided
        if config is None:
            from .config import ConfigManager
            config = ConfigManager().config
        
        self.config = config
        
        # Load data
        with open(employees_file, 'r') as f:
            self.employees = json.load(f)
        
        with open(patients_file, 'r') as f:
            self.patients = json.load(f)
        
        # Initialize tool classes
        self.validator = ConstraintValidator(self.employees, self.patients)
        self.scorer = ScheduleScorer(self.employees, self.patients)
        self.generator = ScheduleGenerator(self.employees, self.patients, self.config)
        self.staffing_analyzer = StaffingAnalyzer(self.employees, self.patients)
        
        # Add rate limiter
        self.rate_limiter = RateLimiter(calls_per_minute=9)  # Stay under 10/min
        
        # Setup LangChain tools
        self.tools = self._create_tools()
        
        # Setup LLM with LangSmith tracing
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Use Gemini 2.5 Flash for fast reasoning and tool calling
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,  # Deterministic for scheduling
            google_api_key=api_key
        )
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        # Store last generated schedule for reference
        self.last_schedule = None
        
        # NEW: Track best schedule (patient=0, lowest employee violations)
        self.best_schedule = None
        self.best_patient_violations = 999
        self.best_employee_violations = 999
    
    def _create_tools(self) -> List[StructuredTool]:
        """Create LangChain tools from our implementations"""
        
        def validate_schedule_tool(schedule: Optional[Union[Dict, str]] = None) -> str:
            """
            Validates a nursing schedule against all hard constraints.
            
            If no schedule provided, validates the last generated schedule.
            
            Checks:
            - All patients have required coverage
            - Nurses have required skills and levels
            - Availability constraints respected
            - Exclusion rules (employee-patient, employee-employee) followed
            - Hour limits (min/max per week) satisfied
            - Consecutive shift limits (max 3 in a row) respected
            
            Returns validation result with CRITICAL patient violations separated from employee violations.
            """
            self.rate_limiter.wait_if_needed()  # Rate limit API calls
            # Use last generated schedule if none provided
            if schedule is None or schedule == 'last_generated_schedule' or schedule == '':
                schedule = self.last_schedule
            
            # Handle if schedule is passed as string (parse it)
            if isinstance(schedule, str) and schedule:
                try:
                    schedule = json.loads(schedule)
                except:
                    schedule = self.last_schedule
            
            if not schedule:
                return json.dumps({
                    "valid": False,
                    "message": "No schedule available to validate",
                    "patient_violations": 0,
                    "employee_violations": 0,
                    "total_violations": 0
                })
            
            is_valid, violations, breakdown = self.validator.validate(schedule)
            
            if is_valid:
                return json.dumps({
                    "valid": True,
                    "message": "✓ Schedule satisfies all constraints!",
                    "patient_violations": 0,
                    "employee_violations": 0,
                    "total_violations": 0
                })
            else:
                # Categorize violations for reporting
                violation_types = {}
                for v in violations:
                    if "requires level" in v and "but highest assigned" in v:
                        violation_types['level_mismatch'] = violation_types.get('level_mismatch', 0) + 1
                    elif "requires skills" in v:
                        violation_types['skill_mismatch'] = violation_types.get('skill_mismatch', 0) + 1
                    elif "needs care during" in v and "no nurses" in v:
                        violation_types['no_coverage'] = violation_types.get('no_coverage', 0) + 1
                    elif "cannot work together" in v:
                        violation_types['employee_conflict'] = violation_types.get('employee_conflict', 0) + 1
                    elif "scheduled for" in v and "but min is" in v:
                        violation_types['min_hours'] = violation_types.get('min_hours', 0) + 1
                    elif "scheduled for" in v and "but max is" in v:
                        violation_types['max_hours'] = violation_types.get('max_hours', 0) + 1
                    elif "not available for" in v:
                        violation_types['availability'] = violation_types.get('availability', 0) + 1
                    elif "consecutive shifts" in v:
                        violation_types['consecutive'] = violation_types.get('consecutive', 0) + 1
                    else:
                        violation_types['other'] = violation_types.get('other', 0) + 1
                
                priority_status = ""
                if breakdown['patient'] > 0:
                    priority_status = " | 🚨 PRIORITY: Reduce patient violations!"
                elif breakdown['employee'] > 0:
                    priority_status = f" | ✓ Patient=0, now optimize employee violations (Current: {breakdown['employee']})"
                else:
                    priority_status = " | ✓ PERFECT SCHEDULE!"
                
                # NEW: Track best schedule
                if breakdown['patient'] == 0:
                    if self.best_schedule is None or breakdown['employee'] < self.best_employee_violations:
                        print(f"  ✨ NEW BEST: P=0, E={breakdown['employee']} (previous best: E={self.best_employee_violations})")
                        self.best_schedule = copy.deepcopy(schedule)
                        self.best_patient_violations = 0
                        self.best_employee_violations = breakdown['employee']

                return json.dumps({
                    "valid": False,
                    "message": f"✗ PATIENT: {breakdown['patient']} | EMPLOYEE: {breakdown['employee']}{priority_status}",
                    "patient_violations": breakdown['patient'],
                    "employee_violations": breakdown['employee'],
                    "total_violations": breakdown['total'],
                    "by_type": violation_types,
                    "critical_note": "Patient violations MUST be 0 for legal/safety compliance",
                    "best_so_far": f"P={self.best_patient_violations}, E={self.best_employee_violations}" if self.best_schedule else "None yet"
                })
        
        def score_schedule_tool(schedule: Optional[Union[Dict, str]] = None,
                               cost_weight: float = 1.0,
                               continuity_weight: float = 2.0,
                               fairness_weight: float = 1.0,
                               overtime_weight: float = 1.5) -> str:
            """
            Scores a schedule on soft optimization objectives.
            
            If no schedule provided, scores the last generated schedule.
            
            Objectives:
            - Cost: Total labor cost (hours × pay rates)
            - Continuity: Number of unique nurses per patient (lower is better)
            - Fairness: Even distribution of hours across employees
            - Overtime: Penalty for exceeding 40 hours/week
            
            Returns score breakdown and total weighted score.
            """
            self.rate_limiter.wait_if_needed()  # Rate limit API calls
            
            # Use last generated schedule if none provided
            if schedule is None or schedule == '':
                schedule = self.last_schedule
            
            # Handle if schedule is passed as string (parse it)
            if isinstance(schedule, str) and schedule:
                try:
                    schedule = json.loads(schedule)
                except:
                    schedule = self.last_schedule
            
            if not schedule:
                return json.dumps({"error": "No schedule available to score"})
            
            weights = {
                'cost': cost_weight,
                'continuity': continuity_weight,
                'fairness': fairness_weight,
                'overtime': overtime_weight
            }
            result = self.scorer.score(schedule, weights)
            return json.dumps(result)
        
        def generate_schedule_tool(strategy: str = "greedy", 
                                  seed_schedule: Union[Dict, str, None] = None,
                                  previous_violations: List[str] = None) -> str:
            """
            Generates a candidate schedule using specified strategy.
            
            Strategies:
            - 'greedy': Initial schedule - START HERE
            - 'iterative': Improve existing schedule - automatically uses last schedule and recent violations
            
            Returns metadata about the generated schedule.
            """
            self.rate_limiter.wait_if_needed()  # Rate limit API calls
            
            # For iterative, automatically use last schedule
            if strategy == "iterative":
                seed_schedule = self.last_schedule
                # Get violations from last validation to guide iterative improvement
                if self.last_schedule:
                    _, full_violations, _ = self.validator.validate(self.last_schedule)
                    # Only pass the actual violation strings (needed by iterative parser)
                    previous_violations = full_violations
                else:
                    previous_violations = []
            
            # Handle string inputs
            if isinstance(seed_schedule, str) and seed_schedule:
                try:
                    seed_schedule = json.loads(seed_schedule)
                except:
                    seed_schedule = self.last_schedule
            
            result = self.generator.generate(strategy, seed_schedule, previous_violations)
            self.last_schedule = result['schedule']  # Store for reference
            
            # Don't return full schedule in tool output (too large)
            return json.dumps({
                "status": "success",
                "metadata": result['metadata'],
                "message": f"Generated {result['metadata']['num_assignments']} assignments using '{strategy}'"
            })
        
        def analyze_staffing_tool(schedule: Optional[Union[Dict, str]] = None, violations: List[str] = None) -> str:
            """
            Analyzes staffing needs based on schedule violations and employee utilization.
            
            If no schedule provided, analyzes the last generated schedule.
            
            Provides actionable recommendations:
            - Which skills/levels to hire for
            - Overworked employees (burnout risk)
            - Underutilized employees (termination candidates)
            - Skill gap analysis
            
            Use this after validation to understand staffing improvements needed.
            """
            self.rate_limiter.wait_if_needed()  # Rate limit API calls
            
            # Use last generated schedule if none provided
            if schedule is None or schedule == 'last_generated_schedule' or schedule == '':
                schedule = self.last_schedule
            
            # Handle if schedule is passed as string
            if isinstance(schedule, str) and schedule:
                try:
                    schedule = json.loads(schedule)
                except:
                    schedule = self.last_schedule
            
            if not schedule:
                return json.dumps({"error": "No schedule available to analyze"})
            
            if violations is None:
                violations = []
            
            result = self.staffing_analyzer.analyze(schedule, violations)
            
            summary = {
                "hiring_recommendations": result['hiring_recommendations'][:5],
                "termination_candidates": result['termination_candidates'][:3],
                "overworked_employees": result['overworked_employees'][:5],
                "skill_gaps": dict(list(result['skill_gaps'].items())[:5]),
                "total_uncovered_shifts": result['total_uncovered_shifts'],
                "affected_patients": result['affected_patients']
            }
            
            return json.dumps(summary)
        
        def compare_schedules_tool(
            old_patient_violations: int,
            old_employee_violations: int,
            new_patient_violations: int,
            new_employee_violations: int
        ) -> str:
            """
            Compare two schedules and determine which is better.
            
            ALWAYS prioritizes patient violations over employee violations.
            Returns recommendation: "keep_new", "keep_old", or "equal"
            """
            self.rate_limiter.wait_if_needed()  # Rate limit API calls
            
            # Priority 1: Patient violations
            if new_patient_violations < old_patient_violations:
                return json.dumps({
                    "recommendation": "keep_new",
                    "reason": f"Patient violations improved: {old_patient_violations} → {new_patient_violations}",
                    "patient_delta": old_patient_violations - new_patient_violations,
                    "employee_delta": old_employee_violations - new_employee_violations,
                    "priority": "PATIENT_IMPROVED",
                    "continue_iterating": True,
                    "note": "Patient violations still exist - MUST continue!" if new_patient_violations > 0 else "Patient=0 reached! Now optimize employee violations"
                })
            elif new_patient_violations > old_patient_violations:
                return json.dumps({
                    "recommendation": "keep_old",
                    "reason": f"Patient violations got worse: {old_patient_violations} → {new_patient_violations}",
                    "patient_delta": old_patient_violations - new_patient_violations,
                    "employee_delta": old_employee_violations - new_employee_violations,
                    "priority": "PATIENT_REGRESSED",
                    "continue_iterating": True,
                    "note": "Patient violations increased - MUST continue iterating!"
                })
            
            # Priority 2: Patient violations are equal
            if old_patient_violations == 0 and new_patient_violations == 0:
                # Both perfect on patient side, optimize employee
                if new_employee_violations < old_employee_violations:
                    return json.dumps({
                        "recommendation": "keep_new",
                        "reason": f"Patient=0 for both, employee violations improved: {old_employee_violations} → {new_employee_violations}",
                        "patient_delta": 0,
                        "employee_delta": old_employee_violations - new_employee_violations,
                        "priority": "EMPLOYEE_IMPROVED",
                        "continue_iterating": True,
                        "note": "Patient=0 maintained, employee violations improved - continue optimizing!"
                    })
                elif new_employee_violations > old_employee_violations:
                    return json.dumps({
                        "recommendation": "keep_old",
                        "reason": f"Patient=0 for both, employee violations got worse: {old_employee_violations} → {new_employee_violations}",
                        "patient_delta": 0,
                        "employee_delta": old_employee_violations - new_employee_violations,
                        "priority": "EMPLOYEE_REGRESSED",
                        "continue_iterating": True,
                        "note": "Patient=0 maintained, but employee violations worsened - try 1-2 more iterations"
                    })
                else:
                    return json.dumps({
                        "recommendation": "equal",
                        "reason": "No change in violations",
                        "patient_delta": 0,
                        "employee_delta": 0,
                        "priority": "NO_CHANGE",
                        "continue_iterating": False,
                        "note": "No improvement in employee violations - consider stopping soon"
                    })
            else:
                # Same patient violations (both > 0), compare employee
                if new_employee_violations < old_employee_violations:
                    return json.dumps({
                        "recommendation": "keep_new",
                        "reason": f"Patient violations same ({old_patient_violations}), employee improved: {old_employee_violations} → {new_employee_violations}",
                        "patient_delta": 0,
                        "employee_delta": old_employee_violations - new_employee_violations,
                        "priority": "EMPLOYEE_IMPROVED_PATIENT_STUCK",
                        "continue_iterating": True,
                        "note": f"Patient violations stuck at {old_patient_violations} - MUST continue iterating!"
                    })
                else:
                    return json.dumps({
                        "recommendation": "keep_old",
                        "reason": f"No improvement (patient violations stuck at {old_patient_violations})",
                        "patient_delta": 0,
                        "employee_delta": old_employee_violations - new_employee_violations,
                        "priority": "NO_IMPROVEMENT",
                        "continue_iterating": True,
                        "note": f"Patient violations stuck at {old_patient_violations} - MUST continue iterating!"
                    })
        
        # Create StructuredTools
        tools = [
            StructuredTool.from_function(
                func=validate_schedule_tool,
                name="validate_schedule",
                description="Validate the last generated schedule against all hard constraints. No input needed - automatically validates most recent schedule.",
                args_schema=ValidateScheduleInput
            ),
            StructuredTool.from_function(
                func=score_schedule_tool,
                name="score_schedule",
                description="Score a schedule on soft optimization objectives (cost, continuity of care, fairness, overtime)",
                args_schema=ScoreScheduleInput
            ),
            StructuredTool.from_function(
                func=generate_schedule_tool,
                name="generate_schedule",
                description="Generate a candidate schedule using greedy or random strategy",
                args_schema=GenerateScheduleInput
            ),
            StructuredTool.from_function(
                func=analyze_staffing_tool,
                name="analyze_staffing",
                description="Analyze staffing needs based on violations - automatically uses last generated schedule. Provides hiring recommendations and identifies over/underutilized staff.",
                args_schema=AnalyzeStaffingInput
            ),
            StructuredTool.from_function(
                func=compare_schedules_tool,
                name="compare_schedules",
                description="Compare two schedules based on patient and employee violations. Always prioritizes patient violations. Returns which schedule to keep.",
                args_schema=CompareSchedulesInput
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools"""
        
        # System prompt for the agent
        system_prompt = """You are an expert nursing shift scheduler. Your goal is to create optimal schedules for a nursing facility and MINIMIZE violations.

You have access to data about:
- {num_employees} employees with varying skills, levels, availability, and pay rates
- {num_patients} patients with specific care requirements

You have 5 specialized tools:
1. generate_schedule: Create a candidate schedule using heuristics (greedy/iterative)
2. validate_schedule: Check if a schedule satisfies all hard constraints
3. compare_schedules: Compare two schedules to decide which is better
4. analyze_staffing: Analyze violations and provide hiring/firing recommendations
5. score_schedule: Evaluate a schedule on optimization objectives

COMPLETE 4-PHASE WORKFLOW:

PHASE 1 - INITIAL GENERATION (Iteration 1):
→ generate_schedule(strategy='greedy')
→ validate_schedule
→ IF patient_violations = 0: Go to PHASE 3a (verification)
→ IF patient_violations > 0: Go to PHASE 2

PHASE 2 - ITERATIVE IMPROVEMENT (Iterations 2-3):
→ generate_schedule(strategy='iterative') [max 2 times]
→ validate_schedule
→ compare_schedules
→ IF patient_violations = 0: Go to PHASE 3a
→ IF patient_violations > 0 after 2 tries: Go to PHASE 2b

PHASE 2b - CONTINUED ITERATIVE IMPROVEMENT (If needed):
If patient_violations > 0 after Phase 2:
→ Continue with generate_schedule(strategy='iterative')
→ validate_schedule after each
→ compare_schedules(old vs new)
→ Keep iterating until patient_violations = 0
→ Maximum 8 total iterations across Phase 2 and 2b

If patient_violations still > 0 after 8 iterations:
→ Proceed to Phase 3c
→ In final response, explain this indicates fundamental staffing shortage
→ analyze_staffing will provide hiring recommendations

PHASE 3a - VERIFY SUCCESS:
→ validate_schedule (confirm patient_violations = 0)
→ Continue to PHASE 3b

PHASE 3b - EMPLOYEE VIOLATION OPTIMIZATION (MANDATORY):
Once patient_violations = 0, optimize employee violations:

QUALITY CHECK DURING PHASE 3b:
When comparing schedules, don't just look at violation counts.
Check the validation breakdown:
- IF max_hours violations > 20: Load imbalance problem  
- IF availability violations > 100: Scheduling conflict problem
- Prefer schedules with BALANCED violation types over concentrated ones

REQUIRED STEPS (5-6 optimization rounds):
1. generate_schedule(strategy='iterative')
2. validate_schedule → get P and E counts
3. compare_schedules(old_patient, old_employee, new_patient, new_employee)
4. Based on compare_schedules response:
   - IF recommendation='keep_new' AND new_patient=0: Accept, update baseline
   - IF recommendation='keep_old' OR new_patient>0: Reject, keep old
5. Repeat steps 1-4 for 5-6 rounds total
6. Stop when: Employee violations don't improve for 2 consecutive rounds

CRITICAL:
- NEVER allow patient_violations to return (must stay 0)
- MUST call compare_schedules after EACH iterative attempt
- Continue for at least 5 rounds even if no improvement

Typical outcome: Employee violations reduce from ~220 to ~110

PHASE 3c - FINAL REPORTING:
→ analyze_staffing + score_schedule

Total expected tool calls: 12-15 depending on complexity

Expected full sequence:
Phase 1: generate_schedule(greedy) → validate
Phase 2: iterative × 2-3 → validate + compare each time
Phase 3a: validate (confirm P=0)
Phase 3b: iterative × 5-6 → validate + compare_schedules each time
Phase 3c: analyze_staffing → score_schedule → final response
Typical tool count: 12-15 calls total

Key optimization priorities (in order):
1. MINIMIZE PATIENT VIOLATIONS = 0 (absolute top priority - legal/safety!)
   - Patient coverage (all shifts staffed)
   - Skill requirements met
   - Level requirements met
   - Exclusion rules (safety)
2. BALANCE WORKLOAD ACROSS ALL EMPLOYEES (critical for fairness)
   - NEVER schedule anyone beyond their max_hours (hard safety limit)
   - Strongly prefer underutilized employees (0-10h/week)
   - Avoid concentrating work on few employees
   - Goal: Every employee should work, none should exceed max capacity
   - Target: 80% of employees working 70-100% of expected hours
   - Red flags: Some at max_hours while others at 0h/week = BAD scheduling
3. MINIMIZE OTHER EMPLOYEE VIOLATIONS (optimization targets)
   - Availability preferences
   - Min hours requirements
   - Consecutive shift limits
4. Continuity of Care: Minimize unique nurses per patient
5. Cost: Minimize total labor cost
6. Fairness: Balance hours across employees

CRITICAL: If patient coverage requires exceeding an employee's max_hours, DO IT.
Patient safety > employee preferences. We'll address staffing shortages through hiring.

When iterating, you can call generate_schedule with "iterative" up to 5 times, but ONLY continue if patient_violations keep decreasing.

FINAL REMINDER BEFORE RESPONDING TO USER:
- Did violations exist at any point? 
- If YES: Did you call analyze_staffing? 
- If NO: STOP and call analyze_staffing NOW before giving final response
- Your response MUST include staffing recommendations if any violations were found
"""
        
        # Replace placeholders manually to avoid template conflicts with JSON examples
        formatted_prompt = system_prompt.replace("{num_employees}", str(len(self.employees)))
        formatted_prompt = formatted_prompt.replace("{num_patients}", str(len(self.patients)))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Create executor with minimal output to save tokens
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,  # Turn off verbose to save tokens
            max_iterations=25,  # Increased from 15
            handle_parsing_errors=True,
            return_intermediate_steps=False,  # Don't return steps to save tokens
            early_stopping_method="generate"  # Generate a response even if not finished
        )
        
        return agent_executor
    
    def get_validation_breakdown(self) -> Dict:
        """Get validation breakdown for the current schedule"""
        if not self.last_schedule:
            return {'patient': 999, 'employee': 999, 'total': 999}
        
        _, violations, breakdown = self.validator.validate(self.last_schedule)
        return breakdown
    
    def run(self, query: str) -> Dict:
        """
        Run the agent on a query.
        
        Args:
            query: Natural language request (e.g., "Create an optimal schedule for next month")
        
        Returns:
            Dict with agent's response and metadata
        """
        try:
            # Apply rate limiting before any LLM interaction
            self.rate_limiter.wait_if_needed()
            result = self.agent_executor.invoke({"input": query})
            return result
        except ValueError as e:
            if "No generation chunks" in str(e):
                print("\n⚠️ LLM returned empty response (likely hit quota)")
                print("But your schedule was generated successfully!")
                # Return partial result with what we have
                return {
                    "output": "Schedule generated but LLM quota exceeded during final response",
                    "intermediate_steps": []
                }
            else:
                raise
        except Exception as e:
            # Handle Google's quota exceeded errors
            if "ResourceExhausted" in str(type(e)) or "429" in str(e) or "quota" in str(e).lower():
                print(f"\n🚫 Google API Quota Exceeded")
                print(f"Error: {str(e)}")
                
                # Try to extract wait time from error message
                import re
                wait_match = re.search(r'retry in ([\d.]+)s', str(e))
                if wait_match:
                    wait_time = float(wait_match.group(1))
                    print(f"⏳ Google suggests waiting {wait_time:.1f} seconds")
                    
                    if wait_time < 300:  # Only wait if less than 5 minutes
                        print(f"⏳ Waiting {wait_time:.1f}s and retrying...")
                        time.sleep(wait_time + 1)  # Add 1 second buffer
                        try:
                            result = self.agent_executor.invoke({"input": query})
                            return result
                        except Exception as retry_e:
                            print(f"⚠️ Retry failed: {retry_e}")
                
                # Return partial result if we can't retry
                return {
                    "output": f"Quota exceeded. Schedule partially generated. Error: {str(e)[:200]}...",
                    "intermediate_steps": getattr(self, '_last_intermediate_steps', [])
                }
            else:
                raise
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the data"""
        total_patient_shifts = sum(
            len(p['care_shifts']) * p['nurses_needed'] 
            for p in self.patients
        )
        
        total_available_shifts = sum(
            len(e['available_shifts']) 
            for e in self.employees
        )
        
        return {
            "num_employees": len(self.employees),
            "num_patients": len(self.patients),
            "total_nurse_shifts_needed": total_patient_shifts,
            "total_available_employee_shifts": total_available_shifts,
            "availability_ratio": round(total_available_shifts / total_patient_shifts, 2) if total_patient_shifts > 0 else 0
        }
    
    def _count_assignments(self, schedule: Dict) -> int:
        """Count total number of assignments in a schedule"""
        count = 0
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                count += len(employee_ids)
        return count


def main():
    """Example usage"""
    
    # Setup logging to file
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/agent_run_{timestamp}.txt"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    
    try:
        # Initialize agent
        print("Initializing Nursing Scheduler Agent with Google Gemini...")
        print(f"Logging to: {log_file}")
        agent = NursingSchedulerAgent()
        
        # Print summary
        stats = agent.get_summary_stats()
        print("\nData Summary:")
        print(f"  Employees: {stats['num_employees']}")
        print(f"  Patients: {stats['num_patients']}")
        print(f"  Total nurse-shifts needed: {stats['total_nurse_shifts_needed']}")
        print(f"  Available employee-shifts: {stats['total_available_employee_shifts']}")
        print(f"  Availability ratio: {stats['availability_ratio']}x")
        print()
        
        # Example queries
        queries = [
            "Create an optimal schedule for the nursing facility for the next month. Prioritize continuity of care.",
        ]
        
        print("\nRunning example query...")
        print(f"Query: {queries[0]}")
        print("="*80)
        
        result = agent.run(queries[0])
        
        # Save the schedule if it was generated
        if agent.last_schedule:
            os.makedirs("schedules", exist_ok=True)
            schedule_file = f"schedules/schedule_{timestamp}.json"
            
            # Use BEST schedule, not last schedule
            schedule_to_save = agent.best_schedule if agent.best_schedule else agent.last_schedule
            
            print("\n" + "="*80)
            print("DEBUGGING: Validating final schedule for save...")
            print(f"Using {'BEST' if agent.best_schedule else 'LAST'} schedule")
            print(f"Schedule has {agent._count_assignments(schedule_to_save)} total assignments")
            
            is_valid, violations, breakdown = agent.validator.validate(schedule_to_save)
            print(f"Final validation: {len(violations)} total violations")
            print(f"  🚨 CRITICAL Patient violations: {breakdown['patient']}")
            print(f"  ⚠️  Employee preference violations: {breakdown['employee']}")
            print(f"Sample violations: {violations[:3]}")
            
            score_result = agent.scorer.score(schedule_to_save)
            
            # NEW: Generate staffing narrative
            staffing_narrative = generate_staffing_narrative(
                violations, agent.employees, agent.patients, schedule_to_save
            )
            
            with open(schedule_file, 'w') as f:
                json.dump({
                    'schedule': schedule_to_save,
                    'timestamp': timestamp,
                    'query': queries[0],
                    'validation': {
                        'is_valid': is_valid,
                        'violations': violations,
                        'total_violations': len(violations),
                        'patient_violations': breakdown['patient'],
                        'employee_violations': breakdown['employee'],
                        'breakdown': breakdown
                    },
                    'score': score_result,
                    'staffing_narrative': staffing_narrative  # NEW: Save the AI analysis
                }, f, indent=2)
            print(f"✓ Schedule saved to: {schedule_file}")
            print(f"  Patient violations (CRITICAL): {breakdown['patient']}")
            print(f"  Employee violations (SOFT): {breakdown['employee']}")
            if staffing_narrative:
                print(f"✓ AI staffing narrative included")
            else:
                print(f"⚠ No AI narrative generated (check GEMINI_API_KEY)")
        
        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        
        # Debug: print what keys are in result
        print(f"\nDEBUG - Result type: {type(result)}")
        print(f"DEBUG - Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # Check for intermediate steps
        if 'intermediate_steps' in result:
            print(f"\nDEBUG - Number of intermediate steps: {len(result['intermediate_steps'])}")
            if result['intermediate_steps']:
                print("\nDEBUG - Last few actions:")
                for i, (action, observation) in enumerate(result['intermediate_steps'][-3:]):
                    print(f"  Step {i}: {action.tool} - {observation[:100]}...")
        
        # Try to extract the output
        if isinstance(result, dict):
            output = result.get('output', '')
            
            if output:
                print("\n--- AGENT OUTPUT ---")
                print(output)
            else:
                print("\n--- NO OUTPUT GENERATED ---")
                print("The agent completed but didn't generate a final response.")
                print("This usually means:")
                print("  1. Hit max iterations")
                print("  2. Tool calls succeeded but no final synthesis")
                print("  3. Model stopped generating")
                
                # Show what was accomplished
                if 'intermediate_steps' in result and result['intermediate_steps']:
                    print(f"\nThe agent made {len(result['intermediate_steps'])} tool calls:")
                    for i, (action, observation) in enumerate(result['intermediate_steps']):
                        print(f"\n  {i+1}. Called: {action.tool}")
                        obs_preview = str(observation)[:200]
                        print(f"     Result: {obs_preview}...")
        else:
            print(f"\n--- UNEXPECTED RESULT TYPE ---")
            print(result)
        
        print(f"\n\n✓ Full log saved to: {log_file}")
        if agent.last_schedule:
            print(f"✓ Schedule saved to: schedules/schedule_{timestamp}.json")
            print(f"\nTo visualize:")
            print(f"  make visualize FILE=schedules/schedule_{timestamp}.json")
            print(f"  make visualize-html FILE=schedules/schedule_{timestamp}.json")
        
    finally:
        # Restore stdout and close log file
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    # Setup LangSmith tracing (optional)
    # Only enable if you have LANGCHAIN_API_KEY set
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "nursing-scheduler"
        print("✓ LangSmith tracing enabled")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    main()