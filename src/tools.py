"""
Nursing Shift Scheduler - Core Tool Implementations

Constraint-based optimization tools with patient safety prioritization:

1. ConstraintValidator - Separates CRITICAL (patient safety) from SOFT (preferences) violations
2. ScheduleScorer - Multi-objective optimization (cost, continuity, fairness, overtime) 
3. ScheduleGenerator - Greedy + iterative improvement with load balancing and exclusion handling
4. StaffingAnalyzer - Hiring recommendations based on violation patterns and utilization gaps
5. ScheduleComparator - Violation-prioritized schedule selection (implemented in agent.py)

Features: Consecutive shift enforcement, employee exclusions, expected vs max hours targeting,
skill/level coverage analysis, and comprehensive violation severity classification.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import copy

@dataclass
class ScheduleAssignment:
    """Single assignment: employee to patient for a shift"""
    shift_id: str
    patient_id: str
    employee_id: str


class ConstraintValidator:
    """
    Comprehensive constraint validation with violation severity prioritization.
    
    CRITICAL violations (patient safety/legal - MUST be 0):
    - Patient coverage gaps, skill mismatches, level requirements, safety exclusions
    
    SOFT violations (operational preferences - optimize but acceptable):
    - Employee availability, hour targets, consecutive shifts, load balancing
    
    Enables iterative improvement by separating life-safety issues from optimization targets.
    """
    
    def __init__(self, employees: List[Dict], patients: List[Dict]):
        self.employees = {e['employee_id']: e for e in employees}
        self.patients = {p['patient_id']: p for p in patients}
    
    def validate(self, schedule: Dict[str, Dict[str, List[str]]]) -> Tuple[bool, List[str], Dict]:
        """
        Validate schedule against all constraints.
        Returns (is_valid, list_of_violations, violation_breakdown)
        
        Violation breakdown separates:
        - patient_violations: CRITICAL - coverage, skills, levels (MUST be 0)
        - employee_violations: SOFT - availability, hours, conflicts (optimize but acceptable)
        """
        violations = []
        patient_violations = []  # CRITICAL - legal/safety issues
        employee_violations = []  # SOFT - preferences/optimization
        
        if not schedule:
            return False, ["Schedule is empty"], {'patient': 1, 'employee': 0, 'total': 1}
        
        # Track employee hours and shift assignments
        employee_hours = defaultdict(int)
        employee_shifts = defaultdict(list)
        
        # Track unique violations to avoid duplicates
        availability_violations_set = set()  # Track (emp_id, shift_id) tuples
        exclusion_violations_set = set()     # Track unique exclusion violations
        
        # Validate each shift assignment
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                
                # Check patient exists and needs this shift
                if patient_id not in self.patients:
                    v = f"Unknown patient {patient_id} in schedule"
                    violations.append(v)
                    patient_violations.append(v)
                    continue
                
                patient = self.patients[patient_id]
                
                if shift_id not in patient['care_shifts']:
                    v = f"Patient {patient_id} ({patient['name']}) does not need care during {shift_id}"
                    violations.append(v)
                    patient_violations.append(v)
                
                # Check minimum number of nurses (CRITICAL - patient safety)
                if len(employee_ids) < patient['nurses_needed']:
                    v = f"Patient {patient_id} ({patient['name']}) needs {patient['nurses_needed']} nurses minimum but only {len(employee_ids)} assigned for {shift_id}"
                    violations.append(v)
                    patient_violations.append(v)
                
                # Check maximum number of nurses (OPERATIONAL HARD CONSTRAINT)
                MAX_NURSES_PER_PATIENT = 3
                if len(employee_ids) > MAX_NURSES_PER_PATIENT:
                    v = f"SYSTEM ERROR: Patient {patient_id} ({patient['name']}) assigned {len(employee_ids)} nurses but maximum is {MAX_NURSES_PER_PATIENT} for {shift_id}"
                    violations.append(v)
                    patient_violations.append(v)
                
                # Validate each employee assignment
                levels_assigned = []
                skills_assigned = set()
                
                for emp_id in employee_ids:
                    if emp_id not in self.employees:
                        v = f"Unknown employee {emp_id} in schedule"
                        violations.append(v)
                        patient_violations.append(v)
                        continue
                    
                    employee = self.employees[emp_id]
                    
                    # Check availability (SOFT - employee preference)
                    # Only count once per unique (employee, shift) combination
                    if shift_id not in employee['available_shifts']:
                        violation_key = (emp_id, shift_id)
                        if violation_key not in availability_violations_set:
                            availability_violations_set.add(violation_key)
                            v = f"Employee {emp_id} ({employee['name']}) not available for {shift_id}"
                            violations.append(v)
                            employee_violations.append(v)
                    
                    # Check exclusions (CRITICAL - safety)
                    if patient_id in employee['excluded_patients']:
                        v = f"Employee {emp_id} ({employee['name']}) cannot work with patient {patient_id} ({patient['name']})"
                        violations.append(v)
                        patient_violations.append(v)
                    
                    if patient_id in self.patients and emp_id in self.patients[patient_id]['excluded_employees']:
                        v = f"Patient {patient_id} ({patient['name']}) cannot work with employee {emp_id} ({employee['name']})"
                        violations.append(v)
                        patient_violations.append(v)
                    
                    # Check employee-employee exclusions (SOFT - can be managed)
                    # Only count once per unique pair
                    for other_emp_id in employee_ids:
                        if other_emp_id != emp_id and other_emp_id in employee['excluded_employees']:
                            # Create a sorted tuple to avoid counting both (A,B) and (B,A)
                            pair_key = tuple(sorted([emp_id, other_emp_id])) + (shift_id,)
                            if pair_key not in exclusion_violations_set:
                                exclusion_violations_set.add(pair_key)
                                v = f"Employees {emp_id} and {other_emp_id} cannot work together in {shift_id}"
                                violations.append(v)
                                employee_violations.append(v)
                    
                    # Track for level and skill validation
                    levels_assigned.append(employee['level'])
                    skills_assigned.update(employee['skills'])
                    
                    # Track hours
                    employee_hours[emp_id] += 4  # 4-hour shift
                    employee_shifts[emp_id].append(shift_id)
                
                # Check minimum level requirement (CRITICAL - patient safety)
                if levels_assigned and max(levels_assigned) < patient['min_level']:
                    v = f"Patient {patient_id} ({patient['name']}) requires level {patient['min_level']} but highest assigned is level {max(levels_assigned)} for {shift_id}"
                    violations.append(v)
                    patient_violations.append(v)
                
                # Check required skills are covered by COMBINED team (CRITICAL - patient safety)
                required_skills = set(patient['required_skills'])
                if required_skills and not required_skills.issubset(skills_assigned):
                    missing = required_skills - skills_assigned
                    v = f"Patient {patient_id} ({patient['name']}) requires skills {list(missing)} but assigned nurses don't have them for {shift_id}"
                    violations.append(v)
                    patient_violations.append(v)
        
        # Validate employee hour constraints (SOFT - optimization)
        for emp_id, hours in employee_hours.items():
            employee = self.employees[emp_id]
            weekly_hours = hours / 4  # Convert monthly to weekly average
            
            # Priority constraint: Should not exceed max_hours_per_week
            if weekly_hours > employee['max_hours_per_week']:
                v = f"Employee {emp_id} ({employee['name']}) scheduled for {weekly_hours:.1f}h/week but max is {employee['max_hours_per_week']}h/week"
                violations.append(v)
                employee_violations.append(v)

            # SOFT constraint: Exceeding expected hours (overtime)
            expected_hours = employee.get('expected_hours_per_week', employee['max_hours_per_week'])
            if weekly_hours > expected_hours and weekly_hours <= employee['max_hours_per_week']:
                overtime_hours = weekly_hours - expected_hours
                v = f"Employee {emp_id} ({employee['name']}) working {overtime_hours:.1f}h overtime ({weekly_hours:.1f}h/week vs {expected_hours}h expected)"
                violations.append(v)
                employee_violations.append(v)
            
            if weekly_hours < employee['min_hours_per_week']:
                v = f"Employee {emp_id} ({employee['name']}) scheduled for {weekly_hours:.1f}h/week but min is {employee['min_hours_per_week']}h/week"
                violations.append(v)
                employee_violations.append(v)
        
        # Validate consecutive shift limits (SOFT - employee wellbeing)
        for emp_id, shifts in employee_shifts.items():
            consecutive_violations = self._check_consecutive_shifts(emp_id, shifts)
            violations.extend(consecutive_violations)
            employee_violations.extend(consecutive_violations)
        
        # Check for extreme load imbalance (employees at 2x+ max while others at 0)
        if employee_hours:
            max_hours_worked = max(employee_hours.values())
            min_hours_worked = min((h for h in employee_hours.values() if h > 0), default=0)
            
            # Flag employees working over their max
            for emp_id, hours in employee_hours.items():
                employee = self.employees[emp_id]
                weekly = hours / 4
                max_weekly = employee['max_hours_per_week']
                
                if weekly > max_weekly:
                    v = f"MAX HOURS EXCEEDED: Employee {emp_id} ({employee['name']}) scheduled for {weekly:.1f}h/week (over their {employee['max_hours_per_week']}h max) - CAPACITY VIOLATION"
                    violations.append(v)
                    employee_violations.append(v)
            
            # Flag severe imbalance (some working 2x while others idle)
            employees_working = len([h for h in employee_hours.values() if h > 0])
            employees_idle = len(self.employees) - employees_working
            
            if employees_idle > len(self.employees) * 0.3:  # More than 30% idle
                v = f"LOAD IMBALANCE: {employees_idle} employees ({employees_idle/len(self.employees)*100:.0f}%) have 0 hours while others are overworked"
                violations.append(v)
                employee_violations.append(v)
        
        # Check all required patient shifts are covered (CRITICAL - patient safety)
        for patient_id, patient in self.patients.items():
            for shift_id in patient['care_shifts']:
                if shift_id not in schedule or patient_id not in schedule[shift_id]:
                    v = f"Patient {patient_id} ({patient['name']}) needs care during {shift_id} but no nurses assigned"
                    violations.append(v)
                    patient_violations.append(v)
        
        breakdown = {
            'patient': len(patient_violations),
            'employee': len(employee_violations),
            'total': len(violations)
        }
        
        return len(violations) == 0, violations, breakdown
    
    def _check_consecutive_shifts(self, emp_id: str, shifts: List[str]) -> List[str]:
        """Check if employee has too many consecutive shifts"""
        violations = []
        
        # Parse shift IDs into (day, shift_num) tuples
        shift_times = []
        for s in shifts:
            try:
                # Format: D01S0 -> day=1, shift=0
                if len(s) < 5 or not s.startswith('D') or 'S' not in s:
                    print(f"⚠️  Skipping malformed shift ID: {s}")
                    continue
                day = int(s[1:3])
                shift_num = int(s[4:])
                shift_times.append((day, shift_num))
            except (ValueError, IndexError) as e:
                print(f"⚠️  Error parsing shift ID '{s}': {e}")
                continue
        
        shift_times.sort()
        
        # Check consecutive shifts
        consecutive_count = 1
        for i in range(1, len(shift_times)):
            prev_day, prev_shift = shift_times[i-1]
            curr_day, curr_shift = shift_times[i]
            
            # Check if shifts are consecutive (same day next shift, or next day first shift)
            is_consecutive = False
            if curr_day == prev_day and curr_shift == prev_shift + 1:
                is_consecutive = True
            elif curr_day == prev_day + 1 and prev_shift == 5 and curr_shift == 0:
                is_consecutive = True
            
            if is_consecutive:
                consecutive_count += 1
                if consecutive_count > 3:
                    employee = self.employees[emp_id]
                    violations.append(
                        f"Employee {emp_id} ({employee['name']}) has {consecutive_count} "
                        f"consecutive shifts (max is 3)"
                    )
            else:
                consecutive_count = 1
        
        return violations


class ScheduleScorer:
    """
    Multi-objective optimization scoring for valid schedules.
    
    Evaluates 4 key operational metrics (lower scores = better):
    - Cost: Total labor expense (hours × pay rates)
    - Continuity: Patient familiarity (fewer unique nurses per patient)  
    - Fairness: Workload distribution equality (coefficient of variation)
    - Overtime: Burnout prevention (penalty for >40h/week)
    
    Enables weighted optimization once patient safety constraints are satisfied.
    """
    
    def __init__(self, employees: List[Dict], patients: List[Dict]):
        self.employees = {e['employee_id']: e for e in employees}
        self.patients = {p['patient_id']: p for p in patients}
    
    def score(self, schedule: Dict[str, Dict[str, List[str]]], 
              weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Score schedule on multiple objectives.
        Returns dict with total_score and breakdown.
        """
        if weights is None:
            weights = {
                'cost': 1.0,
                'continuity': 2.0,
                'fairness': 1.0,
                'overtime': 1.5
            }
        
        # Calculate each component
        cost_score = self._calculate_cost(schedule)
        continuity_score = self._calculate_continuity_penalty(schedule)
        fairness_score = self._calculate_fairness_penalty(schedule)
        overtime_score = self._calculate_overtime_penalty(schedule)
        
        # Weighted total (normalize each component first)
        total_score = (
            weights['cost'] * (cost_score / 1000) +
            weights['continuity'] * continuity_score +
            weights['fairness'] * fairness_score +
            weights['overtime'] * overtime_score
        )
        
        return {
            'total_score': round(total_score, 2),
            'breakdown': {
                'total_cost': round(cost_score, 2),
                'continuity_penalty': round(continuity_score, 2),
                'fairness_penalty': round(fairness_score, 2),
                'overtime_penalty': round(overtime_score, 2)
            }
        }
    
    def _calculate_cost(self, schedule: Dict) -> float:
        """Calculate total labor cost"""
        total_cost = 0.0
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                for emp_id in employee_ids:
                    if emp_id in self.employees:
                        rate = self.employees[emp_id]['hourly_pay_rate']
                        total_cost += rate * 4  # 4-hour shift
        return total_cost
    
    def _calculate_continuity_penalty(self, schedule: Dict) -> float:
        """
        Penalty for high number of unique nurses per patient.
        Lower is better (fewer unique nurses = better continuity).
        """
        patient_nurses = defaultdict(set)
        
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                patient_nurses[patient_id].update(employee_ids)
        
        if not patient_nurses:
            return 0.0
        
        avg_unique_nurses = sum(len(nurses) for nurses in patient_nurses.values()) / len(patient_nurses)
        
        # Normalize: ideal is 2-3 unique nurses, penalty increases beyond that
        penalty = max(0, avg_unique_nurses - 3)
        return penalty
    
    def _calculate_fairness_penalty(self, schedule: Dict) -> float:
        """
        Penalty for uneven distribution of hours.
        Uses coefficient of variation (std dev / mean) of hours worked.
        """
        employee_hours = defaultdict(int)
        
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                for emp_id in employee_ids:
                    employee_hours[emp_id] += 4
        
        if not employee_hours:
            return 0.0
        
        hours_list = list(employee_hours.values())
        mean_hours = sum(hours_list) / len(hours_list)
        
        if mean_hours == 0:
            return 0.0
        
        variance = sum((h - mean_hours) ** 2 for h in hours_list) / len(hours_list)
        std_dev = variance ** 0.5
        
        # Coefficient of variation
        cv = std_dev / mean_hours
        return cv
    
    def _calculate_overtime_penalty(self, schedule: Dict) -> float:
        """
        Penalty for employees working over 40 hours per week.
        """
        employee_hours = defaultdict(int)
        
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                for emp_id in employee_ids:
                    employee_hours[emp_id] += 4
        
        total_overtime = 0.0
        for emp_id, total_hours in employee_hours.items():
            weekly_hours = total_hours / 4  # Approximate weekly from monthly
            employee = self.employees[emp_id]
            expected = employee.get('expected_hours_per_week', 40)
            
            # Count overtime as hours beyond expected (but within max)
            if weekly_hours > expected:
                overtime_hours = min(weekly_hours - expected, employee.get('max_hours_per_week', 40) - expected)
                total_overtime += overtime_hours
        
        return total_overtime


class ScheduleGenerator:
    """
    Intelligent schedule generation with constraint-aware heuristics.
    
    Strategies:
    - Greedy: Initial assignment targeting expected hours with load balancing
    - Iterative: Violation-guided refinement with aggressive patient coverage prioritization
    
    Features: Employee exclusion enforcement, consecutive shift limits (max 3),
    skill/level-aware team building, expected vs max hours optimization,
    and emergency relaxation for critical patient coverage.
    """
    
    MAX_NURSES_PER_PATIENT = 3  # Cost control constraint
    
    def __init__(self, employees: List[Dict], patients: List[Dict], config=None):
        self.employees = {e['employee_id']: e for e in employees}
        self.patients = {p['patient_id']: p for p in patients}
        self.config = config
    
    def generate(self, strategy: str = "greedy", 
                 seed_schedule: Optional[Dict] = None,
                 previous_violations: Optional[List[str]] = None) -> Dict:
        """Generate a schedule using specified strategy."""
        if strategy == "greedy":
            schedule = self._generate_greedy()
        elif strategy == "iterative" and seed_schedule:
            schedule = self._generate_iterative(seed_schedule, previous_violations or [])
        elif strategy == "random":
            schedule = self._generate_random()
        else:
            schedule = self._generate_greedy()  # Default
        
        metadata = {
            'strategy': strategy,
            'num_assignments': self._count_assignments(schedule),
            'seed_used': seed_schedule is not None,
            'improved_from_violations': len(previous_violations) if previous_violations else 0
        }
        
        return {
            'schedule': schedule,
            'metadata': metadata
        }
    
    def _get_week_number(self, shift_id: str) -> int:
        """Extract week number from shift_id. shift_id format: D01S0 -> day 1"""
        day = int(shift_id[1:3])
        return (day - 1) // 7  # Returns 0-3 for weeks 1-4
    
    def _generate_greedy(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Greedy heuristic: For each patient shift, assign best available employees.
        Ensures nurses aren't assigned to multiple patients in the same shift.
        """
        schedule = {}
        employee_hours = defaultdict(int)
        employee_hours_by_week = defaultdict(lambda: defaultdict(int))  # {emp_id: {week_num: hours}}
        employee_shift_history = defaultdict(list)
        patient_nurse_assignments = defaultdict(set)
        shift_employee_assignments = defaultdict(set)
        
        # Budget-based scheduling: Target expected_hours_per_week (not max) for load balancing
        employee_monthly_budget = {}
        for emp_id, emp in self.employees.items():
            expected_weekly = emp.get('expected_hours_per_week', emp.get('max_hours_per_week', 40))
            employee_monthly_budget[emp_id] = expected_weekly * 4  # 4 weeks

        employee_hours_remaining = {emp_id: budget for emp_id, budget in employee_monthly_budget.items()}
        
        # Get all patient shift requirements
        patient_shifts = []
        for patient_id, patient in self.patients.items():
            for shift_id in patient['care_shifts']:
                patient_shifts.append((shift_id, patient_id))
        
        patient_shifts.sort()
        
        # Group by shift
        shifts_by_time = defaultdict(list)
        for shift_id, patient_id in patient_shifts:
            shifts_by_time[shift_id].append(patient_id)
        
        # Process each shift
        for shift_id in sorted(shifts_by_time.keys()):
            patients_needing_care = shifts_by_time[shift_id]
            
            # Prioritize complex patients (harder to staff) to ensure coverage
            def patient_difficulty(patient_id):
                patient = self.patients[patient_id]
                return (
                    -patient.get('min_level', 1),           # Higher level requirements first
                    -len(patient.get('required_skills', [])),  # More skills needed first
                    -patient.get('nurses_needed', 1)       # More nurses needed first
                )
            
            patients_needing_care.sort(key=patient_difficulty)
            
            shift_assignments = {}
            
            for patient_id in patients_needing_care:
                patient = self.patients[patient_id]
                
                # Find eligible employees
                eligible = self._find_eligible_employees(
                    shift_id, patient_id, employee_hours, employee_shift_history,
                    employee_hours_by_week,
                    exclude_employees=shift_employee_assignments[shift_id]
                )
                
                # Try relaxed mode if no one available
                if len(eligible) == 0:
                    eligible = self._find_eligible_employees(
                        shift_id, patient_id, employee_hours, employee_shift_history,
                        employee_hours_by_week,
                        exclude_employees=shift_employee_assignments[shift_id],
                        relax_availability=True
                    )
                
                # Check minimum staffing
                if len(eligible) < patient['nurses_needed']:
                    continue
                
                # Sort eligible by priority (NEW: pass empty list initially for first nurse)
                eligible.sort(key=lambda emp_id: self._employee_priority(
                    emp_id, patient, employee_hours, patient_nurse_assignments,
                    employee_hours_by_week,
                    already_assigned_to_patient=[]
                ))

                # Build team with SMART level assignment: Only ONE nurse needs to meet level requirement
                assigned = []
                combined_skills = set()
                max_level_in_team = 0
                required_skills = set(patient.get('required_skills', []))
                min_nurses = patient['nurses_needed']
                min_level = patient.get('min_level', 1)
                max_nurses = min(self.MAX_NURSES_PER_PATIENT, len(eligible))

                # SMART TEAM BUILDING: 
                # 1. Find ONE nurse who meets the level requirement
                # 2. Fill remaining spots with lowest-level nurses available
                
                # Step 1: Find the BEST nurse who meets level requirement
                level_qualified = [emp_id for emp_id in eligible 
                                 if self.employees[emp_id]['level'] >= min_level]
                
                if not level_qualified:
                    # No one meets level requirement - skip this patient
                    continue
                
                # Sort level-qualified nurses by the same priority system (includes load balancing)
                level_qualified.sort(key=lambda emp_id: self._employee_priority(
                    emp_id, patient, employee_hours, patient_nurse_assignments,
                    employee_hours_by_week,
                    already_assigned_to_patient=[]
                ))
                primary_nurse = level_qualified[0]
                
                # Assign the primary level-qualified nurse
                emp = self.employees[primary_nurse]
                assigned.append(primary_nurse)
                combined_skills.update(emp['skills'])
                max_level_in_team = emp['level']
                
                # Step 2: Fill remaining spots with LOWEST level nurses (Level 1 preferred)
                remaining_slots = min_nurses - 1  # -1 because we already assigned primary
                
                if remaining_slots > 0:
                    # Get remaining eligible nurses (excluding the one we just assigned)
                    remaining_eligible = [emp_id for emp_id in eligible if emp_id != primary_nurse]
                    
                    # Sort remaining nurses by the same priority system
                    remaining_eligible.sort(key=lambda emp_id: self._employee_priority(
                        emp_id, patient, employee_hours, patient_nurse_assignments,
                        employee_hours_by_week,
                        already_assigned_to_patient=assigned
                    ))
                    
                    # Assign filler nurses (with exclusion checking)
                    for i in range(min(remaining_slots, len(remaining_eligible))):
                        filler_nurse = remaining_eligible[i]
                        
                        # Check if this nurse has exclusions with any already assigned nurses
                        has_conflict = False
                        filler_emp = self.employees[filler_nurse]
                        for already_assigned_emp_id in assigned:
                            # Check both directions: A excludes B or B excludes A
                            if (already_assigned_emp_id in filler_emp.get('excluded_employees', []) or
                                filler_nurse in self.employees[already_assigned_emp_id].get('excluded_employees', [])):
                                has_conflict = True
                                break
                        
                        if not has_conflict:
                            emp = self.employees[filler_nurse]
                            assigned.append(filler_nurse)
                            combined_skills.update(emp['skills'])
                        # If there's a conflict, skip this nurse and try the next one
                
                # Step 3: Check if we need ONE more nurse for skill coverage (only if < max_nurses)
                if len(assigned) < max_nurses:
                    missing_skills = required_skills - combined_skills
                    if missing_skills:
                        # Find nurses who can cover missing skills
                        skill_fillers = []
                        for emp_id in eligible:
                            if emp_id not in assigned:
                                emp = self.employees[emp_id]
                                emp_skills = set(emp['skills'])
                                if emp_skills & missing_skills:  # Can cover some missing skills
                                    skill_fillers.append(emp_id)
                        
                        if skill_fillers:
                            # Pick the LOWEST level nurse who covers the most missing skills
                            def skill_filler_priority(emp_id):
                                emp = self.employees[emp_id]
                                emp_skills = set(emp['skills'])
                                skills_covered = len(emp_skills & missing_skills)
                                hours_worked = employee_hours.get(emp_id, 0)
                                return (-skills_covered, hours_worked, emp['level'])
                            
                            skill_fillers.sort(key=skill_filler_priority)
                            
                            # Find first skill filler without exclusion conflicts
                            skill_nurse_added = False
                            for skill_candidate in skill_fillers:
                                # Check if this nurse has exclusions with any already assigned nurses
                                has_conflict = False
                                skill_emp = self.employees[skill_candidate]
                                for already_assigned_emp_id in assigned:
                                    # Check both directions: A excludes B or B excludes A
                                    if (already_assigned_emp_id in skill_emp.get('excluded_employees', []) or
                                        skill_candidate in self.employees[already_assigned_emp_id].get('excluded_employees', [])):
                                        has_conflict = True
                                        break
                                
                                if not has_conflict:
                                    assigned.append(skill_candidate)
                                    emp = self.employees[skill_candidate]
                                    combined_skills.update(emp['skills'])
                                    skill_nurse_added = True
                                    break  # Found a good skill filler, stop looking

                # Final verification
                has_min_nurses = len(assigned) >= min_nurses
                has_min_level = max_level_in_team >= patient.get('min_level', 1) if assigned else False
                has_all_skills = not required_skills or required_skills.issubset(combined_skills)

                if not (has_min_nurses and has_min_level and has_all_skills):
                    # Requirements not met, skip this patient-shift
                    continue

                # Record assignments
                shift_assignments[patient_id] = assigned

                for emp_id in assigned:
                    employee_hours[emp_id] += 4
                    week_num = self._get_week_number(shift_id)
                    employee_hours_by_week[emp_id][week_num] += 4  # NEW LINE
                    employee_shift_history[emp_id].append(shift_id)
                    patient_nurse_assignments[patient_id].add(emp_id)
                    shift_employee_assignments[shift_id].add(emp_id)
            
            if shift_assignments:
                schedule[shift_id] = shift_assignments
        
        # Budget utilization analysis for load balancing assessment
        total_monthly_budget = sum(emp['max_hours_per_week'] * 4 for emp in self.employees.values())
        total_hours_used = sum(employee_hours.values())
        budget_utilization = (total_hours_used / total_monthly_budget) * 100
        
        # Count employees by budget usage
        high_usage = 0  # > 80% of max
        medium_usage = 0  # 50-80% of max
        low_usage = 0  # < 50% of max
        
        for emp_id, hours in employee_hours.items():
            if emp_id in self.employees:
                max_hours = self.employees[emp_id]['max_hours_per_week'] * 4
                usage_pct = (hours / max_hours) * 100
                if usage_pct > 80:
                    high_usage += 1
                elif usage_pct > 50:
                    medium_usage += 1
                else:
                    low_usage += 1
        
        print(f"💰 Budget-Based Scheduling Results:")
        print(f"   Total budget utilization: {budget_utilization:.1f}%")
        print(f"   Employee distribution: {high_usage} high, {medium_usage} medium, {low_usage} low usage")
        
        return schedule
    
    def _employee_priority(self, emp_id: str, patient: Dict, 
                          employee_hours: Dict, patient_nurse_assignments: Dict,
                          employee_hours_by_week: Dict = None,
                          already_assigned_to_patient: List[str] = None) -> tuple:
        """
        Multi-factor priority calculation for optimal employee assignment.
        
        Balances: load distribution (target expected hours), skill efficiency,
        cost optimization, continuity of care, and overwork prevention.
        Returns tuple for sorting (lower = higher priority).
        """
        emp = self.employees[emp_id]
        patient_id = patient['patient_id']
        patient_min_level = patient.get('min_level', 1)
        patient_required_skills = set(patient.get('required_skills', []))
        
        # Weekly distribution smoothing (prevents cramming hours into few weeks)
        if employee_hours_by_week:
            weeks_hours = [employee_hours_by_week[emp_id][w] for w in range(4)]
            if weeks_hours and any(h > 0 for h in weeks_hours):
                avg_week = sum(weeks_hours) / 4
                max_deviation = max(abs(h - avg_week) for h in weeks_hours)
                weekly_imbalance_penalty = max_deviation * 2
            else:
                weekly_imbalance_penalty = 0
        else:
            weekly_imbalance_penalty = 0
        
        # Calculate metrics
        is_new_nurse = emp_id not in patient_nurse_assignments[patient_id]
        level_overqualified = max(0, emp['level'] - patient_min_level)
        emp_skills = set(emp['skills'])
        skill_overlap = len(patient_required_skills & emp_skills)
        hours_worked = employee_hours[emp_id]
        cost = emp['hourly_pay_rate']
        
        # Calculate utilization for load balancing
        # Calculate budget utilization
        monthly_budget = emp.get('expected_hours_per_week', emp.get('max_hours_per_week', 40)) * 4
        max_monthly = emp.get('max_hours_per_week', 40) * 4
        budget_used_ratio = hours_worked / monthly_budget if monthly_budget > 0 else 0
        approaching_max_ratio = hours_worked / max_monthly if max_monthly > 0 else 0

        # Load balancing strategy: Target expected hours, not max hours (prevents overwork)  
        max_weekly = emp.get('max_hours_per_week', 40)
        expected_weekly = emp.get('expected_hours_per_week', max_weekly)
        min_weekly = emp.get('min_hours_per_week', expected_weekly * 0.8)
        current_weekly = hours_worked / 4

        # Calculate distance from expected hours (our TARGET)
        hours_from_expected = current_weekly - expected_weekly
        
        # Priority: Aim for expected hours, respect min/max bounds
        if current_weekly >= max_weekly:
            # At max hours - extremely high penalty (desperate situations only)
            overwork_penalty = 1000000
        elif current_weekly > expected_weekly * 1.2:
            # Well over expected - very high penalty
            overwork_penalty = 50000 + (hours_from_expected * 5000)
        elif current_weekly > expected_weekly:
            # Over expected - penalty proportional to excess
            overwork_penalty = hours_from_expected * 3000  # Increased penalty
        elif current_weekly < min_weekly:
            # Below minimum - very strong bonus to get them to min hours
            underwork_bonus = (min_weekly - current_weekly) * 5000  # Much stronger bonus
            overwork_penalty = -underwork_bonus
        elif current_weekly < expected_weekly * 0.8:
            # Well below expected - strong bonus
            underwork_bonus = abs(hours_from_expected) * 2000  # Much stronger bonus
            overwork_penalty = -underwork_bonus
        elif current_weekly < expected_weekly * 0.9:
            # Below expected but above min - moderate bonus
            underwork_bonus = abs(hours_from_expected) * 1000  # Stronger bonus
            overwork_penalty = -underwork_bonus
        else:
            # Close to expected hours - minimal penalty
            overwork_penalty = abs(hours_from_expected) * 50
        
        # Team composition optimization: avoid skill duplication and waste
        skill_waste_penalty = 0
        skill_duplication_penalty = 0
        
        if already_assigned_to_patient:
            # Get skills already covered by the team
            team_skills = set()
            for other_emp_id in already_assigned_to_patient:
                if other_emp_id in self.employees:
                    team_skills.update(self.employees[other_emp_id]['skills'])
            
            # Calculate NEW skills this employee would add
            new_skills_added = emp_skills - team_skills
            required_new_skills = patient_required_skills - team_skills
            
            # Penalize if this employee adds NO new required skills (pure duplication)
            if required_new_skills and not (new_skills_added & patient_required_skills):
                skill_duplication_penalty = 50  # They're just duplicating existing coverage
            
            # Penalize employees with many skills when team is almost complete
            if len(team_skills & patient_required_skills) >= len(patient_required_skills) - 1:
                # Team almost has all required skills, prefer specialists
                unused_skills = len(emp_skills - patient_required_skills)
                skill_waste_penalty = unused_skills * 2  # Penalize each unused skill
            else:
                # Team needs more skills, prefer generalists
                unused_skills = len(emp_skills - patient_required_skills)
                skill_waste_penalty = unused_skills * 0.5  # Light penalty for unused skills
        else:
            # First nurse on this patient - prefer someone with multiple required skills
            unused_skills = len(emp_skills - patient_required_skills)
            skill_waste_penalty = unused_skills * 1  # Moderate penalty
        
        return (
            is_new_nurse,                    # Priority 1: Prefer new nurses for continuity
            overwork_penalty,                # Priority 2: CRITICAL - Load balancing (favor underutilized)
            -skill_overlap,                  # Priority 3: Prefer nurses with required skills
            level_overqualified,             # Priority 4: Avoid overqualified (expensive)
            skill_duplication_penalty,       # Priority 5: Avoid pure duplication
            skill_waste_penalty,             # Priority 6: Minimize unused skills  
            approaching_max_ratio,           # Priority 7: Prefer employees with more hours remaining
            hours_worked,                    # Priority 8: Secondary load balance
            cost                             # Priority 9: Lower cost
        )
    
    def _find_eligible_employees(self, shift_id: str, patient_id: str,
                                 employee_hours: Dict, 
                                 employee_shift_history: Dict,
                                 employee_hours_by_week: Dict,
                                 exclude_employees: set = None,
                                 relax_availability: bool = False) -> List[str]:
        """Find employees eligible for this shift."""
        if exclude_employees is None:
            exclude_employees = set()
        
        eligible = []
        patient = self.patients[patient_id]
        
        for emp_id, employee in self.employees.items():
            # Skip if already assigned this shift
            if emp_id in exclude_employees:
                continue
            
            # Check availability (can be relaxed)
            if not relax_availability:
                if shift_id not in employee['available_shifts']:
                    continue
            
            # Check exclusions (NEVER relax)
            if patient_id in employee['excluded_patients']:
                continue
            if emp_id in patient['excluded_employees']:
                continue
            
            # SAFETY: Never schedule beyond max_hours (strict hard limit)
            total_hours = employee_hours[emp_id]
            weekly_hours = total_hours / 4
            absolute_max = employee['max_hours_per_week']  # max_hours is absolute ceiling
            if weekly_hours + 1 > absolute_max:
                continue  # Skip this employee - they're at max capacity
            
            # Check consecutive shifts - look at more history to properly prevent 4+ consecutive
            if len(employee_shift_history[emp_id]) >= 1:
                # Check if adding this shift would exceed 3 consecutive shifts
                if self._would_exceed_consecutive(employee_shift_history[emp_id], shift_id):
                    continue
            
            # Weekly balance is handled by priority function, not hard filter
            # Don't exclude employees here - let them be considered
            
            eligible.append(emp_id)
        
        # Emergency: Include ALL employees except those with hard exclusions
        # Patient coverage is priority #1, max_hours is priority #2 (can violate)
        if not eligible:
            emergency_eligible = []
            for emp_id, employee in self.employees.items():
                # Hard exclusions
                if emp_id in exclude_employees:
                    continue
                if emp_id in patient['excluded_employees']:
                    continue
                if patient_id in employee.get('excluded_patients', []):
                    continue
                
                # SAFETY: Even in emergency, don't schedule beyond max_hours
                weekly_hours = employee_hours[emp_id] / 4
                absolute_max = employee['max_hours_per_week']
                if weekly_hours + 1 > absolute_max:
                    continue
                
                # Check consecutive shifts - use comprehensive check
                if self._would_exceed_consecutive(employee_shift_history[emp_id], shift_id):
                    continue
                
                emergency_eligible.append(emp_id)
            
            # Sort emergency candidates by hours worked (prefer underutilized)
            emergency_eligible.sort(key=lambda e: employee_hours.get(e, 0))
            
            if emergency_eligible:
                print(f"🚨 Emergency override activated for shift {shift_id}: using {len(emergency_eligible)} near-max employees")
                eligible = emergency_eligible
        
        # Prioritize underutilized employees for load balancing
        eligible_with_hours = [(emp_id, employee_hours.get(emp_id, 0)) for emp_id in eligible]
        eligible_with_hours.sort(key=lambda x: x[1])  # Sort by hours worked (ascending)
        eligible = [emp_id for emp_id, _ in eligible_with_hours]
        
        return eligible
    
    def _would_exceed_consecutive(self, shift_history: List[str], new_shift: str) -> bool:
        """Comprehensive check to prevent more than 3 consecutive shifts."""
        if not shift_history:
            return False
        
        def parse_shift(s):
            day = int(s[1:3])
            shift_num = int(s[4:])
            return (day, shift_num)
        
        def are_consecutive(shift1_str: str, shift2_str: str) -> bool:
            """Check if two shifts are consecutive."""
            s1 = parse_shift(shift1_str)
            s2 = parse_shift(shift2_str)
            
            # Same day, sequential shifts
            if s1[0] == s2[0] and s2[1] == s1[1] + 1:
                return True
            # Cross-day: last shift of day -> first shift of next day
            elif s2[0] == s1[0] + 1 and s1[1] == 5 and s2[1] == 0:
                return True
            return False
        
        # Create a list including the new shift, sorted by chronological order
        all_shifts = shift_history + [new_shift]
        
        # Sort shifts chronologically
        def shift_sort_key(shift_str):
            day, shift_num = parse_shift(shift_str)
            return (day, shift_num)
        
        all_shifts.sort(key=shift_sort_key)
        
        # Count consecutive sequences
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(all_shifts)):
            if are_consecutive(all_shifts[i-1], all_shifts[i]):
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        # Return True if adding this shift would exceed 3 consecutive
        return max_consecutive > 3
    
    def _generate_random(self) -> Dict:
        """Random assignment (fallback to greedy)."""
        return self._generate_greedy()
    
    def _generate_iterative(self, seed_schedule: Dict, violations: List[str]) -> Dict:
        """
        Iterative improvement: Start with seed schedule and AGGRESSIVELY fix patient violations.
        Uses multiple strategies to eliminate coverage gaps.
        """
        schedule = copy.deepcopy(seed_schedule)
        
        # If no violations passed, re-validate to find them
        if not violations:
            print("  ⚠ Re-validating seed to find violations...")
            validator = ConstraintValidator(
                list(self.employees.values()),
                list(self.patients.values())
            )
            _, violations, _ = validator.validate(seed_schedule)
            print(f"  Found {len(violations)} violations to fix")
        
        # Track current state
        employee_hours = defaultdict(int)
        shift_employee_assignments = defaultdict(set)
        
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                for emp_id in employee_ids:
                    employee_hours[emp_id] += 4
                    shift_employee_assignments[shift_id].add(emp_id)
        
        # Parse PATIENT violations (CRITICAL - must fix these!)
        uncovered_shifts = []  # (shift_id, patient_id)
        insufficient_shifts = []  # (shift_id, patient_id, needed_count)
        skill_gap_shifts = []  # (shift_id, patient_id)
        
        for violation in violations:
            if "needs care during" in violation and "no nurses assigned" in violation:
                patient_id, shift_id = self._parse_uncovered_violation(violation)
                if patient_id and shift_id:
                    uncovered_shifts.append((shift_id, patient_id))
            
            elif "needs" in violation and "nurses minimum but only" in violation and "assigned for" in violation:
                patient_id, shift_id, needed = self._parse_insufficient_violation(violation, schedule)
                if patient_id and shift_id and needed > 0:
                    insufficient_shifts.append((shift_id, patient_id, needed))
            
            elif "requires skills" in violation and "but assigned nurses don't have them" in violation:
                patient_id, shift_id = self._parse_skill_gap_violation(violation)
                if patient_id and shift_id:
                    skill_gap_shifts.append((shift_id, patient_id))
            
            elif "requires level" in violation and "but highest assigned is level" in violation:
                patient_id, shift_id = self._parse_level_violation(violation)
                if patient_id and shift_id:
                    # Treat level violations like skill gaps - need to rebuild team
                    skill_gap_shifts.append((shift_id, patient_id))
        
        print(f"  Iterative fixing: {len(uncovered_shifts)} uncovered, {len(insufficient_shifts)} insufficient, {len(skill_gap_shifts)} skill/level gaps")
        
        # FIX 1: Uncovered shifts (HIGHEST PRIORITY)
        for shift_id, patient_id in uncovered_shifts:
            self._assign_team_to_shift_aggressive(
                schedule, shift_id, patient_id,
                employee_hours, shift_employee_assignments
            )
        
        # FIX 2: Insufficient staffing
        for shift_id, patient_id, needed_count in insufficient_shifts:
            self._add_nurses_to_shift_aggressive(
                schedule, shift_id, patient_id, needed_count,
                employee_hours, shift_employee_assignments
            )
        
        # FIX 3: Skill/Level gaps - rebuild the team
        for shift_id, patient_id in skill_gap_shifts:
            if shift_id in schedule and patient_id in schedule[shift_id]:
                # Remove current inadequate team
                current_team = schedule[shift_id][patient_id]
                for emp_id in current_team:
                    employee_hours[emp_id] -= 4
                    shift_employee_assignments[shift_id].discard(emp_id)
                
                # Rebuild with proper team
                self._assign_team_to_shift_aggressive(
                    schedule, shift_id, patient_id,
                    employee_hours, shift_employee_assignments
                )
        
        return schedule

    def _assign_team_to_shift_aggressive(self, schedule: Dict, shift_id: str, patient_id: str,
                                         employee_hours: Dict, shift_employee_assignments: Dict):
        """
        AGGRESSIVELY assign a team to a shift, relaxing constraints as needed.
        PRIORITY: Patient coverage > Employee preferences
        """
        patient = self.patients[patient_id]
        required_skills = set(patient.get('required_skills', []))
        min_level = patient.get('min_level', 1)
        min_nurses = patient['nurses_needed']
        
        # PASS 1: Try with available nurses only
        eligible = self._find_candidates_for_shift(
            shift_id, patient_id, employee_hours, shift_employee_assignments,
            relax_availability=False, schedule=schedule
        )
        
        team = self._build_optimal_team(eligible, patient, min_nurses, employee_hours)
        
        # PASS 2: Relax availability if needed
        if not self._team_is_valid(team, patient):
            eligible = self._find_candidates_for_shift(
                shift_id, patient_id, employee_hours, shift_employee_assignments,
                relax_availability=True, schedule=schedule
            )
            team = self._build_optimal_team(eligible, patient, min_nurses, employee_hours)
        
        # PASS 3: Still relaxing availability only (hours remain hard constraint)
        if not self._team_is_valid(team, patient):
            eligible = self._find_candidates_for_shift(
                shift_id, patient_id, employee_hours, shift_employee_assignments,
                relax_availability=True, schedule=schedule
            )
            team = self._build_optimal_team(eligible, patient, min_nurses, employee_hours)
        
        # PASS 4: DESPERATE - ignore almost everything except hard exclusions
        if not self._team_is_valid(team, patient):
            # Need to rebuild employee_shift_history for consecutive checking
            employee_shift_history = defaultdict(list)
            for s_id, patients in schedule.items():
                for p_id, emp_ids in patients.items():
                    for emp_id in emp_ids:
                        employee_shift_history[emp_id].append(s_id)
            
            # Sort each employee's shift history
            for emp_id in employee_shift_history:
                employee_shift_history[emp_id].sort(key=lambda s: (int(s[1:3]), int(s[4:])))
            
            eligible = []
            for emp_id, emp in self.employees.items():
                # Only check HARD constraints
                if emp_id in shift_employee_assignments[shift_id]:
                    continue
                if patient_id in emp['excluded_patients']:
                    continue
                if emp_id in patient['excluded_employees']:
                    continue
                
                # SAFETY: Even in desperate mode, don't exceed max_hours
                total_hours = employee_hours[emp_id]
                weekly_hours = total_hours / 4
                absolute_max = emp['max_hours_per_week']
                if weekly_hours + 1 > absolute_max:
                    continue
                
                # SAFETY: Even in desperate mode, don't exceed 3 consecutive shifts
                if self._would_exceed_consecutive(employee_shift_history[emp_id], shift_id):
                    continue
                
                eligible.append(emp_id)
            
            team = self._build_optimal_team(eligible, patient, min_nurses, employee_hours)
        
        # Assign if valid
        if self._team_is_valid(team, patient):
            if shift_id not in schedule:
                schedule[shift_id] = {}
            schedule[shift_id][patient_id] = team
            
            for emp_id in team:
                employee_hours[emp_id] += 4
                shift_employee_assignments[shift_id].add(emp_id)

    def _find_candidates_for_shift(self, shift_id: str, patient_id: str,
                                   employee_hours: Dict, shift_employee_assignments: Dict,
                                   relax_availability: bool = False,
                                   schedule: Dict = None) -> List[str]:
        """Find candidate employees for a shift with optional constraint relaxation."""
        eligible = []
        patient = self.patients[patient_id]
        
        # Build employee shift history for consecutive checking
        employee_shift_history = defaultdict(list)
        if schedule:
            for s_id, patients in schedule.items():
                for p_id, emp_ids in patients.items():
                    for emp_id in emp_ids:
                        employee_shift_history[emp_id].append(s_id)
            
            # Sort each employee's shift history
            for emp_id in employee_shift_history:
                employee_shift_history[emp_id].sort(key=lambda s: (int(s[1:3]), int(s[4:])))
        
        for emp_id, emp in self.employees.items():
            # NEVER relax: Already assigned this shift
            if emp_id in shift_employee_assignments[shift_id]:
                continue
            
            # NEVER relax: Hard exclusions
            if patient_id in emp['excluded_patients']:
                continue
            if emp_id in patient['excluded_employees']:
                continue
            
            # Can relax: Availability
            if not relax_availability:
                if shift_id not in emp['available_shifts']:
                    continue
            
            # SAFETY: Never schedule beyond max_hours (strict hard limit)
            total_hours = employee_hours[emp_id]
            weekly_hours = total_hours / 4
            absolute_max = emp['max_hours_per_week']
            if weekly_hours + 1 > absolute_max:
                continue  # Skip this employee - they're at max capacity
            
            # SAFETY: Never exceed 3 consecutive shifts
            if self._would_exceed_consecutive(employee_shift_history[emp_id], shift_id):
                continue
            
            eligible.append(emp_id)
        
        # Sort by hours worked (prefer underutilized)
        eligible.sort(key=lambda e: employee_hours.get(e, 0))
        
        return eligible

    def _build_optimal_team(self, eligible: List[str], patient: Dict, min_nurses: int, employee_hours: Dict = None) -> List[str]:
        """Build optimal team: ONE nurse meets level, others are lowest level possible."""
        if len(eligible) < min_nurses:
            return []
        
        team = []
        combined_skills = set()
        required_skills = set(patient.get('required_skills', []))
        min_level = patient.get('min_level', 1)
        
        if employee_hours is None:
            employee_hours = {}
        
        # SMART TEAM BUILDING: Only ONE nurse needs to meet level requirement
        
        # Step 1: Find the BEST nurse who meets level requirement
        level_qualified = [emp_id for emp_id in eligible 
                         if self.employees[emp_id]['level'] >= min_level]
        
        if not level_qualified:
            return []  # No one meets level requirement
        
        # Sort level-qualified nurses by: skills coverage, hours worked, prefer lowest qualifying level
        def level_nurse_priority(emp_id):
            emp = self.employees[emp_id]
            emp_skills = set(emp['skills'])
            skill_coverage = len(emp_skills & required_skills)
            hours_worked = employee_hours.get(emp_id, 0)
            return (-skill_coverage, hours_worked, emp['level'])
        
        level_qualified.sort(key=level_nurse_priority)
        primary_nurse = level_qualified[0]
        
        # Assign primary nurse
        emp = self.employees[primary_nurse]
        team.append(primary_nurse)
        combined_skills.update(emp['skills'])
        
        # Step 2: Fill remaining spots with LOWEST level nurses
        remaining_slots = min_nurses - 1
        
        if remaining_slots > 0:
            remaining_eligible = [emp_id for emp_id in eligible if emp_id != primary_nurse]
            
            # Sort by: missing skills coverage, hours worked, then LOWEST level
            def filler_priority(emp_id):
                emp = self.employees[emp_id]
                emp_skills = set(emp['skills'])
                missing_skills = required_skills - combined_skills
                new_skills = len(emp_skills & missing_skills)
                hours_worked = employee_hours.get(emp_id, 0)
                return (-new_skills, hours_worked, emp['level'])
            
            remaining_eligible.sort(key=filler_priority)
            
            # Add filler nurses with exclusion checking
            for candidate in remaining_eligible:
                if len(team) >= min_nurses:
                    break  # We have enough nurses
                    
                # Check if this nurse has exclusions with any already assigned nurses
                has_conflict = False
                candidate_emp = self.employees[candidate]
                for already_assigned_emp_id in team:
                    # Check both directions: A excludes B or B excludes A
                    if (already_assigned_emp_id in candidate_emp.get('excluded_employees', []) or
                        candidate in self.employees[already_assigned_emp_id].get('excluded_employees', [])):
                        has_conflict = True
                        break
                
                if not has_conflict:
                    team.append(candidate)
                    emp = self.employees[candidate]
                    combined_skills.update(emp['skills'])
        
        # Step 3: Add one more nurse if needed for skills (up to max 3)
        if len(team) < self.MAX_NURSES_PER_PATIENT:
            missing_skills = required_skills - combined_skills
            if missing_skills:
                candidates = [emp_id for emp_id in eligible if emp_id not in team]
                skill_candidates = []
                
                for emp_id in candidates:
                    emp = self.employees[emp_id]
                    emp_skills = set(emp['skills'])
                    if emp_skills & missing_skills:
                        skill_candidates.append(emp_id)
                
                if skill_candidates:
                    def skill_priority(emp_id):
                        emp = self.employees[emp_id]
                        emp_skills = set(emp['skills'])
                        skills_covered = len(emp_skills & missing_skills)
                        hours_worked = employee_hours.get(emp_id, 0)
                        return (-skills_covered, hours_worked, emp['level'])
                    
                    skill_candidates.sort(key=skill_priority)
                    
                    # Find first skill candidate without exclusion conflicts
                    for skill_candidate in skill_candidates:
                        # Check if this nurse has exclusions with any already assigned nurses
                        has_conflict = False
                        skill_emp = self.employees[skill_candidate]
                        for already_assigned_emp_id in team:
                            # Check both directions: A excludes B or B excludes A
                            if (already_assigned_emp_id in skill_emp.get('excluded_employees', []) or
                                skill_candidate in self.employees[already_assigned_emp_id].get('excluded_employees', [])):
                                has_conflict = True
                                break
                        
                        if not has_conflict:
                            team.append(skill_candidate)
                            break  # Found a good skill candidate, stop looking
        
        return team

    def _team_is_valid(self, team: List[str], patient: Dict) -> bool:
        """Check if a team meets patient requirements."""
        if len(team) < patient['nurses_needed']:
            return False
        
        if not team:
            return False
        
        # Check level
        max_level = max(self.employees[emp_id]['level'] for emp_id in team)
        if max_level < patient.get('min_level', 1):
            return False
        
        # Check skills
        required_skills = set(patient.get('required_skills', []))
        if required_skills:
            team_skills = set()
            for emp_id in team:
                team_skills.update(self.employees[emp_id]['skills'])
            if not required_skills.issubset(team_skills):
                return False
        
        return True

    def _parse_level_violation(self, violation: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract patient_id and shift_id from level requirement violation."""
        patient_id = None
        shift_id = None
        
        for pid in self.patients.keys():
            if pid in violation:
                patient_id = pid
                break
        
        parts = violation.split()
        for part in parts:
            if part.startswith('D') and 'S' in part and len(part) <= 6:
                shift_id = part
                break
        
        return patient_id, shift_id

    def _parse_skill_gap_violation(self, violation: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract patient_id and shift_id from skill gap violation."""
        patient_id = None
        shift_id = None
        
        for pid in self.patients.keys():
            if pid in violation:
                patient_id = pid
                break
        
        parts = violation.split()
        for part in parts:
            if part.startswith('D') and 'S' in part and len(part) <= 6:
                shift_id = part
                break
        
        return patient_id, shift_id

    def _add_nurses_to_shift_aggressive(self, schedule: Dict, shift_id: str, patient_id: str,
                                        needed_count: int, employee_hours: Dict,
                                        shift_employee_assignments: Dict):
        """Aggressively add nurses to insufficient shifts."""
        if shift_id not in schedule or patient_id not in schedule[shift_id]:
            return
        
        current_team = schedule[shift_id][patient_id]
        patient = self.patients[patient_id]
        
        # Find additional nurses with relaxed constraints
        eligible = self._find_candidates_for_shift(
            shift_id, patient_id, employee_hours, shift_employee_assignments,
            relax_availability=True, schedule=schedule
        )
        
        # Remove already assigned
        eligible = [e for e in eligible if e not in current_team]
        
        # Check max constraint
        max_additional = min(needed_count, self.MAX_NURSES_PER_PATIENT - len(current_team))
        
        if eligible and max_additional > 0:
            # Sort by usefulness
            required_skills = set(patient.get('required_skills', []))
            current_skills = set()
            for emp_id in current_team:
                current_skills.update(self.employees[emp_id]['skills'])
            
            def usefulness(emp_id):
                emp = self.employees[emp_id]
                new_skills = len((set(emp['skills']) & required_skills) - current_skills)
                return (-new_skills, employee_hours.get(emp_id, 0))
            
            eligible.sort(key=usefulness)
            
            # Add nurses with exclusion checking
            additional = []
            for candidate in eligible:
                if len(additional) >= max_additional:
                    break  # We have enough additional nurses
                    
                # Check if this nurse has exclusions with any current team members
                has_conflict = False
                candidate_emp = self.employees[candidate]
                for current_emp_id in current_team:
                    # Check both directions: A excludes B or B excludes A
                    if (current_emp_id in candidate_emp.get('excluded_employees', []) or
                        candidate in self.employees[current_emp_id].get('excluded_employees', [])):
                        has_conflict = True
                        break
                
                if not has_conflict:
                    additional.append(candidate)
            
            schedule[shift_id][patient_id].extend(additional)
            
            for emp_id in additional:
                employee_hours[emp_id] += 4
                shift_employee_assignments[shift_id].add(emp_id)

    def _count_assignments(self, schedule: Dict) -> int:
        """Count total number of assignments in a schedule"""
        count = 0
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                count += len(employee_ids)
        return count

    def _parse_uncovered_violation(self, violation: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract patient_id and shift_id from uncovered violation."""
        patient_id = None
        shift_id = None
        
        for pid in self.patients.keys():
            if pid in violation:
                patient_id = pid
                break
        
        parts = violation.split()
        for part in parts:
            if part.startswith('D') and 'S' in part and len(part) <= 6:
                shift_id = part
                break
        
        return patient_id, shift_id

    def _parse_insufficient_violation(self, violation: str, schedule: Dict) -> Tuple[Optional[str], Optional[str], int]:
        """Extract patient_id, shift_id, and needed count from insufficient violation."""
        patient_id = None
        shift_id = None
        
        for pid in self.patients.keys():
            if pid in violation:
                patient_id = pid
                break
        
        parts = violation.split()
        for part in parts:
            if part.startswith('D') and 'S' in part:
                shift_id = part
                break
        
        needed = 0
        if patient_id and shift_id:
            patient = self.patients[patient_id]
            current_count = len(schedule.get(shift_id, {}).get(patient_id, []))
            needed = patient['nurses_needed'] - current_count
        
        return patient_id, shift_id, needed

    def _parse_skill_gap_violation(self, violation: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract patient_id and shift_id from skill gap violation."""
        patient_id = None
        shift_id = None
        
        for pid in self.patients.keys():
            if pid in violation:
                patient_id = pid
                break
        
        parts = violation.split()
        for part in parts:
            if part.startswith('D') and 'S' in part and len(part) <= 6:
                shift_id = part
                break
        
        return patient_id, shift_id

    def _parse_level_violation(self, violation: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract patient_id and shift_id from level violation."""
        patient_id = None
        shift_id = None
        
        for pid in self.patients.keys():
            if pid in violation:
                patient_id = pid
                break
        
        parts = violation.split()
        for part in parts:
            if part.startswith('D') and 'S' in part and len(part) <= 6:
                shift_id = part
                break
        
        return patient_id, shift_id


class StaffingAnalyzer:
    """
    Strategic workforce planning based on violation patterns and utilization analysis.
    
    Identifies hiring needs from critical coverage gaps and skill mismatches.
    Detects termination candidates from chronic underutilization.
    Provides prioritized recommendations with specific shift timing and skill requirements.
    
    Supports data-driven staffing decisions aligned with operational needs.
    """
    
    def __init__(self, employees: List[Dict], patients: List[Dict]):
        self.employees = {e['employee_id']: e for e in employees}
        self.patients = {p['patient_id']: p for p in patients}
    
    def analyze(self, schedule: Dict[str, Dict[str, List[str]]], 
                violations: List[str]) -> Dict:
        """Analyze staffing based on schedule and violations."""
        # Track utilization
        employee_shifts = defaultdict(int)
        
        for shift_id, patient_assignments in schedule.items():
            for patient_id, employee_ids in patient_assignments.items():
                for emp_id in employee_ids:
                    employee_shifts[emp_id] += 1
        
        # Analyze violations
        skill_gaps = defaultdict(int)
        level_gaps = defaultdict(int)
        uncovered_shifts = defaultdict(list)
        
        for violation in violations:
            if "needs care during" in violation and "no nurses assigned" in violation:
                parts = violation.split()
                patient_id = parts[1]
                
                if patient_id in self.patients:
                    patient = self.patients[patient_id]
                    for skill in patient['required_skills']:
                        skill_gaps[skill] += 1
                    level_gaps[patient['min_level']] += 1
        
        # Analyze utilization
        overworked = []
        underutilized = []
        
        for emp_id, emp in self.employees.items():
            shifts_worked = employee_shifts.get(emp_id, 0)
            total_hours = shifts_worked * 4
            avg_weekly = total_hours / 4
            max_hours = emp['max_hours_per_week']
            min_hours = emp['min_hours_per_week']
            
            utilization = (avg_weekly / max_hours * 100) if max_hours > 0 else 0
            
            if avg_weekly > max_hours * 0.95:
                overworked.append({
                    'employee_id': emp_id,
                    'name': emp['name'],
                    'level': emp['level'],
                    'actual_hours': round(avg_weekly, 1),
                    'max_hours': max_hours,
                    'utilization': round(utilization, 1),
                    'reason': 'At or exceeding maximum capacity - burnout risk'
                })
            elif avg_weekly < min_hours * 0.5:
                underutilized.append({
                    'employee_id': emp_id,
                    'name': emp['name'],
                    'level': emp['level'],
                    'actual_hours': round(avg_weekly, 1),
                    'min_hours': min_hours,
                    'utilization': round(utilization, 1),
                    'reason': 'Significantly underutilized - not meeting minimum hours'
                })
        
        # Generate recommendations
        hiring_recommendations = []
        
        if skill_gaps:
            top_skill_gaps = sorted(skill_gaps.items(), key=lambda x: -x[1])[:3]
            for skill, gap_count in top_skill_gaps:
                avg_level = 2
                for patient in self.patients.values():
                    if skill in patient['required_skills']:
                        avg_level = max(avg_level, patient['min_level'])
                
                hiring_recommendations.append({
                    'skill': skill,
                    'level': avg_level,
                    'urgency': 'HIGH' if gap_count > 50 else 'MEDIUM' if gap_count > 20 else 'LOW',
                    'gaps': gap_count,
                    'reason': f'{gap_count} shifts lack this critical skill'
                })
        
        if level_gaps:
            for level, gap_count in sorted(level_gaps.items(), key=lambda x: -x[1]):
                if gap_count > 30:
                    hiring_recommendations.append({
                        'skill': 'General nursing',
                        'level': level,
                        'urgency': 'HIGH' if gap_count > 50 else 'MEDIUM',
                        'gaps': gap_count,
                        'reason': f'Need more Level {level} nurses to cover {gap_count} shifts'
                    })
        
        # Termination candidates
        termination_candidates = []
        for emp in underutilized:
            if emp['utilization'] < 20:
                termination_candidates.append({
                    'employee_id': emp['employee_id'],
                    'name': emp['name'],
                    'level': emp['level'],
                    'utilization': emp['utilization'],
                    'actual_hours': emp['actual_hours'],
                    'reason': f'Only working {emp["actual_hours"]}h/week ({emp["utilization"]:.0f}% capacity)',
                    'alternative': 'Could reassign to PRN/on-call status'
                })
        
        return {
            'hiring_recommendations': hiring_recommendations,
            'termination_candidates': termination_candidates,
            'overworked_employees': overworked,
            'skill_gaps': dict(sorted(skill_gaps.items(), key=lambda x: -x[1])),
            'level_gaps': dict(sorted(level_gaps.items(), key=lambda x: -x[1])),
            'total_uncovered_shifts': sum(len(shifts) for shifts in uncovered_shifts.values()),
            'affected_patients': len(uncovered_shifts)
        }


# Tool wrapper functions

def validate_schedule(schedule: Dict, employees: List[Dict], patients: List[Dict]) -> Dict:
    """Tool wrapper: Validate schedule against all constraints."""
    validator = ConstraintValidator(employees, patients)
    is_valid, violations, breakdown = validator.validate(schedule)
    return {
        'valid': is_valid,
        'violations': violations,
        'breakdown': breakdown
    }


def score_schedule(schedule: Dict, employees: List[Dict], patients: List[Dict],
                  weights: Optional[Dict] = None) -> Dict:
    """Tool wrapper: Score schedule on soft constraints."""
    scorer = ScheduleScorer(employees, patients)
    return scorer.score(schedule, weights)


def generate_schedule(employees: List[Dict], patients: List[Dict],
                     strategy: str = "greedy",
                     seed_schedule: Optional[Dict] = None,
                     previous_violations: Optional[List[str]] = None,
                     config=None) -> Dict:
    """Tool wrapper: Generate a candidate schedule."""
    generator = ScheduleGenerator(employees, patients, config)
    return generator.generate(strategy, seed_schedule, previous_violations)


def analyze_staffing(schedule: Dict, violations: List[str],
                    employees: List[Dict], patients: List[Dict]) -> Dict:
    """Tool wrapper: Analyze staffing needs and suggest improvements."""
    analyzer = StaffingAnalyzer(employees, patients)
    return analyzer.analyze(schedule, violations)