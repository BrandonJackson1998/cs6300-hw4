"""
Nursing Shift Scheduler - Tool Implementations

Four tools for constraint-based shift scheduling optimization:
1. Constraint Validator
2. Schedule Scorer
3. Schedule Generator
4. Staffing Analyzer
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
    Tool 1: Validates that a schedule satisfies all hard constraints.
    
    Input:
        - schedule: Dict[str, Dict[str, List[str]]]  
          Format: {shift_id: {patient_id: [employee_ids]}}
        - employees: List[Dict]
        - patients: List[Dict]
    
    Output:
        - valid: bool
        - violations: List[str] (empty if valid)
        - breakdown: Dict with patient vs employee violation counts
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
    Tool 2: Scores a valid schedule on soft constraints (lower is better).
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
    Tool 3: Generates candidate schedules using heuristics.
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
        
        # Calculate expected monthly hours for each employee
        employee_monthly_budget = {}
        for emp_id, emp in self.employees.items():
            expected_weekly = emp.get('expected_hours_per_week', emp.get('max_hours_per_week', 40))
            employee_monthly_budget[emp_id] = expected_weekly * 4  # 4 weeks

        # Track how much budget is left
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
            
            # Sort patients by difficulty (hardest first)
            def patient_difficulty(patient_id):
                patient = self.patients[patient_id]
                return (
                    -patient.get('min_level', 1),
                    -len(patient.get('required_skills', [])),
                    -patient.get('nurses_needed', 1)
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

                # Build team iteratively with skill optimization
                assigned = []
                combined_skills = set()
                max_level_in_team = 0
                required_skills = set(patient.get('required_skills', []))
                min_nurses = patient['nurses_needed']
                max_nurses = min(self.MAX_NURSES_PER_PATIENT, len(eligible))

                # Iteratively build team, re-sorting after each addition
                for team_size in range(max_nurses):
                    if not eligible:
                        break
                    
                    # Re-sort based on current team composition
                    eligible.sort(key=lambda emp_id: self._employee_priority(
                        emp_id, patient, employee_hours, patient_nurse_assignments,
                        employee_hours_by_week,
                        already_assigned_to_patient=assigned
                    ))
                    
                    # Pick best remaining employee
                    best_emp = eligible.pop(0)
                    emp = self.employees[best_emp]
                    
                    assigned.append(best_emp)
                    combined_skills.update(emp['skills'])
                    max_level_in_team = max(max_level_in_team, emp['level'])
                    
                    # Check if we can stop early (met all requirements)
                    has_min_nurses = len(assigned) >= min_nurses
                    has_min_level = max_level_in_team >= patient.get('min_level', 1)
                    has_all_skills = not required_skills or required_skills.issubset(combined_skills)
                    
                    if has_min_nurses and has_min_level and has_all_skills:
                        # All requirements met, don't add more nurses
                        break

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
        
        # NEW: Final validation - log budget utilization success
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
        """Calculate priority score for employee assignment."""
        emp = self.employees[emp_id]
        patient_id = patient['patient_id']
        patient_min_level = patient.get('min_level', 1)
        patient_required_skills = set(patient.get('required_skills', []))
        
        # NEW: Calculate weekly imbalance penalty
        if employee_hours_by_week:
            weeks_hours = [employee_hours_by_week[emp_id][w] for w in range(4)]
            if weeks_hours and any(h > 0 for h in weeks_hours):
                avg_week = sum(weeks_hours) / 4
                max_deviation = max(abs(h - avg_week) for h in weeks_hours)
                weekly_imbalance_penalty = max_deviation * 2  # Penalize uneven distribution
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

        # CRITICAL: Strong penalty if approaching max (reserve capacity for later shifts)
        if approaching_max_ratio > 0.9:
            overwork_penalty = 10000  # Almost at max - avoid unless desperate
        elif approaching_max_ratio > 0.75:
            overwork_penalty = 5000   # Getting high - prefer others
        elif budget_used_ratio > 1.2:
            overwork_penalty = 1000   # Over expected by 20%
        elif budget_used_ratio > 1.0:
            overwork_penalty = 100    # Over expected slightly
        else:
            overwork_penalty = 0      # Within expected
        
        # NEW: Calculate skill efficiency when building a TEAM
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
            -skill_overlap,                  # Priority 2: Prefer nurses with required skills
            level_overqualified,             # Priority 3: Avoid overqualified (expensive)
            skill_duplication_penalty,       # Priority 4: Avoid pure duplication
            skill_waste_penalty,             # Priority 5: Minimize unused skills  
            overwork_penalty,                # Priority 6: CRITICAL - Avoid exhausting employees early
            approaching_max_ratio,           # Priority 7: Prefer employees with more hours remaining
            hours_worked,                    # Priority 8: Load balance
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
            
            # Check hour limits (allow exceeding max, but penalize in priority)
            # Don't hard-filter here - let priority function handle it
            # This allows patient coverage even if it means overtime
            
            # Check consecutive shifts
            if len(employee_shift_history[emp_id]) >= 2:
                recent_shifts = employee_shift_history[emp_id][-2:]
                if self._would_exceed_consecutive(recent_shifts, shift_id):
                    continue
            
            # Check weekly balance - don't let any week exceed weekly target by more than 4 hours
            week_num = self._get_week_number(shift_id)
            weekly_target = employee['max_hours_per_week']
            current_week_hours = employee_hours_by_week[emp_id][week_num]
            
            if current_week_hours + 4 > weekly_target + 4:  # Allow 4h buffer
                continue
            
            eligible.append(emp_id)
        
        # Emergency: Include ALL employees except those with hard exclusions
        # Patient coverage is priority #1, max_hours is priority #2 (can violate)
        if not eligible:
            emergency_eligible = []
            for emp_id, employee in self.employees.items():
                # Only check HARD exclusions
                if emp_id in patient['excluded_employees']:
                    continue
                if patient_id in employee.get('excluded_patients', []):
                    continue
                
                # Don't check max_hours - patient coverage is more important
                # Check if they can work this specific shift without violating consecutive limits
                if len(employee_shift_history[emp_id]) >= 2:
                    recent_shifts = employee_shift_history[emp_id][-2:]
                    if self._would_exceed_consecutive(recent_shifts, shift_id):
                        continue
                
                emergency_eligible.append(emp_id)
            
            if emergency_eligible:
                print(f"🚨 Emergency override activated for shift {shift_id}: using {len(emergency_eligible)} near-max employees")
                eligible = emergency_eligible
        
        # NEW: Boost underutilized employees to front of list
        # This ensures they get considered before overworked employees
        eligible_with_hours = [(emp_id, employee_hours.get(emp_id, 0)) for emp_id in eligible]
        eligible_with_hours.sort(key=lambda x: x[1])  # Sort by hours worked (ascending)
        eligible = [emp_id for emp_id, _ in eligible_with_hours]
        
        return eligible
    
    def _would_exceed_consecutive(self, recent_shifts: List[str], new_shift: str) -> bool:
        """Check if adding new_shift would create 4+ consecutive shifts."""
        if len(recent_shifts) < 2:
            return False
        
        def parse_shift(s):
            day = int(s[1:3])
            shift_num = int(s[4:])
            return (day, shift_num)
        
        prev1 = parse_shift(recent_shifts[-1])
        prev2 = parse_shift(recent_shifts[-2])
        new = parse_shift(new_shift)
        
        # Simple heuristic: if all same day and sequential
        if prev2[0] == prev1[0] == new[0]:
            if new[1] == prev1[1] + 1 and prev1[1] == prev2[1] + 1:
                return True
        
        return False
    
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
            
            elif "needs" in violation and "nurses" in violation and "but only" in violation:
                patient_id, shift_id, needed = self._parse_insufficient_violation(violation, schedule)
                if patient_id and shift_id and needed > 0:
                    insufficient_shifts.append((shift_id, patient_id, needed))
            
            elif "requires skills" in violation:
                patient_id, shift_id = self._parse_skill_gap_violation(violation)
                if patient_id and shift_id:
                    skill_gap_shifts.append((shift_id, patient_id))
            
            elif "requires level" in violation:
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
            relax_availability=False
        )
        
        team = self._build_optimal_team(eligible, patient, min_nurses)
        
        # PASS 2: Relax availability if needed
        if not self._team_is_valid(team, patient):
            eligible = self._find_candidates_for_shift(
                shift_id, patient_id, employee_hours, shift_employee_assignments,
                relax_availability=True
            )
            team = self._build_optimal_team(eligible, patient, min_nurses)
        
        # PASS 3: Still relaxing availability only (hours remain hard constraint)
        if not self._team_is_valid(team, patient):
            eligible = self._find_candidates_for_shift(
                shift_id, patient_id, employee_hours, shift_employee_assignments,
                relax_availability=True
            )
            team = self._build_optimal_team(eligible, patient, min_nurses)
        
        # PASS 4: DESPERATE - ignore almost everything except hard exclusions
        if not self._team_is_valid(team, patient):
            eligible = []
            for emp_id, emp in self.employees.items():
                # Only check HARD constraints
                if emp_id in shift_employee_assignments[shift_id]:
                    continue
                if patient_id in emp['excluded_patients']:
                    continue
                if emp_id in patient['excluded_employees']:
                    continue
                eligible.append(emp_id)
            
            team = self._build_optimal_team(eligible, patient, min_nurses)
        
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
                                   relax_availability: bool = False) -> List[str]:
        """Find candidate employees for a shift with optional constraint relaxation."""
        eligible = []
        patient = self.patients[patient_id]
        
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
            
            # Check max_hours as soft constraint - prefer not to exceed, but allow for patient coverage
            # Note: We don't filter out employees over max_hours here
            # The priority/sorting will prefer underutilized employees, but won't exclude overworked ones
            # This ensures patient coverage is prioritized over employee hour preferences
            
            eligible.append(emp_id)
        
        # Sort by hours worked (prefer underutilized)
        eligible.sort(key=lambda e: employee_hours.get(e, 0))
        
        return eligible

    def _build_optimal_team(self, eligible: List[str], patient: Dict, min_nurses: int) -> List[str]:
        """Build an optimal team from eligible employees."""
        if len(eligible) < min_nurses:
            return []
        
        team = []
        combined_skills = set()
        max_level = 0
        required_skills = set(patient.get('required_skills', []))
        min_level = patient.get('min_level', 1)
        
        # Strategy: Pick nurses that together cover all requirements
        # Sort by usefulness for this patient
        def nurse_value(emp_id):
            emp = self.employees[emp_id]
            emp_skills = set(emp['skills'])
            new_skills = len((emp_skills & required_skills) - combined_skills)
            meets_level = emp['level'] >= min_level
            return (-meets_level, -new_skills, -emp['level'], len(emp_skills))
        
        # Greedy team building
        for emp_id in sorted(eligible, key=nurse_value):
            emp = self.employees[emp_id]
            
            team.append(emp_id)
            combined_skills.update(emp['skills'])
            max_level = max(max_level, emp['level'])
            
            # Check if requirements met
            has_min_nurses = len(team) >= min_nurses
            has_level = max_level >= min_level
            has_skills = not required_skills or required_skills.issubset(combined_skills)
            
            if has_min_nurses and has_level and has_skills:
                break
            
            if len(team) >= self.MAX_NURSES_PER_PATIENT:
                break
        
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
            relax_availability=True
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
            additional = eligible[:max_additional]
            
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
    Tool 4: Analyzes staffing needs and suggests hiring/termination recommendations.
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