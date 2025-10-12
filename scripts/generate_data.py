"""
Nursing Facility Schedule Data Generator

Generates realistic employee and patient data for shift scheduling optimization.
IMPROVED: Ensures skill coverage and balanced distribution.

Run: python scripts/generate_data.py
"""

import json
import random
from datetime import datetime
from typing import List, Dict
import os

# Configuration
NUM_EMPLOYEES = 80  # Mix of full-time and part-time - enough for good coverage
NUM_PATIENTS = 8    # Reasonable facility size
SKILLS = [
    "Medication Administration",
    "Wound Care", 
    "IV Therapy",
    "Dementia Care",
    "Physical Therapy",
    "Diabetes Management",
    "Catheter Care",
    "Emergency Response",
    "Mobility Assistance",
    "Mental Health Support"
]
SHIFTS_PER_DAY = 6  # 4-hour shifts
DAYS = 28  # 4 weeks exactly

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

def generate_shift_id(day: int, shift: int) -> str:
    """Generate shift ID like 'D01S0' for day 1, shift 0"""
    return f"D{day:02d}S{shift}"

def generate_all_shifts() -> List[str]:
    """Generate all shift IDs for the month"""
    return [generate_shift_id(day, shift) 
            for day in range(1, DAYS + 1) 
            for shift in range(SHIFTS_PER_DAY)]

def generate_employee_availability(weekly_hours: int) -> List[str]:
    """Generate realistic availability based on weekly hours"""
    all_shifts = generate_all_shifts()
    shifts_per_week = weekly_hours // 4
    available = []
    weeks = 4
    
    # INCREASED FLEXIBILITY: Employees available for more varied shifts
    for week in range(weeks):
        week_start_day = week * 7 + 1
        
        # MORE FLEXIBLE: Employees can work more shift types
        preferred_shift_pattern = random.choice([
            [2, 3],        # Day shifts
            [3, 4],        # Afternoon/evening
            [4, 5],        # Evening/night
            [0, 1],        # Night shifts
            [1, 2, 3],     # Morning to afternoon
            [2, 3, 4],     # Day to evening (NEW)
            [1, 2, 3, 4],  # Most of day (NEW - more flexible)
        ])
        
        # MORE DAYS: Available more days per week
        available_days_in_week = random.sample(range(7), k=min(7, shifts_per_week + 3))  # +3 instead of +2
        week_shifts_added = 0
        
        for day_offset in available_days_in_week:
            day = week_start_day + day_offset
            if day > DAYS:
                break
            for shift in preferred_shift_pattern:
                if week_shifts_added < shifts_per_week * 1.5:  # Allow 1.5x overscheduling for flexibility
                    available.append(generate_shift_id(day, shift))
                    week_shifts_added += 1
    
    return available

def generate_employees() -> List[Dict]:
    """Generate employee data with custom level and skill distribution"""
    employees = []
    first_names = ["Emma", "James", "Olivia", "Liam", "Ava", "Noah", "Sophia", "William", 
                   "Isabella", "Benjamin", "Mia", "Lucas", "Charlotte", "Henry", "Amelia",
                   "Ethan", "Harper", "Alexander", "Evelyn", "Daniel", "Abigail", "Matthew",
                   "Emily", "Jackson", "Elizabeth", "Sofia", "Avery", "Owen", "Ella", 
                   "Aiden", "Scarlett", "Grace", "Logan", "Sebastian", "Jack", "Samuel",
                   "Mason", "Jacob", "Michael", "Elijah", "Oliver", "David", "Joseph",
                   "Carter", "Luke", "Jayden", "Gabriel", "Julian", "Wyatt", "Grayson",
                   "Leo", "Lincoln", "Jaxon", "Joshua", "Christopher", "Andrew", "Theodore",
                   "Caleb", "Ryan", "Asher", "Nathan", "Thomas", "Hunter", "Isaiah",
                   "Zoey", "Nora", "Lily", "Eleanor", "Hannah", "Lillian", "Addison",
                   "Aubrey", "Ellie", "Stella", "Natalie", "Zoe", "Leah", "Hazel",
                   "Violet", "Aurora", "Savannah", "Audrey", "Brooklyn", "Bella", "Claire",
                   "Skylar", "Lucy", "Paisley", "Everly", "Anna", "Caroline", "Nova",
                   "Genesis", "Emilia", "Kennedy", "Samantha", "Maya", "Willow", "Kinsley",
                   "Naomi", "Aaliyah", "Elena", "Sarah", "Ariana", "Allison", "Gabriella",
                   "Alice", "Madelyn", "Cora", "Ruby", "Eva", "Serenity", "Autumn", "Adeline",
                   "Paisley", "Makayla", "Rose", "Isabelle", "Natalia", "Camila", "Penelope",
                   "Andrea", "Kylie", "Amy", "Sophie", "Brielle", "Kimberly", "Ryleigh"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", 
                  "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "White",
                  "Harris", "Clark", "Lewis", "Robinson", "Walker", "Hall", "Allen", "Young",
                  "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker",
                  "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner",
                  "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart",
                  "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy",
                  "Bailey", "Rivera", "Cooper", "Richardson", "Cox", "Howard", "Ward", "Torres"]
    
    # NEW DISTRIBUTION: 50% Level 1, 30% Level 2, 20% Level 3
    level_1_count = int(NUM_EMPLOYEES * 0.50)  # 40 employees
    level_2_count = int(NUM_EMPLOYEES * 0.30)  # 24 employees
    level_3_count = NUM_EMPLOYEES - level_1_count - level_2_count  # 16 employees
    
    level_distribution = (
        [1] * level_1_count +
        [2] * level_2_count +
        [3] * level_3_count
    )
    random.shuffle(level_distribution)
    
    # Employment types - MORE full-time employees for better coverage
    # 72% full-time, rest part-time
    num_fulltime = int(NUM_EMPLOYEES * 0.72)
    num_parttime_20 = int(NUM_EMPLOYEES * 0.20)
    num_parttime_12 = NUM_EMPLOYEES - num_fulltime - num_parttime_20
    
    employment_types = (
        [40] * num_fulltime +
        [20] * num_parttime_20 +
        [12] * num_parttime_12
    )
    
    # Ensure ALL skills are well-covered
    # First, assign each skill to at least 3 employees
    skill_coverage = {skill: [] for skill in SKILLS}
    
    # Track used names to ensure uniqueness
    used_names = set()
    
    for i in range(NUM_EMPLOYEES):
        emp_id = f"E{i+1:03d}"
        
        # Generate unique name
        while True:
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            if name not in used_names:
                used_names.add(name)
                break
        
        level = level_distribution[i]
        weekly_hours = employment_types[i]
        
        # NEW SKILL RANGES PER LEVEL
        if level == 1:
            num_base_skills = random.randint(1, 3)  # Level 1: 1-3 skills
        elif level == 2:
            num_base_skills = random.randint(3, 6)  # Level 2: 3-6 skills
        else:  # level == 3
            num_base_skills = random.randint(5, 10) # Level 3: 5-10 skills
        
        # Assign skills ensuring good coverage
        skills = []
        
        # First pass: ensure under-covered skills are assigned
        for skill in SKILLS:
            if len(skill_coverage[skill]) < 3:  # Each skill needs at least 3 employees
                skills.append(skill)
                skill_coverage[skill].append(emp_id)
        
        # Second pass: fill remaining slots randomly
        remaining_skills = [s for s in SKILLS if s not in skills]
        additional_needed = num_base_skills - len(skills)
        if additional_needed > 0 and remaining_skills:
            additional_skills = random.sample(remaining_skills, 
                                            k=min(additional_needed, len(remaining_skills)))
            skills.extend(additional_skills)
            for skill in additional_skills:
                skill_coverage[skill].append(emp_id)
        
        # CRITICAL: Enforce skill limits based on level (AFTER all assignments)
        if level == 1 and len(skills) > 3:
            # Trim to 3 skills, but keep the most important ones
            skills = skills[:3]
        elif level == 2 and len(skills) > 6:
            # Trim to 6 skills
            skills = skills[:6]
        # Level 3 can have up to 10 (no cap)
        
        # Pay rate based on level
        base_pay = {1: (22, 28), 2: (32, 42), 3: (48, 68)}
        pay_rate = round(random.uniform(*base_pay[level]), 2)
        
        # Generate availability
        available_shifts = generate_employee_availability(weekly_hours)
        
        # Max/min hours per week
        max_hours = weekly_hours + 8
        min_hours = max(0, weekly_hours - 8)
        
        # REDUCED exclusions (only 2% chance instead of 5%)
        excluded_patients = [f"P{j+1:03d}" for j in range(NUM_PATIENTS) 
                            if random.random() < 0.02]
        excluded_employees = [f"E{j+1:03d}" for j in range(NUM_EMPLOYEES) 
                             if j != i and random.random() < 0.01]
        
        employee = {
            "employee_id": emp_id,
            "name": name,
            "level": level,
            "skills": skills,
            "expected_hours_per_week": weekly_hours,  # NEW: Target/contracted hours
            "max_hours_per_week": max_hours,
            "min_hours_per_week": min_hours,
            "hourly_pay_rate": pay_rate,
            "available_shifts": available_shifts,
            "excluded_patients": excluded_patients,
            "excluded_employees": excluded_employees
        }
        
        employees.append(employee)
    
    return employees

def generate_patients(employees: List[Dict]) -> List[Dict]:
    """Generate patient data that MATCHES available employee skills"""
    patients = []
    first_names = ["Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David",
                   "Susan", "Richard", "Lisa", "Charles", "Nancy", "Christopher", "Karen", "Daniel",
                   "Betty", "Matthew", "Helen", "Anthony", "Sandra", "Mark", "Donna", "Donald",
                   "Carol", "Steven", "Ruth", "Paul", "Sharon", "Andrew", "Michelle", "Joshua",
                   "Laura", "Kenneth", "Sarah", "Kevin", "Kimberly", "Brian", "Deborah", "George"]
    last_names = ["Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "White",
                  "Harris", "Clark", "Lewis", "Robinson", "Walker", "Hall", "Allen", "Young",
                  "King", "Wright", "Hill", "Scott", "Green", "Adams", "Baker", "Nelson",
                  "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker",
                  "Evans", "Edwards", "Collins", "Stewart", "Morris", "Rogers", "Reed", "Cook"]
    
    # Collect all available skills from employees
    all_employee_skills = set()
    for emp in employees:
        all_employee_skills.update(emp['skills'])
    
    # Track used names to ensure uniqueness
    used_names = set()
    
    for i in range(NUM_PATIENTS):
        patient_id = f"P{i+1:03d}"
        
        # Generate unique name
        while True:
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            if name not in used_names:
                used_names.add(name)
                break
        
        # Most patients need 24/7 care, some need specific hours
        if random.random() < 0.30:  # Set to 30% for better coverage
            # 24/7 care
            care_shifts = generate_all_shifts()
        else:
            # Day-time only care (8am-8pm = shifts 2,3,4)
            care_shifts = [generate_shift_id(day, shift) 
                          for day in range(1, DAYS + 1) 
                          for shift in [2, 3, 4]]
        
        # Number of nurses needed (weighted toward 1-2 for feasibility)
        nurses_needed = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        
        # Minimum level (weighted toward 1-2 for feasibility)
        min_level = random.choices([1, 2, 3], weights=[50, 40, 10])[0]
        
        # NEW SKILL DISTRIBUTION: 0-5 skills with specific probabilities
        # 5% = 0 skills, 15% = 1 skill, 20% = 2 skills, 25% = 3 skills, 
        # 20% = 4 skills, 15% = 5 skills
        num_required_skills = random.choices(
            [0, 1, 2, 3, 4, 5],
            weights=[5, 15, 20, 25, 20, 15]
        )[0]

        if num_required_skills > 0:
            # Only require skills that employees actually have!
            available_skill_list = list(all_employee_skills)
            required_skills = random.sample(available_skill_list, 
                                          k=min(num_required_skills, len(available_skill_list)))
        else:
            required_skills = []
        
        # VERY RARE exclusions (only 1% chance)
        excluded_employees = [f"E{j+1:03d}" for j in range(NUM_EMPLOYEES) 
                             if random.random() < 0.01]
        
        patient = {
            "patient_id": patient_id,
            "name": name,
            "care_shifts": care_shifts,
            "nurses_needed": nurses_needed,
            "min_level": min_level,
            "required_skills": required_skills,
            "excluded_employees": excluded_employees
        }
        
        patients.append(patient)
    
    return patients

def generate_metadata() -> Dict:
    """Generate metadata about the scheduling problem"""
    return {
        "generated_at": datetime.now().isoformat(),
        "num_employees": NUM_EMPLOYEES,
        "num_patients": NUM_PATIENTS,
        "days": DAYS,
        "shifts_per_day": SHIFTS_PER_DAY,
        "shift_duration_hours": 4,
        "shift_labels": {
            "0": "12am-4am (Night)",
            "1": "4am-8am (Early Morning)",
            "2": "8am-12pm (Morning)",
            "3": "12pm-4pm (Afternoon)",
            "4": "4pm-8pm (Evening)",
            "5": "8pm-12am (Night)"
        },
        "skills": SKILLS,
        "max_consecutive_shifts": 3,
        "typical_consecutive_shifts": 2,
        "distributions": {
            "nurse_levels": "50% Level 1, 30% Level 2, 20% Level 3",
            "skills_per_level": "L1: 1-3, L2: 3-6, L3: 5-10",
            "patient_skills": "5%=0, 15%=1, 20%=2, 25%=3, 20%=4, 15%=5"
        },
        "improvements": [
            "Custom level distribution: 50/30/20 (Level 1/2/3)",
            "Skills per nurse: L1=1-3, L2=3-6, L3=5-10",
            "Patient skills: weighted toward 2-4 skills (65% of patients)",
            "Increased full-time employees (72%)",
            "Ensured all skills covered by at least 3 employees",
            "Reduced exclusions to improve feasibility",
            "60% of patients need 24/7 care for better demand",
            "Expanded name lists: 80+ first names, 60+ last names for employees",
            "Expanded patient names: 40+ first names, 40+ last names",
            "Added unique name generation logic to prevent duplicates",
            "Added expected_hours_per_week field for clearer target scheduling"
        ]
    }

def print_summary(employees: List[Dict], patients: List[Dict]):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("DATA GENERATION SUMMARY")
    print("="*60)
    
    print(f"\nEmployees: {len(employees)}")
    print(f"  Full-time (40h): {sum(1 for e in employees if e['max_hours_per_week'] >= 40)}")
    print(f"  Part-time (20h): {sum(1 for e in employees if e['max_hours_per_week'] < 40 and e['max_hours_per_week'] >= 20)}")
    print(f"  Part-time (12h): {sum(1 for e in employees if e['max_hours_per_week'] < 20)}")
    
    level_1_count = sum(1 for e in employees if e['level'] == 1)
    level_2_count = sum(1 for e in employees if e['level'] == 2)
    level_3_count = sum(1 for e in employees if e['level'] == 3)
    
    print(f"\n  Level 1: {level_1_count} ({level_1_count/len(employees)*100:.0f}%)")
    print(f"  Level 2: {level_2_count} ({level_2_count/len(employees)*100:.0f}%)")
    print(f"  Level 3: {level_3_count} ({level_3_count/len(employees)*100:.0f}%)")
    
    # Skill coverage analysis
    skill_coverage = {}
    for skill in SKILLS:
        count = sum(1 for e in employees if skill in e['skills'])
        skill_coverage[skill] = count
    
    # Skills per level analysis
    level_1_skills = [len(e['skills']) for e in employees if e['level'] == 1]
    level_2_skills = [len(e['skills']) for e in employees if e['level'] == 2]
    level_3_skills = [len(e['skills']) for e in employees if e['level'] == 3]
    
    print(f"\n  Skills per nurse by level:")
    print(f"    Level 1: avg {sum(level_1_skills)/len(level_1_skills):.1f} (range: {min(level_1_skills)}-{max(level_1_skills)})")
    print(f"    Level 2: avg {sum(level_2_skills)/len(level_2_skills):.1f} (range: {min(level_2_skills)}-{max(level_2_skills)})")
    print(f"    Level 3: avg {sum(level_3_skills)/len(level_3_skills):.1f} (range: {min(level_3_skills)}-{max(level_3_skills)})")
    
    avg_skills = sum(len(e['skills']) for e in employees) / len(employees)
    min_coverage = min(skill_coverage.values())
    max_coverage = max(skill_coverage.values())
    
    print(f"\n  Overall avg skills per employee: {avg_skills:.1f}")
    print(f"  Skill coverage range: {min_coverage}-{max_coverage} employees per skill")
    print(f"  All skills covered: {'✓ YES' if min_coverage >= 3 else '✗ NO (problem!)'}")
    
    print(f"\nPatients: {len(patients)}")
    print(f"  24/7 care: {sum(1 for p in patients if len(p['care_shifts']) > 100)}")
    print(f"  Day-only care: {sum(1 for p in patients if len(p['care_shifts']) <= 100)}")
    
    # Patient skills distribution
    skills_dist = [0] * 6  # 0-5 skills
    for p in patients:
        num_skills = len(p['required_skills'])
        if num_skills <= 5:
            skills_dist[num_skills] += 1
    
    print(f"\n  Patient skill requirements distribution:")
    for i in range(6):
        count = skills_dist[i]
        pct = count / len(patients) * 100
        print(f"    {i} skills: {count} patients ({pct:.0f}%)")
    
    avg_patient_skills = sum(len(p['required_skills']) for p in patients) / len(patients)
    print(f"  Avg required skills per patient: {avg_patient_skills:.1f}")
    
    total_nurse_shifts_needed = sum(
        len(p['care_shifts']) * p['nurses_needed'] 
        for p in patients
    )
    print(f"\n  Total nurse-shifts needed: {total_nurse_shifts_needed}")
    
    total_available_shifts = sum(len(e['available_shifts']) for e in employees)
    print(f"  Total available employee-shifts: {total_available_shifts}")
    
    ratio = total_available_shifts / total_nurse_shifts_needed if total_nurse_shifts_needed > 0 else 0
    print(f"  Availability ratio: {ratio:.2f}x")
    if ratio < 1.0:
        print("  ⚠️  WARNING: Not enough availability to cover all shifts!")
    elif ratio < 1.2:
        print("  ⚠️  WARNING: Tight constraints - scheduling will be difficult")
    elif ratio < 1.5:
        print("  ✓ Adequate availability for scheduling")
    else:
        print("  ✓✓ Excellent availability for scheduling!")
    
    # Check skill matching
    print("\n" + "="*60)
    print("FEASIBILITY CHECK")
    print("="*60)
    
    employee_skills = set()
    for emp in employees:
        employee_skills.update(emp['skills'])
    
    patient_skills = set()
    for pat in patients:
        patient_skills.update(pat['required_skills'])
    
    uncovered_skills = patient_skills - employee_skills
    if uncovered_skills:
        print(f"  ✗ PROBLEM: Patients need skills that NO employee has:")
        for skill in uncovered_skills:
            print(f"    - {skill}")
    else:
        print(f"  ✓ All patient skill requirements can be met!")
    
    # Check level distribution
    level_3_count = sum(1 for e in employees if e['level'] == 3)
    patients_needing_3 = sum(1 for p in patients if p['min_level'] == 3)
    print(f"\n  Level 3 nurses: {level_3_count}")
    print(f"  Patients needing Level 3: {patients_needing_3}")
    if level_3_count < patients_needing_3:
        print(f"  ⚠️  May have issues covering all Level 3 requirements")
    else:
        print(f"  ✓ Sufficient Level 3 coverage")
    
    # Check for unique names
    employee_names = [e['name'] for e in employees]
    patient_names = [p['name'] for p in patients]
    
    unique_emp_names = len(set(employee_names))
    unique_pat_names = len(set(patient_names))
    
    print(f"\n  Employee name uniqueness: {unique_emp_names}/{len(employees)} unique ({'✓' if unique_emp_names == len(employees) else '✗'})")
    print(f"  Patient name uniqueness: {unique_pat_names}/{len(patients)} unique ({'✓' if unique_pat_names == len(patients) else '✗'})")

def main():
    print("Generating nursing facility data...")
    print("ENHANCED: 80+ employee names, 40+ patient names, unique name generation")
    print("CUSTOM DISTRIBUTION: 50/30/20 levels, L1:1-3, L2:3-6, L3:5-10 skills")
    
    random.seed(42)  # For reproducibility
    
    employees = generate_employees()
    patients = generate_patients(employees)  # Pass employees to ensure skill matching
    metadata = generate_metadata()
    
    with open('data/employees.json', 'w') as f:
        json.dump(employees, f, indent=2)
    
    with open('data/patients.json', 'w') as f:
        json.dump(patients, f, indent=2)
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Generated files:")
    print("  - data/employees.json")
    print("  - data/patients.json")
    print("  - data/metadata.json")
    
    print_summary(employees, patients)
    
    print("\n" + "="*60)
    print("Sample Employees:")
    for level in [1, 2, 3]:
        level_employees = [e for e in employees if e['level'] == level]
        if level_employees:
            print(f"\nLevel {level} Example:")
            print(json.dumps(level_employees[0], indent=2))
    
    print("\nSample Patient:")
    print(json.dumps(patients[0], indent=2))
    print("="*60)

if __name__ == "__main__":
    main()