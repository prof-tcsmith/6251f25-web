"""
Generate synthetic healthcare data for Week 5 Assignment
This script creates a realistic patient readmission dataset for classification analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_healthcare_data(n_patients=2000):
    """Generate synthetic patient readmission data with realistic relationships"""
    
    patients = []
    
    # Define hospital departments
    departments = ['Emergency', 'Cardiology', 'Orthopedics', 'General Medicine', 
                   'Surgery', 'Neurology', 'Pulmonology']
    
    # Insurance types
    insurance_types = ['Private', 'Medicare', 'Medicaid', 'Self-Pay']
    
    # Admission types
    admission_types = ['Emergency', 'Urgent', 'Elective', 'Newborn']
    
    # Diagnosis categories (simplified)
    diagnosis_categories = ['Circulatory', 'Respiratory', 'Digestive', 'Injury', 
                           'Musculoskeletal', 'Nervous', 'Endocrine', 'Infectious']
    
    for i in range(n_patients):
        # Patient demographics
        age = np.random.gamma(4, 15)  # Skewed towards older patients
        age = min(100, max(18, int(age)))
        
        # Gender
        gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
        
        # Insurance (correlated with age)
        if age >= 65:
            insurance = np.random.choice(insurance_types, p=[0.2, 0.6, 0.15, 0.05])
        elif age <= 30:
            insurance = np.random.choice(insurance_types, p=[0.5, 0.05, 0.3, 0.15])
        else:
            insurance = np.random.choice(insurance_types, p=[0.6, 0.1, 0.2, 0.1])
        
        # Admission type (affects readmission risk)
        admission_type = np.random.choice(admission_types, p=[0.35, 0.3, 0.3, 0.05])
        
        # Department
        department = np.random.choice(departments)
        
        # Diagnosis category
        diagnosis = np.random.choice(diagnosis_categories)
        
        # Length of stay (days) - depends on admission type and age
        if admission_type == 'Emergency':
            los_base = np.random.gamma(2, 3)
        elif admission_type == 'Urgent':
            los_base = np.random.gamma(2, 2.5)
        elif admission_type == 'Elective':
            los_base = np.random.gamma(2, 2)
        else:  # Newborn
            los_base = np.random.gamma(2, 1.5)
        
        # Age adjustment for length of stay
        age_factor = 1 + (age - 50) * 0.01 if age > 50 else 1
        length_of_stay = max(1, int(los_base * age_factor))
        
        # Number of previous admissions
        if age < 30:
            prev_admissions = np.random.poisson(0.5)
        elif age < 50:
            prev_admissions = np.random.poisson(1.5)
        elif age < 70:
            prev_admissions = np.random.poisson(3)
        else:
            prev_admissions = np.random.poisson(5)
        
        # Number of procedures during stay
        if admission_type == 'Surgery':
            num_procedures = np.random.poisson(3) + 1
        elif admission_type == 'Emergency':
            num_procedures = np.random.poisson(2)
        else:
            num_procedures = np.random.poisson(1)
        
        # Number of diagnoses (comorbidities)
        if age < 40:
            num_diagnoses = np.random.poisson(2) + 1
        elif age < 60:
            num_diagnoses = np.random.poisson(3) + 1
        else:
            num_diagnoses = np.random.poisson(4) + 2
        
        # Number of medications
        num_medications = int(num_diagnoses * np.random.uniform(2, 4))
        
        # Lab results (simplified as abnormal count)
        num_lab_tests = np.random.poisson(10) + 5
        abnormal_lab_results = np.random.binomial(num_lab_tests, 0.3)
        
        # Emergency visits in last year
        if prev_admissions > 3:
            er_visits_year = np.random.poisson(4)
        else:
            er_visits_year = np.random.poisson(1)
        
        # Discharge disposition
        if age > 75 and length_of_stay > 7:
            discharge_options = ['Home', 'SNF', 'Rehab', 'Home Health']
            discharge_weights = [0.3, 0.3, 0.2, 0.2]
        else:
            discharge_options = ['Home', 'SNF', 'Rehab', 'Home Health']
            discharge_weights = [0.7, 0.1, 0.1, 0.1]
        discharge_disposition = np.random.choice(discharge_options, p=discharge_weights)
        
        # Has primary care physician
        has_pcp = np.random.choice([0, 1], p=[0.2, 0.8])
        
        # Missed appointments (higher risk factor)
        if has_pcp:
            missed_appointments = np.random.poisson(0.5)
        else:
            missed_appointments = np.random.poisson(2)
        
        # Charlson Comorbidity Index (simplified)
        cci_base = 0
        if age > 50:
            cci_base += (age - 50) // 10
        if diagnosis in ['Circulatory', 'Respiratory', 'Endocrine']:
            cci_base += 2
        if num_diagnoses > 5:
            cci_base += 2
        charlson_index = min(10, cci_base + np.random.poisson(1))
        
        # Risk scores
        fall_risk = 1 if (age > 65 and num_medications > 10) else 0
        
        # Social factors
        lives_alone = 1 if (age > 65 and np.random.random() < 0.3) else 0
        
        # Distance from hospital (miles)
        distance_from_hospital = np.random.gamma(2, 10)
        
        # Vital signs at discharge (simplified as stability score 0-10)
        discharge_vitals_stable = np.random.uniform(5, 10)
        if admission_type == 'Emergency':
            discharge_vitals_stable -= 1
        
        # Follow-up appointment scheduled
        followup_scheduled = 1 if np.random.random() < 0.7 else 0
        
        # Calculate readmission probability based on risk factors
        readmission_prob = 0.1  # Base probability
        
        # Age factor
        if age > 65:
            readmission_prob += 0.1
        if age > 80:
            readmission_prob += 0.15
        
        # Clinical factors
        if admission_type == 'Emergency':
            readmission_prob += 0.15
        if length_of_stay > 7:
            readmission_prob += 0.1
        if prev_admissions > 2:
            readmission_prob += 0.2
        if num_diagnoses > 5:
            readmission_prob += 0.15
        if abnormal_lab_results > 5:
            readmission_prob += 0.1
        if discharge_disposition != 'Home':
            readmission_prob += 0.1
        if charlson_index > 3:
            readmission_prob += 0.15
        
        # Social/system factors
        if not has_pcp:
            readmission_prob += 0.1
        if missed_appointments > 2:
            readmission_prob += 0.15
        if lives_alone:
            readmission_prob += 0.05
        if not followup_scheduled:
            readmission_prob += 0.1
        if discharge_vitals_stable < 7:
            readmission_prob += 0.1
        
        # Insurance factor
        if insurance == 'Self-Pay':
            readmission_prob += 0.05
        elif insurance == 'Medicaid':
            readmission_prob += 0.03
        
        # Cap probability
        readmission_prob = min(0.9, readmission_prob)
        
        # Generate readmission outcome
        readmitted_30days = 1 if np.random.random() < readmission_prob else 0
        
        # If readmitted, determine days to readmission
        if readmitted_30days:
            # Earlier readmission for higher risk patients
            if readmission_prob > 0.5:
                days_to_readmission = np.random.gamma(2, 5)
            else:
                days_to_readmission = np.random.gamma(3, 6)
            days_to_readmission = min(30, int(days_to_readmission))
        else:
            days_to_readmission = None
        
        # Admission and discharge dates
        admission_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 300))
        discharge_date = admission_date + timedelta(days=length_of_stay)
        
        patients.append({
            'patient_id': f'PAT{str(i+1).zfill(6)}',
            'age': age,
            'gender': gender,
            'insurance_type': insurance,
            'admission_type': admission_type,
            'admission_date': admission_date,
            'discharge_date': discharge_date,
            'length_of_stay': length_of_stay,
            'department': department,
            'diagnosis_category': diagnosis,
            'num_procedures': num_procedures,
            'num_diagnoses': num_diagnoses,
            'num_medications': num_medications,
            'num_lab_tests': num_lab_tests,
            'abnormal_lab_results': abnormal_lab_results,
            'previous_admissions': prev_admissions,
            'er_visits_last_year': er_visits_year,
            'discharge_disposition': discharge_disposition,
            'has_pcp': has_pcp,
            'missed_appointments': missed_appointments,
            'charlson_comorbidity_index': charlson_index,
            'fall_risk': fall_risk,
            'lives_alone': lives_alone,
            'distance_from_hospital': round(distance_from_hospital, 1),
            'discharge_vitals_stable': round(discharge_vitals_stable, 1),
            'followup_scheduled': followup_scheduled,
            'readmitted_30days': readmitted_30days,
            'days_to_readmission': days_to_readmission
        })
    
    return pd.DataFrame(patients)

def generate_lab_results(patient_df):
    """Generate detailed lab results for each patient"""
    
    lab_data = []
    
    lab_tests = {
        'Hemoglobin': {'normal_range': (12, 17), 'unit': 'g/dL'},
        'WBC': {'normal_range': (4.5, 11), 'unit': 'K/uL'},
        'Platelets': {'normal_range': (150, 400), 'unit': 'K/uL'},
        'Creatinine': {'normal_range': (0.6, 1.2), 'unit': 'mg/dL'},
        'BUN': {'normal_range': (7, 20), 'unit': 'mg/dL'},
        'Glucose': {'normal_range': (70, 100), 'unit': 'mg/dL'},
        'Sodium': {'normal_range': (136, 145), 'unit': 'mEq/L'},
        'Potassium': {'normal_range': (3.5, 5.0), 'unit': 'mEq/L'},
        'Calcium': {'normal_range': (8.5, 10.5), 'unit': 'mg/dL'},
        'Albumin': {'normal_range': (3.5, 5.0), 'unit': 'g/dL'}
    }
    
    for _, patient in patient_df.iterrows():
        num_abnormal = patient['abnormal_lab_results']
        tests_to_abnormal = random.sample(list(lab_tests.keys()), 
                                        min(num_abnormal, len(lab_tests)))
        
        for test_name, test_info in lab_tests.items():
            if test_name in tests_to_abnormal:
                # Generate abnormal value
                if np.random.random() < 0.5:
                    # Below normal
                    value = test_info['normal_range'][0] * np.random.uniform(0.5, 0.9)
                else:
                    # Above normal
                    value = test_info['normal_range'][1] * np.random.uniform(1.1, 1.5)
            else:
                # Generate normal value
                value = np.random.uniform(test_info['normal_range'][0], 
                                        test_info['normal_range'][1])
            
            lab_data.append({
                'patient_id': patient['patient_id'],
                'test_name': test_name,
                'value': round(value, 2),
                'unit': test_info['unit'],
                'is_abnormal': 1 if test_name in tests_to_abnormal else 0
            })
    
    return pd.DataFrame(lab_data)

def generate_medications(patient_df):
    """Generate medication data for each patient"""
    
    medications_list = [
        {'name': 'Metformin', 'class': 'Antidiabetic', 'high_risk': 0},
        {'name': 'Lisinopril', 'class': 'ACE Inhibitor', 'high_risk': 0},
        {'name': 'Atorvastatin', 'class': 'Statin', 'high_risk': 0},
        {'name': 'Warfarin', 'class': 'Anticoagulant', 'high_risk': 1},
        {'name': 'Furosemide', 'class': 'Diuretic', 'high_risk': 0},
        {'name': 'Insulin', 'class': 'Antidiabetic', 'high_risk': 1},
        {'name': 'Aspirin', 'class': 'Antiplatelet', 'high_risk': 0},
        {'name': 'Metoprolol', 'class': 'Beta Blocker', 'high_risk': 0},
        {'name': 'Omeprazole', 'class': 'PPI', 'high_risk': 0},
        {'name': 'Levothyroxine', 'class': 'Thyroid', 'high_risk': 0},
        {'name': 'Albuterol', 'class': 'Bronchodilator', 'high_risk': 0},
        {'name': 'Prednisone', 'class': 'Corticosteroid', 'high_risk': 1},
        {'name': 'Gabapentin', 'class': 'Anticonvulsant', 'high_risk': 0},
        {'name': 'Hydrocodone', 'class': 'Opioid', 'high_risk': 1},
        {'name': 'Amoxicillin', 'class': 'Antibiotic', 'high_risk': 0}
    ]
    
    medication_data = []
    
    for _, patient in patient_df.iterrows():
        num_meds = patient['num_medications']
        patient_meds = random.sample(medications_list, min(num_meds, len(medications_list)))
        
        for med in patient_meds:
            medication_data.append({
                'patient_id': patient['patient_id'],
                'medication_name': med['name'],
                'medication_class': med['class'],
                'high_risk_medication': med['high_risk']
            })
    
    return pd.DataFrame(medication_data)

if __name__ == "__main__":
    # Generate the main patient data
    patients_df = generate_healthcare_data(2000)
    
    # Generate lab results
    lab_results_df = generate_lab_results(patients_df)
    
    # Generate medications
    medications_df = generate_medications(patients_df)
    
    # Add some missing values for realism
    missing_indices = np.random.choice(patients_df.index, size=100, replace=False)
    patients_df.loc[missing_indices[:30], 'discharge_vitals_stable'] = np.nan
    patients_df.loc[missing_indices[30:60], 'missed_appointments'] = np.nan
    patients_df.loc[missing_indices[60:80], 'distance_from_hospital'] = np.nan
    patients_df.loc[missing_indices[80:], 'followup_scheduled'] = np.nan
    
    # Save to CSV files
    patients_df.to_csv('patient_readmissions.csv', index=False)
    lab_results_df.to_csv('patient_lab_results.csv', index=False)
    medications_df.to_csv('patient_medications.csv', index=False)
    
    print("Healthcare data generation complete!")
    print(f"Patients: {len(patients_df)} records")
    print(f"Lab results: {len(lab_results_df)} records")
    print(f"Medications: {len(medications_df)} records")
    
    # Print class balance
    readmission_rate = patients_df['readmitted_30days'].mean()
    print(f"\nReadmission Statistics:")
    print(f"  30-day readmission rate: {readmission_rate:.2%}")
    print(f"  Readmitted patients: {patients_df['readmitted_30days'].sum()}")
    print(f"  Not readmitted: {len(patients_df) - patients_df['readmitted_30days'].sum()}")
    
    # Print feature statistics
    print(f"\nKey Statistics:")
    print(f"  Average age: {patients_df['age'].mean():.1f} years")
    print(f"  Average length of stay: {patients_df['length_of_stay'].mean():.1f} days")
    print(f"  Average previous admissions: {patients_df['previous_admissions'].mean():.1f}")
    print(f"  Patients with PCP: {patients_df['has_pcp'].mean():.1%}")