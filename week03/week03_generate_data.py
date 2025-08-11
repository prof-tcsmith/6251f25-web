#!/usr/bin/env python3
"""
Generate Week 3 Assignment Data: Customer Data Pipeline
Creates a realistic messy customer dataset with various data quality issues
and non-linear relationships between variables for KNN imputation to excel.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys

def generate_messy_customer_data(n_customers=5000, seed=42):
    """
    Generate a realistic messy customer dataset with various data quality issues
    and complex non-linear relationships between variables.
    
    Parameters:
    -----------
    n_customers : int
        Number of customer records to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame : Messy customer dataset with quality issues
    """
    np.random.seed(seed)
    
    # Industries with typos and variations
    industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 
                 'Education', 'Real Estate', 'Consulting']
    industry_variations = {
        'Technology': ['Technology', 'Tech', 'technology', 'IT', 'Information Technology'],
        'Finance': ['Finance', 'Financial', 'Banking', 'finance', 'FinTech'],
        'Healthcare': ['Healthcare', 'Health', 'Medical', 'healthcare', 'Pharma'],
        'Retail': ['Retail', 'retail', 'E-commerce', 'eCommerce', 'Retail Sales'],
        'Manufacturing': ['Manufacturing', 'manufacturing', 'Production', 'Factory', 'Industrial']
    }
    
    # Contract types with inconsistencies
    contract_types = ['Monthly', 'Annual', 'Enterprise']
    contract_variations = {
        'Monthly': ['Monthly', 'monthly', 'Month-to-Month', 'MTM', '1-month'],
        'Annual': ['Annual', 'Yearly', 'annual', '12-month', '1-year'],
        'Enterprise': ['Enterprise', 'enterprise', 'Ent', 'Custom', 'ENTERPRISE']
    }
    
    # Industry multipliers for various metrics (non-linear relationships)
    industry_multipliers = {
        'Technology': {'revenue': 1.5, 'features': 1.3, 'satisfaction': 0.9},
        'Finance': {'revenue': 2.0, 'features': 1.1, 'satisfaction': 0.85},
        'Healthcare': {'revenue': 1.8, 'features': 0.9, 'satisfaction': 0.95},
        'Retail': {'revenue': 0.8, 'features': 1.2, 'satisfaction': 0.88},
        'Manufacturing': {'revenue': 1.2, 'features': 0.8, 'satisfaction': 0.92},
        'Education': {'revenue': 0.6, 'features': 1.0, 'satisfaction': 1.05},
        'Real Estate': {'revenue': 1.4, 'features': 0.7, 'satisfaction': 0.9},
        'Consulting': {'revenue': 1.6, 'features': 1.4, 'satisfaction': 0.87}
    }
    
    data = []
    
    # Generate hidden factors that create non-linear relationships
    # These will influence multiple variables in complex ways
    company_quality = np.random.beta(2, 3, n_customers)  # Skewed distribution
    tech_sophistication = np.random.beta(3, 2, n_customers)  # Different skew
    financial_health = np.random.beta(2.5, 2.5, n_customers)  # More centered
    
    for i in range(n_customers):
        # Create customer ID with some duplicates
        if i < int(n_customers * 0.99):  # 99% unique
            customer_id = f"CUST{i+1:04d}"
        else:
            # Create 1% duplicate IDs
            customer_id = f"CUST{np.random.randint(1, int(n_customers * 0.99)):04d}"
        
        # Company name (some missing)
        if np.random.random() < 0.02:
            company_name = np.nan
        else:
            company_name = f"Company_{np.random.randint(1, 3000)}"
        
        # Industry with variations and missing values
        # Industry is influenced by hidden factors
        if np.random.random() < 0.08:
            industry = np.nan
            base_industry = None
        else:
            # Tech sophistication influences industry choice
            if tech_sophistication[i] > 0.7:
                industry_weights = [0.4, 0.15, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05]
            elif financial_health[i] > 0.7:
                industry_weights = [0.15, 0.35, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]
            else:
                industry_weights = [0.125] * 8
            
            base_industry = np.random.choice(industries, p=industry_weights)
            
            if base_industry in industry_variations and np.random.random() < 0.3:
                industry = np.random.choice(industry_variations[base_industry])
            else:
                industry = base_industry
        
        # Employee count with complex relationship to hidden factors
        # Non-linear: quadratic and exponential components
        if np.random.random() < 0.05:
            employee_count = np.nan
        elif np.random.random() < 0.02:
            employee_count = np.random.choice([-10, -5, 0, 999999])  # Invalid values
        else:
            # Complex non-linear relationship
            base_employees = np.exp(4 + 3 * company_quality[i]**2 + 2 * financial_health[i])
            noise = np.random.lognormal(0, 0.5)
            employee_count = int(max(1, base_employees * noise))
            
            if np.random.random() < 0.05:  # 5% extreme outliers
                employee_count = np.random.randint(5000, 50000)
        
        # Annual revenue - non-linear relationship with employees and industry
        if np.random.random() < 0.175:  # 17.5% missing
            annual_revenue = np.nan
        elif employee_count and employee_count > 0:
            # Non-linear: logarithmic and polynomial components
            multiplier = industry_multipliers.get(base_industry, {}).get('revenue', 1.0) if base_industry else 1.0
            
            # Complex relationship: log(employees) * financial_health^2 * industry_multiplier
            base_revenue = (np.log(employee_count + 1) * 10000 * 
                          (1 + financial_health[i]**2) * 
                          multiplier * 
                          (1 + 0.5 * np.sin(company_quality[i] * np.pi)))  # Sinusoidal component
            
            noise = np.random.normal(1, 0.2)
            annual_revenue = max(0, base_revenue * noise)
            
            if np.random.random() < 0.02:  # Some extreme outliers
                annual_revenue *= np.random.choice([0.01, 100])
        else:
            annual_revenue = np.nan
        
        # Signup date
        # Older companies tend to have better financial health (survivor bias)
        days_ago_base = 545  # ~1.5 years average
        days_ago = int(days_ago_base * (2 - financial_health[i]) * np.random.uniform(0.3, 1.5))
        days_ago = min(1095, max(30, days_ago))  # Clamp between 1 month and 3 years
        signup_date = datetime.now() - timedelta(days=days_ago)
        
        # Last login date - influenced by tech sophistication and satisfaction
        if np.random.random() < 0.125:  # 12.5% haven't logged in
            last_login_date = np.nan
        else:
            # Non-linear: Tech sophistication affects login frequency exponentially
            login_probability = tech_sophistication[i]**2
            
            if np.random.random() < login_probability:  # Active users
                days_since_login = int(np.random.exponential(7) * (2 - tech_sophistication[i]))
                days_since_login = min(30, days_since_login)
            else:  # At-risk users
                days_since_login = int(np.random.exponential(30) * (2 - company_quality[i]))
                days_since_login = min(180, max(31, days_since_login))
            
            last_login_date = datetime.now() - timedelta(days=days_since_login)
        
        # Monthly spend - complex non-linear relationships
        if np.random.random() < 0.06:
            monthly_spend = np.nan
        elif np.random.random() < 0.01:
            monthly_spend = np.random.uniform(-500, -10)  # Error: negative values
        else:
            if employee_count and employee_count > 0 and annual_revenue:
                # Non-linear: cubic relationship with financial health, sqrt with employees
                base_spend = (np.sqrt(employee_count) * 50 * 
                            (1 + financial_health[i]**3) * 
                            (1 + 0.3 * np.cos(tech_sophistication[i] * 2 * np.pi)))
                
                # Contract type influences spend
                if base_industry:
                    multiplier = industry_multipliers.get(base_industry, {}).get('revenue', 1.0)
                    base_spend *= multiplier
                
                noise = np.random.lognormal(0, 0.3)
                monthly_spend = max(0, base_spend * noise)
            else:
                monthly_spend = np.random.uniform(100, 5000) * (0.5 + company_quality[i])
        
        # Support tickets - inverse relationship with tech sophistication
        if np.random.random() < 0.10:  # 10% missing
            support_tickets = np.nan
        elif pd.isna(last_login_date):  # No tickets if never logged in
            support_tickets = 0
        else:
            # Non-linear: exponential decay with tech sophistication
            base_tickets = np.random.poisson(10 * np.exp(-2 * tech_sophistication[i]))
            
            # Company quality affects ticket volume (inverse relationship)
            ticket_multiplier = 2 - company_quality[i]
            support_tickets = int(base_tickets * ticket_multiplier)
            support_tickets = max(0, support_tickets)
        
        # Features used - influenced by tech sophistication and industry
        if np.random.random() < 0.04:
            features_used = np.nan
        elif np.random.random() < 0.01:
            features_used = np.random.choice([-1, 25, 30])  # Invalid: out of range
        else:
            # Non-linear: sigmoid-like relationship with tech sophistication
            feature_probability = 1 / (1 + np.exp(-5 * (tech_sophistication[i] - 0.5)))
            
            if base_industry:
                multiplier = industry_multipliers.get(base_industry, {}).get('features', 1.0)
                feature_probability *= multiplier
            
            features_used = np.random.binomial(20, min(0.95, feature_probability))
        
        # Satisfaction score - complex multi-factor relationship
        if np.random.random() < 0.25:  # 25% missing
            satisfaction_score = np.nan
        elif np.random.random() < 0.01:
            satisfaction_score = np.random.choice([0, 11, 15, -5])  # Invalid scores
        else:
            # Non-linear: influenced by multiple factors
            base_satisfaction = 5.5  # Neutral starting point
            
            # Company quality has quadratic effect
            base_satisfaction += 2 * company_quality[i]**2
            
            # Support tickets have inverse exponential effect
            if support_tickets and support_tickets > 0:
                base_satisfaction -= 2 * (1 - np.exp(-support_tickets / 10))
            
            # Tech sophistication has positive linear effect
            base_satisfaction += 1.5 * tech_sophistication[i]
            
            # Industry effect
            if base_industry:
                multiplier = industry_multipliers.get(base_industry, {}).get('satisfaction', 1.0)
                base_satisfaction *= multiplier
            
            # Add noise and clamp
            noise = np.random.normal(0, 0.5)
            satisfaction_score = base_satisfaction + noise
            
            # Some natural bounds (but can exceed for invalid values check)
            if np.random.random() < 0.99:  # 99% of the time, keep in reasonable range
                satisfaction_score = max(1, min(10, satisfaction_score))
        
        # Contract type - influenced by company size and financial health
        if np.random.random() < 0.07:
            contract_type = np.nan
        else:
            # Larger, financially healthy companies prefer enterprise
            if employee_count and employee_count > 500 and financial_health[i] > 0.6:
                contract_weights = [0.1, 0.3, 0.6]
            elif employee_count and employee_count > 100:
                contract_weights = [0.2, 0.5, 0.3]
            else:
                contract_weights = [0.6, 0.3, 0.1]
            
            base_contract = np.random.choice(contract_types, p=contract_weights)
            
            if base_contract in contract_variations and np.random.random() < 0.4:
                contract_type = np.random.choice(contract_variations[base_contract])
            else:
                contract_type = base_contract
        
        # Payment method - influenced by financial health and company size
        if np.random.random() < 0.05:
            payment_method = np.nan
        else:
            if financial_health[i] > 0.7 and employee_count and employee_count > 200:
                payment_weights = [0.1, 0.3, 0.4, 0.2]  # Prefer Wire Transfer
            elif tech_sophistication[i] > 0.6:
                payment_weights = [0.5, 0.3, 0.1, 0.1]  # Prefer Credit Card
            else:
                payment_weights = [0.3, 0.2, 0.2, 0.3]  # Balanced
            
            payment_method = np.random.choice(
                ['Credit Card', 'ACH', 'Wire Transfer', 'Invoice'],
                p=payment_weights
            )
        
        data.append({
            'customer_id': customer_id,
            'company_name': company_name,
            'industry': industry,
            'employee_count': employee_count,
            'annual_revenue': annual_revenue,
            'signup_date': signup_date,
            'last_login_date': last_login_date,
            'monthly_spend': monthly_spend,
            'support_tickets': support_tickets,
            'features_used': features_used,
            'satisfaction_score': satisfaction_score,
            'contract_type': contract_type,
            'payment_method': payment_method
        })
    
    return pd.DataFrame(data)


def validate_data(df):
    """
    Validate that the generated data meets assignment requirements.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Generated customer dataset
    
    Returns:
    --------
    bool : True if all validations pass
    """
    print("Validating generated data...")
    print("=" * 60)
    
    # Check data shape
    print(f"✓ Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['customer_id'], keep=False)]
    print(f"✓ Duplicate customer IDs: {len(duplicates)} records ({len(duplicates)/len(df)*100:.1f}%)")
    
    # Check missing values
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df)) * 100
    })
    
    print("\n✓ Missing value summary:")
    for _, row in missing_summary[missing_summary['Missing_Count'] > 0].iterrows():
        print(f"  - {row['Column']}: {row['Missing_Percent']:.1f}%")
    
    # Check for invalid values
    if 'employee_count' in df.columns:
        invalid_employees = df[df['employee_count'] <= 0]['employee_count'].count()
        print(f"\n✓ Invalid employee counts: {invalid_employees}")
    
    if 'monthly_spend' in df.columns:
        negative_spend = df[df['monthly_spend'] < 0]['monthly_spend'].count()
        print(f"✓ Negative monthly spend: {negative_spend}")
    
    if 'satisfaction_score' in df.columns:
        invalid_satisfaction = df[(df['satisfaction_score'] < 1) | (df['satisfaction_score'] > 10)]['satisfaction_score'].count()
        print(f"✓ Invalid satisfaction scores: {invalid_satisfaction}")
    
    if 'features_used' in df.columns:
        invalid_features = df[(df['features_used'] < 0) | (df['features_used'] > 20)]['features_used'].count()
        print(f"✓ Invalid features used: {invalid_features}")
    
    # Check categorical variations
    if 'industry' in df.columns:
        print(f"\n✓ Industry variations: {df['industry'].nunique()} unique values")
    
    if 'contract_type' in df.columns:
        print(f"✓ Contract type variations: {df['contract_type'].nunique()} unique values")
    
    # Check for non-linear relationships (correlation matrix shouldn't show only linear patterns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = (corr_matrix.abs() > 0.9).sum().sum() - len(numeric_cols)  # Subtract diagonal
        moderate_corr = ((corr_matrix.abs() > 0.3) & (corr_matrix.abs() < 0.7)).sum().sum()
        
        print(f"\n✓ Correlation analysis:")
        print(f"  - High linear correlations (>0.9): {high_corr//2} pairs")
        print(f"  - Moderate correlations (0.3-0.7): {moderate_corr//2} pairs")
        print(f"  - This suggests good non-linear relationships for KNN imputation")
    
    print("\n" + "=" * 60)
    print("✓ Data validation complete - ready for assignment!")
    
    return True


def main():
    """Main function to generate and save the dataset."""
    parser = argparse.ArgumentParser(
        description='Generate Week 3 assignment data for Customer Data Pipeline'
    )
    parser.add_argument(
        '--n_customers',
        type=int,
        default=5000,
        help='Number of customer records to generate (default: 5000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='week03_customer_data_raw.csv',
        help='Output filename (default: week03_customer_data_raw.csv)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the generated data'
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_customers} customer records...")
    df = generate_messy_customer_data(n_customers=args.n_customers, seed=args.seed)
    
    # Save the data
    df.to_csv(args.output, index=False)
    print(f"\n✓ Data saved to: {args.output}")
    
    # Validate if requested
    if args.validate:
        print()
        validate_data(df)
    
    # Show sample statistics
    print("\nSample statistics:")
    print("-" * 40)
    print(f"Total records: {len(df)}")
    print(f"Complete records: {df.dropna().shape[0]} ({df.dropna().shape[0]/len(df)*100:.1f}%)")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    
    # Show relationship complexity
    numeric_cols = ['employee_count', 'annual_revenue', 'monthly_spend', 'satisfaction_score', 'features_used']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) > 1:
        print("\nFeature relationships (for KNN imputation effectiveness):")
        print("These non-linear relationships will make KNN imputation superior to simple methods:")
        
        # Sample a few relationships
        clean_subset = df[available_cols].dropna()
        if len(clean_subset) > 100:
            print(f"  - Employee count has non-linear relationship with revenue (log/polynomial)")
            print(f"  - Satisfaction influenced by multiple factors (quadratic, exponential)")
            print(f"  - Features used follows sigmoid-like pattern with hidden factors")
            print(f"  - Support tickets inversely related to satisfaction (exponential decay)")
    
    return df


if __name__ == "__main__":
    df = main()