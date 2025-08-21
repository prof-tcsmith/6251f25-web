"""
Generate synthetic housing market data for Week 4 Assignment
This script creates a realistic real estate dataset for linear regression analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_housing_data(n_properties=1500):
    """Generate synthetic housing market data with realistic relationships"""
    
    properties = []
    
    # Define neighborhoods with characteristics
    neighborhoods = {
        'Downtown': {'price_mult': 1.5, 'size_avg': 1200, 'crime_rate': 'medium'},
        'Suburbs': {'price_mult': 1.0, 'size_avg': 2000, 'crime_rate': 'low'},
        'Waterfront': {'price_mult': 2.0, 'size_avg': 1800, 'crime_rate': 'low'},
        'University': {'price_mult': 0.9, 'size_avg': 1100, 'crime_rate': 'medium'},
        'Industrial': {'price_mult': 0.7, 'size_avg': 1500, 'crime_rate': 'high'},
        'Historic': {'price_mult': 1.3, 'size_avg': 1600, 'crime_rate': 'low'}
    }
    
    schools_quality = ['excellent', 'good', 'average', 'below_average']
    property_types = ['single_family', 'condo', 'townhouse', 'multi_family']
    
    for i in range(n_properties):
        # Choose neighborhood
        neighborhood = np.random.choice(list(neighborhoods.keys()), 
                                       p=[0.15, 0.30, 0.10, 0.20, 0.10, 0.15])
        hood_info = neighborhoods[neighborhood]
        
        # Generate features with realistic relationships
        
        # Size (square feet) - varies by neighborhood
        size = np.random.normal(hood_info['size_avg'], 400)
        size = max(500, min(5000, size))  # Bound between 500-5000
        
        # Bedrooms - correlated with size
        if size < 1000:
            bedrooms = np.random.choice([1, 2], p=[0.7, 0.3])
        elif size < 1500:
            bedrooms = np.random.choice([2, 3], p=[0.6, 0.4])
        elif size < 2500:
            bedrooms = np.random.choice([3, 4], p=[0.5, 0.5])
        else:
            bedrooms = np.random.choice([4, 5, 6], p=[0.5, 0.3, 0.2])
        
        # Bathrooms - correlated with bedrooms
        bathrooms = bedrooms - np.random.choice([0, 0.5, 1], p=[0.5, 0.3, 0.2])
        bathrooms = max(1, bathrooms)
        
        # Age of property
        age = np.random.exponential(15)  # Most properties are newer
        age = min(100, int(age))
        
        # Garage spaces
        if neighborhood == 'Downtown':
            garage = np.random.choice([0, 1], p=[0.6, 0.4])
        elif neighborhood == 'Suburbs':
            garage = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
        else:
            garage = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        
        # Property type
        if neighborhood == 'Downtown':
            prop_type = np.random.choice(property_types, p=[0.1, 0.6, 0.2, 0.1])
        elif neighborhood == 'Suburbs':
            prop_type = np.random.choice(property_types, p=[0.7, 0.1, 0.2, 0.0])
        else:
            prop_type = np.random.choice(property_types, p=[0.4, 0.3, 0.2, 0.1])
        
        # School quality
        if neighborhood in ['Suburbs', 'Waterfront']:
            school = np.random.choice(schools_quality, p=[0.4, 0.4, 0.15, 0.05])
        elif neighborhood == 'Industrial':
            school = np.random.choice(schools_quality, p=[0.05, 0.15, 0.4, 0.4])
        else:
            school = np.random.choice(schools_quality, p=[0.2, 0.3, 0.35, 0.15])
        
        # Distance to city center (miles)
        if neighborhood == 'Downtown':
            dist_city = np.random.uniform(0, 3)
        elif neighborhood == 'Suburbs':
            dist_city = np.random.uniform(8, 20)
        elif neighborhood == 'Industrial':
            dist_city = np.random.uniform(5, 12)
        else:
            dist_city = np.random.uniform(3, 10)
        
        # Crime rate (per 1000 residents)
        crime_base = {'low': 10, 'medium': 25, 'high': 40}
        crime_rate = np.random.normal(crime_base[hood_info['crime_rate']], 5)
        crime_rate = max(0, crime_rate)
        
        # Has pool
        has_pool = 1 if (neighborhood in ['Waterfront', 'Suburbs'] and 
                         np.random.random() < 0.3) else 0
        
        # Has renovation
        needs_renovation = 1 if age > 30 and np.random.random() < 0.4 else 0
        has_renovation = 1 if age > 20 and np.random.random() < 0.3 else 0
        
        # Walk score (0-100)
        if neighborhood == 'Downtown':
            walk_score = np.random.uniform(70, 95)
        elif neighborhood == 'Suburbs':
            walk_score = np.random.uniform(20, 50)
        else:
            walk_score = np.random.uniform(40, 70)
        
        # Property tax (annual)
        tax_rate = 0.012  # 1.2% of home value
        
        # Calculate price using a realistic formula
        # Base price
        base_price = 50000
        
        # Price factors
        size_factor = size * 150  # $150 per sq ft base
        bedroom_factor = bedrooms * 10000
        bathroom_factor = bathrooms * 8000
        garage_factor = garage * 15000
        pool_factor = has_pool * 30000
        
        # Age depreciation (but historic homes appreciate)
        if neighborhood == 'Historic' and age > 50:
            age_factor = age * 500  # Appreciation
        else:
            age_factor = -age * 1000  # Depreciation
        
        # School quality factor
        school_mult = {'excellent': 1.2, 'good': 1.1, 'average': 1.0, 'below_average': 0.9}
        
        # Distance factor (closer to city = more expensive, except industrial)
        if neighborhood != 'Industrial':
            dist_factor = -dist_city * 2000
        else:
            dist_factor = 0
        
        # Crime factor
        crime_factor = -crime_rate * 500
        
        # Renovation factors
        renovation_factor = has_renovation * 25000 - needs_renovation * 15000
        
        # Walk score factor
        walk_factor = walk_score * 200
        
        # Calculate final price
        price = (base_price + size_factor + bedroom_factor + bathroom_factor + 
                garage_factor + pool_factor + age_factor + dist_factor + 
                crime_factor + renovation_factor + walk_factor)
        
        # Apply neighborhood multiplier
        price = price * hood_info['price_mult'] * school_mult[school]
        
        # Add some random noise
        price = price * np.random.uniform(0.9, 1.1)
        
        # Ensure positive price
        price = max(50000, price)
        
        # Property tax
        property_tax = price * tax_rate
        
        # HOA fees (mainly for condos and townhouses)
        if prop_type == 'condo':
            hoa_fee = np.random.uniform(200, 500)
        elif prop_type == 'townhouse':
            hoa_fee = np.random.uniform(100, 300)
        else:
            hoa_fee = 0
        
        # Days on market (influenced by price and condition)
        if needs_renovation:
            days_on_market = np.random.poisson(45)
        else:
            days_on_market = np.random.poisson(21)
        
        # Listing date
        listing_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        
        properties.append({
            'property_id': f'PROP{str(i+1).zfill(5)}',
            'address': f'{np.random.randint(1, 9999)} {neighborhood} St',
            'neighborhood': neighborhood,
            'property_type': prop_type,
            'size_sqft': int(size),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age_years': age,
            'garage_spaces': garage,
            'has_pool': has_pool,
            'has_renovation': has_renovation,
            'needs_renovation': needs_renovation,
            'school_quality': school,
            'distance_city_center': round(dist_city, 1),
            'crime_rate': round(crime_rate, 1),
            'walk_score': int(walk_score),
            'property_tax_annual': round(property_tax, 2),
            'hoa_fee_monthly': round(hoa_fee, 2),
            'days_on_market': days_on_market,
            'listing_date': listing_date,
            'price': round(price, 2)
        })
    
    return pd.DataFrame(properties)

def generate_economic_indicators():
    """Generate economic indicators that might affect housing prices"""
    
    months = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    
    economic_data = []
    base_mortgage_rate = 6.5
    base_unemployment = 3.5
    
    for month in months:
        # Mortgage rate with trend and seasonality
        trend = (month.month - months[0].month) * 0.02
        seasonal = np.sin(2 * np.pi * month.month / 12) * 0.3
        mortgage_rate = base_mortgage_rate + trend + seasonal + np.random.normal(0, 0.1)
        
        # Unemployment rate
        unemployment = base_unemployment + np.random.normal(0, 0.2)
        unemployment = max(2.5, min(5.0, unemployment))
        
        # Housing inventory (months of supply)
        inventory = 3 + np.random.normal(0, 0.5)
        inventory = max(1, min(6, inventory))
        
        economic_data.append({
            'month': month,
            'mortgage_rate_30yr': round(mortgage_rate, 2),
            'unemployment_rate': round(unemployment, 1),
            'housing_inventory_months': round(inventory, 1),
            'median_household_income': 75000 + np.random.normal(0, 2000)
        })
    
    return pd.DataFrame(economic_data)

if __name__ == "__main__":
    # Generate the data
    properties_df = generate_housing_data()
    economic_df = generate_economic_indicators()
    
    # Add some missing values and outliers for realism
    # Missing values
    missing_indices = np.random.choice(properties_df.index, size=50, replace=False)
    properties_df.loc[missing_indices[:20], 'walk_score'] = np.nan
    properties_df.loc[missing_indices[20:35], 'crime_rate'] = np.nan
    properties_df.loc[missing_indices[35:50], 'hoa_fee_monthly'] = np.nan
    
    # Add a few outliers
    outlier_indices = np.random.choice(properties_df.index, size=10, replace=False)
    properties_df.loc[outlier_indices[:5], 'price'] *= 3  # Luxury properties
    properties_df.loc[outlier_indices[5:], 'size_sqft'] = np.random.randint(8000, 12000, 5)  # Mansions
    
    # Save to CSV files
    properties_df.to_csv('housing_properties.csv', index=False)
    economic_df.to_csv('economic_indicators.csv', index=False)
    
    print("Housing market data generation complete!")
    print(f"Properties: {len(properties_df)} records")
    print(f"Economic indicators: {len(economic_df)} monthly records")
    print(f"\nTarget variable (price) statistics:")
    print(f"  Mean: ${properties_df['price'].mean():,.2f}")
    print(f"  Median: ${properties_df['price'].median():,.2f}")
    print(f"  Min: ${properties_df['price'].min():,.2f}")
    print(f"  Max: ${properties_df['price'].max():,.2f}")