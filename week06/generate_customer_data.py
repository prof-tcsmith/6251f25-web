"""
Generate synthetic customer behavior data for Week 6 Assignment
This script creates a realistic customer dataset for KNN analysis and recommendation systems
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
random.seed(42)

def generate_customer_behavior_data(n_customers=1500):
    """Generate synthetic customer behavior data with natural clusters"""
    
    customers = []
    
    # Define customer segments with distinct behaviors
    segments = {
        'Premium_Shoppers': {
            'size': 0.15,
            'age_range': (35, 65),
            'income_range': (80000, 200000),
            'avg_purchase': (150, 500),
            'frequency': (8, 20),  # purchases per month
            'categories': ['Electronics', 'Fashion', 'Home', 'Beauty'],
            'online_pref': 0.7,
            'loyalty': 0.9
        },
        'Budget_Conscious': {
            'size': 0.25,
            'age_range': (25, 45),
            'income_range': (30000, 60000),
            'avg_purchase': (20, 80),
            'frequency': (3, 8),
            'categories': ['Groceries', 'Home', 'Kids'],
            'online_pref': 0.4,
            'loyalty': 0.6
        },
        'Young_Trendsetters': {
            'size': 0.20,
            'age_range': (18, 30),
            'income_range': (20000, 50000),
            'avg_purchase': (30, 120),
            'frequency': (5, 15),
            'categories': ['Fashion', 'Electronics', 'Beauty', 'Sports'],
            'online_pref': 0.9,
            'loyalty': 0.4
        },
        'Family_Focused': {
            'size': 0.25,
            'age_range': (30, 50),
            'income_range': (50000, 100000),
            'avg_purchase': (80, 200),
            'frequency': (6, 12),
            'categories': ['Groceries', 'Kids', 'Home', 'Sports'],
            'online_pref': 0.5,
            'loyalty': 0.8
        },
        'Occasional_Buyers': {
            'size': 0.15,
            'age_range': (25, 60),
            'income_range': (40000, 80000),
            'avg_purchase': (50, 150),
            'frequency': (1, 4),
            'categories': ['Electronics', 'Home', 'Fashion'],
            'online_pref': 0.6,
            'loyalty': 0.3
        }
    }
    
    # Product categories
    all_categories = ['Electronics', 'Fashion', 'Groceries', 'Home', 'Beauty', 'Sports', 'Kids', 'Books']
    
    customer_id = 1
    for segment_name, segment_info in segments.items():
        n_segment = int(n_customers * segment_info['size'])
        
        for _ in range(n_segment):
            # Demographics
            age = np.random.randint(segment_info['age_range'][0], segment_info['age_range'][1])
            income = np.random.uniform(segment_info['income_range'][0], segment_info['income_range'][1])
            
            # Shopping behavior
            avg_purchase_amount = np.random.uniform(segment_info['avg_purchase'][0], segment_info['avg_purchase'][1])
            purchase_frequency = np.random.uniform(segment_info['frequency'][0], segment_info['frequency'][1])
            
            # Add some noise to make it more realistic
            avg_purchase_amount *= np.random.uniform(0.8, 1.2)
            purchase_frequency *= np.random.uniform(0.8, 1.2)
            
            # Online vs offline preference
            online_purchase_ratio = segment_info['online_pref'] + np.random.uniform(-0.2, 0.2)
            online_purchase_ratio = np.clip(online_purchase_ratio, 0, 1)
            
            # Category preferences
            category_scores = {}
            for cat in all_categories:
                if cat in segment_info['categories']:
                    # High affinity for preferred categories
                    category_scores[cat] = np.random.uniform(0.6, 1.0)
                else:
                    # Low affinity for other categories
                    category_scores[cat] = np.random.uniform(0.0, 0.4)
            
            # Normalize category scores to sum to 1
            total_score = sum(category_scores.values())
            for cat in category_scores:
                category_scores[cat] = category_scores[cat] / total_score
            
            # Customer lifetime value components
            months_as_customer = np.random.randint(1, 60)
            total_purchases = int(purchase_frequency * months_as_customer)
            total_spent = total_purchases * avg_purchase_amount * np.random.uniform(0.9, 1.1)
            
            # Engagement metrics
            email_open_rate = segment_info['loyalty'] * np.random.uniform(0.5, 1.0)
            email_click_rate = email_open_rate * np.random.uniform(0.1, 0.4)
            
            # Customer satisfaction
            satisfaction_score = np.random.beta(
                segment_info['loyalty'] * 10, 
                (1 - segment_info['loyalty']) * 10
            ) * 5  # Scale to 0-5
            
            # Days since last purchase (inversely related to frequency)
            if purchase_frequency > 10:
                days_since_last = np.random.randint(1, 15)
            elif purchase_frequency > 5:
                days_since_last = np.random.randint(5, 30)
            else:
                days_since_last = np.random.randint(15, 90)
            
            # Return/complaint rate (inversely related to satisfaction)
            return_rate = (5 - satisfaction_score) / 5 * 0.2 * np.random.uniform(0.5, 1.5)
            
            # Device usage
            mobile_usage = online_purchase_ratio * np.random.uniform(0.3, 0.7)
            desktop_usage = online_purchase_ratio * (1 - mobile_usage)
            
            # Time-based patterns
            preferred_shopping_hour = np.random.choice([10, 14, 19, 21], 
                                                      p=[0.2, 0.3, 0.35, 0.15])
            weekend_ratio = 0.4 + np.random.uniform(-0.2, 0.2)
            
            # Seasonal patterns
            holiday_spending_increase = 1 + segment_info['loyalty'] + np.random.uniform(0, 0.5)
            
            # Social influence
            referred_customers = np.random.poisson(segment_info['loyalty'] * 2)
            social_media_influence = np.random.uniform(0, 1) * (1 if age < 40 else 0.5)
            
            # Price sensitivity (inverse of income)
            price_sensitivity = 1 - (income - 20000) / 180000
            price_sensitivity = np.clip(price_sensitivity, 0.1, 0.9)
            
            # Promotion responsiveness
            promo_response_rate = price_sensitivity * np.random.uniform(0.5, 1.0)
            
            customers.append({
                'customer_id': f'CUST{str(customer_id).zfill(5)}',
                'age': age,
                'income': income,
                'months_as_customer': months_as_customer,
                'total_purchases': total_purchases,
                'total_spent': round(total_spent, 2),
                'avg_purchase_amount': round(avg_purchase_amount, 2),
                'purchase_frequency_monthly': round(purchase_frequency, 2),
                'online_purchase_ratio': round(online_purchase_ratio, 3),
                'mobile_usage_ratio': round(mobile_usage, 3),
                'desktop_usage_ratio': round(desktop_usage, 3),
                'days_since_last_purchase': days_since_last,
                'email_open_rate': round(email_open_rate, 3),
                'email_click_rate': round(email_click_rate, 3),
                'satisfaction_score': round(satisfaction_score, 2),
                'return_rate': round(return_rate, 3),
                'referred_customers': referred_customers,
                'social_media_influence': round(social_media_influence, 3),
                'price_sensitivity': round(price_sensitivity, 3),
                'promo_response_rate': round(promo_response_rate, 3),
                'preferred_shopping_hour': preferred_shopping_hour,
                'weekend_shopping_ratio': round(weekend_ratio, 3),
                'holiday_spending_multiplier': round(holiday_spending_increase, 2),
                # Category preferences
                'cat_electronics': round(category_scores.get('Electronics', 0), 3),
                'cat_fashion': round(category_scores.get('Fashion', 0), 3),
                'cat_groceries': round(category_scores.get('Groceries', 0), 3),
                'cat_home': round(category_scores.get('Home', 0), 3),
                'cat_beauty': round(category_scores.get('Beauty', 0), 3),
                'cat_sports': round(category_scores.get('Sports', 0), 3),
                'cat_kids': round(category_scores.get('Kids', 0), 3),
                'cat_books': round(category_scores.get('Books', 0), 3),
                'true_segment': segment_name  # Hidden true segment for validation
            })
            
            customer_id += 1
    
    return pd.DataFrame(customers)

def generate_product_catalog(n_products=500):
    """Generate product catalog with features for recommendation"""
    
    products = []
    
    categories = {
        'Electronics': {'price_range': (50, 2000), 'margin': 0.15, 'return_rate': 0.05},
        'Fashion': {'price_range': (20, 500), 'margin': 0.40, 'return_rate': 0.15},
        'Groceries': {'price_range': (2, 50), 'margin': 0.20, 'return_rate': 0.02},
        'Home': {'price_range': (10, 1000), 'margin': 0.35, 'return_rate': 0.08},
        'Beauty': {'price_range': (5, 200), 'margin': 0.50, 'return_rate': 0.10},
        'Sports': {'price_range': (15, 500), 'margin': 0.30, 'return_rate': 0.07},
        'Kids': {'price_range': (10, 150), 'margin': 0.35, 'return_rate': 0.12},
        'Books': {'price_range': (5, 50), 'margin': 0.40, 'return_rate': 0.03}
    }
    
    for i in range(n_products):
        category = np.random.choice(list(categories.keys()))
        cat_info = categories[category]
        
        # Product attributes
        price = np.random.uniform(cat_info['price_range'][0], cat_info['price_range'][1])
        
        # Quality score affects customer satisfaction
        quality_score = np.random.beta(3, 2) * 5  # Skewed towards higher quality
        
        # Popularity (some products are bestsellers)
        if np.random.random() < 0.1:  # 10% are popular
            popularity_score = np.random.uniform(0.7, 1.0)
        else:
            popularity_score = np.random.uniform(0.1, 0.7)
        
        # Inventory and availability
        stock_quantity = np.random.randint(0, 1000)
        is_available = 1 if stock_quantity > 0 else 0
        
        # Ratings and reviews
        num_reviews = int(np.random.exponential(20))
        if num_reviews > 0:
            avg_rating = np.random.beta(quality_score, 5 - quality_score) * 5
            avg_rating = max(1, min(5, avg_rating))
        else:
            avg_rating = 0
        
        # Promotional status
        is_on_sale = 1 if np.random.random() < 0.2 else 0
        discount_percentage = np.random.uniform(0.1, 0.5) if is_on_sale else 0
        
        # Seasonal relevance
        seasonal_item = 1 if np.random.random() < 0.3 else 0
        
        products.append({
            'product_id': f'PROD{str(i+1).zfill(5)}',
            'product_name': f'{category}_Item_{i+1}',
            'category': category,
            'price': round(price, 2),
            'quality_score': round(quality_score, 2),
            'popularity_score': round(popularity_score, 3),
            'stock_quantity': stock_quantity,
            'is_available': is_available,
            'num_reviews': num_reviews,
            'avg_rating': round(avg_rating, 2),
            'is_on_sale': is_on_sale,
            'discount_percentage': round(discount_percentage, 2),
            'margin_rate': cat_info['margin'],
            'return_rate': cat_info['return_rate'],
            'seasonal_item': seasonal_item
        })
    
    return pd.DataFrame(products)

def generate_purchase_history(customers_df, products_df, n_transactions=10000):
    """Generate purchase history for collaborative filtering"""
    
    transactions = []
    
    for _ in range(n_transactions):
        # Select customer weighted by their purchase frequency
        weights = customers_df['purchase_frequency_monthly'].values
        weights = weights / weights.sum()
        customer_idx = np.random.choice(len(customers_df), p=weights)
        customer = customers_df.iloc[customer_idx]
        
        # Select product based on customer's category preferences
        category_prefs = {
            'Electronics': customer['cat_electronics'],
            'Fashion': customer['cat_fashion'],
            'Groceries': customer['cat_groceries'],
            'Home': customer['cat_home'],
            'Beauty': customer['cat_beauty'],
            'Sports': customer['cat_sports'],
            'Kids': customer['cat_kids'],
            'Books': customer['cat_books']
        }
        
        # Choose category based on preferences
        categories = list(category_prefs.keys())
        probs = list(category_prefs.values())
        probs = [p / sum(probs) for p in probs]  # Normalize
        chosen_category = np.random.choice(categories, p=probs)
        
        # Select product from chosen category
        category_products = products_df[products_df['category'] == chosen_category]
        if len(category_products) > 0:
            # Weight by popularity
            if len(category_products) > 1:
                product_weights = category_products['popularity_score'].values
                product_weights = product_weights / product_weights.sum()
                product_idx = np.random.choice(len(category_products), p=product_weights)
                product = category_products.iloc[product_idx]
            else:
                product = category_products.iloc[0]
            
            # Generate transaction details
            quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            
            # Apply discount if on sale and customer is price sensitive
            if product['is_on_sale'] and customer['price_sensitivity'] > 0.5:
                purchase_amount = product['price'] * (1 - product['discount_percentage']) * quantity
            else:
                purchase_amount = product['price'] * quantity
            
            # Generate rating (influenced by quality and satisfaction)
            if np.random.random() < 0.3:  # 30% leave ratings
                rating = np.random.normal(
                    product['quality_score'] * customer['satisfaction_score'] / 5 * 5,
                    0.5
                )
                rating = max(1, min(5, round(rating)))
            else:
                rating = None
            
            # Purchase date
            days_ago = np.random.randint(0, 365)
            purchase_date = datetime.now() - timedelta(days=days_ago)
            
            transactions.append({
                'transaction_id': f'TRX{str(len(transactions)+1).zfill(6)}',
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'category': chosen_category,
                'purchase_date': purchase_date,
                'quantity': quantity,
                'purchase_amount': round(purchase_amount, 2),
                'rating': rating
            })
    
    return pd.DataFrame(transactions)

if __name__ == "__main__":
    # Generate the data
    print("Generating customer behavior data...")
    customers_df = generate_customer_behavior_data(1500)
    
    print("Generating product catalog...")
    products_df = generate_product_catalog(500)
    
    print("Generating purchase history...")
    transactions_df = generate_purchase_history(customers_df, products_df, 10000)
    
    # Add some missing values for realism
    missing_indices = np.random.choice(customers_df.index, size=100, replace=False)
    customers_df.loc[missing_indices[:30], 'social_media_influence'] = np.nan
    customers_df.loc[missing_indices[30:60], 'referred_customers'] = np.nan
    customers_df.loc[missing_indices[60:], 'email_click_rate'] = np.nan
    
    # Remove the true segment before saving (students should discover this)
    customers_save = customers_df.drop('true_segment', axis=1)
    
    # Save to CSV files
    customers_save.to_csv('customer_behavior.csv', index=False)
    products_df.to_csv('product_catalog.csv', index=False)
    transactions_df.to_csv('purchase_history.csv', index=False)
    
    # Save true segments separately for validation
    true_segments = customers_df[['customer_id', 'true_segment']]
    true_segments.to_csv('customer_segments_truth.csv', index=False)
    
    print("\nCustomer segmentation data generation complete!")
    print(f"Customers: {len(customers_df)} records")
    print(f"Products: {len(products_df)} records")
    print(f"Transactions: {len(transactions_df)} records")
    
    print(f"\nTrue segment distribution:")
    print(customers_df['true_segment'].value_counts())
    
    print(f"\nCategory distribution in transactions:")
    print(transactions_df['category'].value_counts())