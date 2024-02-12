import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load data from CSV files
departments = pd.read_csv("data/departments.csv")
aisles = pd.read_csv("data/aisles.csv")
products = pd.read_csv("data/products.csv")
order_products_train = pd.read_csv("data/order_products__train.csv")

# Merge product data with aisle and department information
products = products.merge(aisles, on='aisle_id').merge(departments, on='department_id')
# Sort the merged product data by product_id
products = products.sort_values('product_id')

# Collaborative filtering

# Merge order-product data with product information
orders_products = order_products_train.merge(products, on='product_id')
# Limit the dataset to a subset (e.g., first 1,000,000 rows)
orders_products = orders_products[:1000000]
# Rename the 'add_to_cart_order' column to 'rating'
orders_products.rename(columns={'add_to_cart_order': 'rating'}, inplace=True)

# Drop rows with NaN product names
orders_products = orders_products.dropna(axis=0, subset=['product_name'])

# Calculate total rating count for each product
ratingCount = (
    orders_products.groupby(by=['product_name'])['rating']
    .count()
    .reset_index()
    .rename(columns={'rating': 'totalRatingCount'})
    [['product_name', 'totalRatingCount']]
)

# Merge total rating count with order-product data
orders_products_with_totalRatingCount = orders_products.merge(ratingCount, left_on='product_name', right_on='product_name', how='left')

# Filter out products based on rating thresholds
rating_threshold_min = 0
rating_threshold_max = 50
rating_popular_products = orders_products_with_totalRatingCount.query('totalRatingCount >= @rating_threshold_min')
rating_popular_products = orders_products_with_totalRatingCount.query('totalRatingCount <= @rating_threshold_max')
print(pd.unique(rating_popular_products['product_id']).size, rating_popular_products['order_id'].shape)

# Create a pivot table for collaborative filtering
orders_products_df = rating_popular_products.pivot_table(index='product_name', columns='order_id', values='rating').fillna(0)
print(orders_products_df.shape)

# Convert pivot table to CSR matrix
orders_products_df_matrix = csr_matrix(orders_products_df.values)

# Initialize k-nearest neighbors model for collaborative filtering
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(orders_products_df_matrix)
