import pandas as pd
from application.remommendationMethods.products import prodList

def ordersProducts():
    """
    Generate Orders-Products DataFrame.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing order-product information.

    Notes:
    ------
    This function reads the 'order_products__train.csv' file and combines it
    with product information obtained from the 'prodList' function to create
    a DataFrame containing order-product details.

    The function calculates the total rating count for each product and filters
    out products based on rating thresholds.

    Finally, it generates a pivot table with products as rows, order IDs as
    columns, and ratings as values, filling NaN values with 0.

    The resulting DataFrame is returned.
    """
    # Read order-product data and product list
    order_products_train = pd.read_csv("data/order_products__train.csv")
    products = prodList()

    # Create DataFrame for product information
    products = pd.DataFrame(products, columns=['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department'])

    # Merge order-product data with product information
    orders_products = order_products_train.merge(products, on='product_id')
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

    # Print information about the filtered products
    print(pd.unique(rating_popular_products['product_id']).size, rating_popular_products['order_id'].shape)

    # Generate pivot table for orders-products DataFrame
    orders_products_df = rating_popular_products.pivot_table(index='product_name', columns='order_id', values='rating').fillna(0)

    return orders_products_df
