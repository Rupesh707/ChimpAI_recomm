import pandas as pd
import numpy as np

def prodList():
    """
    Generate Product List.

    Returns:
    --------
    list
        A list of lists containing product information.

    Notes:
    ------
    This function reads the 'products.csv', 'aisles.csv', and 'departments.csv'
    files and merges them to create a DataFrame containing product details
    along with aisle and department information.

    The resulting DataFrame is sorted by product_id and limited to the first
    1000 rows.

    Finally, the DataFrame is converted to a list of lists format and returned.
    """
    # Read product, aisle, and department data
    products = pd.read_csv("data/products.csv")
    aisles = pd.read_csv("data/aisles.csv")
    departments = pd.read_csv("data/departments.csv")

    # Merge the DataFrames
    merged_products = products.merge(aisles, on='aisle_id').merge(departments, on='department_id')

    # Sort and limit to 1000 rows
    merged_products = merged_products.sort_values('product_id').head(1000)

    # Convert to list of lists
    products_list = merged_products.values.tolist()

    return products_list

# Test the function
prodList()
