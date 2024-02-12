import pandas as pd
from application.remommendationMethods.products import prodList
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

def cbf(product_name):
    """
    Content-Based Filtering Recommender System.

    Parameters:
    -----------
    product_name : str
        The name of the product for which recommendations are requested.

    Returns:
    --------
    list
        A list containing details of recommended products based on content-based filtering.

    Notes:
    ------
    This function implements a content-based filtering recommender system
    to suggest products similar to the one specified by the product_name.

    It utilizes the product descriptions (concatenation of 'aisle' and 'department')
    to compute pairwise similarity scores using the sigmoid kernel.

    The function returns a list of recommended products along with
    their details such as product_id, product_name, aisle_id, department_id,
    aisle, and department.

    Example Usage:
    --------------
    recommended_products = cbf("Product Name")

    The 'recommended_products' list contains details of recommended products.
    """
    # Load the product list into a DataFrame
    products = pd.DataFrame(prodList(), columns=['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department'])

    # Initialize the TfidfVectorizer
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern='\w{1,}',
                          ngram_range=(1,3),
                          stop_words='english')

    # Create a new column 'description' by concatenating 'aisle' and 'department'
    products['description'] = products.apply(lambda x: x['aisle'] + ' ' + x['department'], axis=1)

    # Fill any NaNs with empty string
    products['description'] = products['description'].fillna('')
    
    # Fit and transform the 'description' column
    tfv_matrix = tfv.fit_transform(products['description'])
    
    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

    # Create a Series with the product names as the index
    indices = pd.Series(products.index, index=products['product_name']).drop_duplicates()

    # Get the index corresponding to the original product_name
    idx = indices[product_name]
    
    # Get the pairwise similarity scores
    sig_scores = list(enumerate(sig[idx]))
    
    # Sort the products based on the similarity scores
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 12 most similar products (excluding itself)
    sig_scores = sig_scores[1:13]
    
    # Get the product indices
    product_indices = [i[0] for i in sig_scores]

    # Select recommended products based on indices
    recommended_products = products.iloc[product_indices][['product_id', 'product_name']]

    # Merge with the original products DataFrame to add additional details
    recommended_products_details = recommended_products.merge(
        products[['product_id', 'aisle_id', 'department_id', 'aisle', 'department']],
        on='product_id',
        how='left'
    )

    # Convert to the desired list format if necessary
    # Example: Converting to a list of lists for each product recommendation
    recommended_products_list = recommended_products_details.values.tolist()

    return recommended_products_list
