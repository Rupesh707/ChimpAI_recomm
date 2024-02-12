import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from application.remommendationMethods.orders_products_df import ordersProducts
from application.remommendationMethods.products import prodList


def cf(query_index):
    """
    Collaborative Filtering Recommender System.

    Parameters:
    -----------
    query_index : int
        The index of the product for which recommendations are requested.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing recommended products based on collaborative filtering.

    Notes:
    ------
    This function implements a collaborative filtering recommender system
    to suggest products similar to the one specified by the query_index.

    It retrieves product information from the 'orders_products_df' DataFrame
    and 'prodList' function, constructs a sparse matrix, and computes the
    cosine similarity between products using k-nearest neighbors algorithm.

    The function returns a DataFrame of recommended products along with
    their details such as product_id, product_name, aisle_id, department_id,
    aisle, and department.
    """
    # Load product list and order-product data
    products = prodList()
    products = pd.DataFrame(products, columns=['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department'])
    orders_products_df = ordersProducts()

    # Create sparse matrix for nearest neighbors computation
    orders_products_df_matrix = csr_matrix(orders_products_df.values)
    
    # Initialize nearest neighbors model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(orders_products_df_matrix)

    # Compute distances and indices of nearest neighbors
    distances, indices = model_knn.kneighbors(orders_products_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors=13)

    # Prepare recommended products DataFrame
    reco = []
    for i in range(len(distances.flatten())):
        reco.append(orders_products_df.index[indices.flatten()[i]])
    reco = pd.DataFrame(reco, columns=['product_name'])
    reco = reco.merge(products, on='product_name')
    swap_list = ["product_id","product_name","aisle_id","department_id", "aisle", "department"]
    reco = reco.reindex(columns=swap_list)
    
    return reco
