import pandas as pd
from application import app
from flask import render_template, request
from application.remommendationMethods.collaborativeFiltering import cf
from application.remommendationMethods.contentBasedFiltering import cbf
from application.remommendationMethods.pearsonsRCorrelation import prc
from application.remommendationMethods.products import prodList
from application.remommendationMethods.orders_products_df import ordersProducts


@app.route('/',methods=['POST','GET'])

def home():
    """
    Home Route.

    Returns:
    --------
    str
        Rendered HTML template for the home page.

    Notes:
    ------
    This route renders the home page with a list of items retrieved from
    the 'prodList' function. If a POST request is received and the button
    value is not "refresh", it invokes the collaborative filtering function
    'cf' to generate recommendations based on the selected product. It then
    renders the home page with the recommendations and updates the shopping cart.
    """
    return render_template('home.html', items = prodList())

@app.route('/collaborativeFiltering',methods=['POST','GET'])
def collaborativeFiltering():
    """
    Collaborative Filtering Route.

    Returns:
    --------
    str
        Rendered HTML template for the collaborative filtering page.

    Notes:
    ------
    This route performs collaborative filtering to generate recommendations
    based on the selected product. It retrieves product information and order-
    product data, filters out popular products, and generates recommendations
    using the 'cf' function. It then renders the collaborative filtering page
    with the recommendations and updates the shopping cart.
    """
    products = prodList()
    products = pd.DataFrame(products, columns =['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department'])
    orders_products_df = ordersProducts()
    finalProducts = pd.DataFrame(orders_products_df.index).merge(products, on='product_name')
    swap_list = ["product_id","product_name","aisle_id","department_id", "aisle", "department"]
    finalProducts = finalProducts.reindex(columns=swap_list)
    finalProductsList = finalProducts.to_numpy().tolist()
    if request.method == 'POST' and request.form.get('button') != "refresh":
        product = finalProducts.index[finalProducts['product_id'] == int(request.form.get('button'))]
        result = cf(product)
        item = result["product_name"][0]
        result = result[1:]
        return render_template('home.html', recommendation = result.to_numpy().tolist(), route = "/collaborativeFiltering", items = finalProductsList, cart = item)
    return render_template('home.html', recommendation = [], route = "/collaborativeFiltering", items = finalProductsList)


@app.route('/contentBasedFiltering', methods=['POST', 'GET'])
def contentBasedFiltering():
    """
    Content-Based Filtering Route.

    Returns:
    --------
    str
        Rendered HTML template for the content-based filtering page.

    Notes:
    ------
    This route performs content-based filtering to generate recommendations
    based on the selected product. It retrieves product information and uses
    the 'cbf' function to generate recommendations. It then renders the content-
    based filtering page with the recommendations and updates the shopping cart.
    """
    if request.method == 'POST' and request.form.get('button') != "refresh":
        # Load product data
        products = pd.DataFrame(prodList(), columns=['product_id', 'product_name', 'aisle_id', 'department_id', 'aisle', 'department'])
        # Extract the product ID from the button value
        button_value = request.form.get('button')
        if button_value and button_value.isdigit():
            product_id = int(button_value)
            if product_id in products['product_id'].values:
                # Retrieve the product name corresponding to the product ID
                product_name = products.loc[products['product_id'] == product_id, 'product_name'].iloc[0]
                
                # Perform content-based filtering using the product name
                recommendation = cbf(product_name)
                
                # Print debugging message to check the result
                print("Product Name:", product_name)
                print("Recommendation:", recommendation)
                
                # Render the template with the recommendation and cart item
                return render_template('home.html', recommendation=recommendation, route="/contentBasedFiltering", items=prodList(), cart=product_name)
        
        # If the product ID is not valid or not found, render the template without any recommendation or cart item
        return render_template('home.html', recommendation=[], route="/contentBasedFiltering", items=prodList())

    # If the request method is GET or the button value is "refresh", render the template without any recommendation or cart item
    return render_template('home.html', recommendation=[], route="/contentBasedFiltering", items=prodList())




@app.route('/pearsonsRCorrelation',methods=['POST','GET'])

def pearsonsRCorrelation():
    """
    Pearson's R Correlation Route.

    Returns:
    --------
    str
        Rendered HTML template for the Pearson's R Correlation page.

    Notes:
    ------
    This route performs Pearson's R correlation to generate recommendations.
    It uses the 'prc' function to calculate correlations and renders the
    Pearson's R Correlation page with the recommendations.
    """
    if request.method == 'POST':
        result = prc()
        return render_template('home.html', recommendation = result, items = prodList())
    return render_template('home.html', recommendation = [], route = "/pearsonsRCorrelation", items = prodList())

