from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import joblib
import os

# Load the trained model
model = joblib.load( os.path.join('artifacts', 'kmeans_model.pkl'))

app = Flask(__name__)

cluster_names = {
    0: 'Silver',
    1: 'Platinum',
    2: 'Gold',
    3: 'Bronze'
}

cluster_details = {
    'Silver': {
        'Spendings': 'Not more than 100',
        'Product Purchase': 'Purchased less, mostly basic items',
        'Website Use': 'Visits website regularly',
        'Campaign Response': 'Accepted fewer campaigns',
        'Recommendation': 'Basic products, affordable items'
    },
    'Platinum': {
        'Spendings': 'greater than 1000 to 1500',
        'Product Purchase': 'Purchased more, especially Wine, Meat, and Gold products',
        'Website Use': 'Visits website regularly, purchases more from stores',
        'Campaign Response': 'Responded to most campaigns',
        'Recommendation': 'Premium products, high-end items'
    },
    'Gold': {
        'Spendings': 'from 500 to 1000',
        'Product Purchase': 'Purchased mid-range products',
        'Website Use': 'Visits website regularly',
        'Campaign Response': 'Responded to most campaigns',
        'Recommendation': 'Mid-range products, value-for-money items'
    },
    'Bronze': {
        'Spendings': 'about 200 to 250',
        'Product Purchase': 'Purchased all product categories',
        'Website Use': 'Visits website regularly, purchases more from stores',
        'Campaign Response': 'Accepted all campaigns',
        'Recommendation': 'Affordable products, daily essentials'
    }
}

plot_paths = [
    [
        "static/plots/cluster_0_campaign_acceptance.png",
        "static/plots/cluster_0_Habits.png",
        "static/plots/cluster_0_Habits.png"
    ],
    [
        "static/plots/cluster_1_products.png",
        "static/plots/cluster_1_campaign_acceptance.png",
        "static/plots/cluster_1_Habits.png"
    ],
    [
        "static/plots/cluster_2_products.png",
        "static/plots/cluster_2_campaign_acceptance.png",
        "static/plots/cluster_2_Habits.png"
    ],
    [
        "static/plots/cluster_3_products.png",
        "static/plots/cluster_3_campaign_acceptance.png",
        "static/plots/cluster_3_Habits.png"
    ]
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    income = data['income']
    age = data['age']
    month_customer = data['month_customer']
    total_spending = data['total_spending']
    children = data['children']

    input_data = pd.DataFrame({
        'Income': [income],
        'Age': [age],
        'Month_Customer': [month_customer],
        'Total_Spending': [total_spending],
        'children': [children]
    })

    cluster = model.predict(input_data)[0]
    cluster_name = cluster_names[cluster]
    details = cluster_details[cluster_name]

    plot_urls = [url_for('static', filename=path.split('static/')[1]) for path in plot_paths[cluster]]

    return jsonify({
        'cluster': cluster_name,
        'details': details,
        'plots': plot_urls
    })

if __name__ == '__main__':
    app.run(debug=True)
