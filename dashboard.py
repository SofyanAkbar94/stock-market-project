import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import joblib

# Load your data and model
transformed_df = pd.read_parquet('local_data/transformed_df.parquet')
model = joblib.load('local_data/random_forest_model.joblib')

# Print the columns to verify the structure
print(transformed_df.columns)

# Adjust the column name if needed
stock_column = 'Stock' if 'Stock' in transformed_df.columns else 'Ticker'

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1('Financial Dashboard'),
    
    dcc.Graph(id='price-chart'),
    dcc.Graph(id='volume-chart'),
    
    html.Label('Select Stock:'),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': stock, 'value': stock} for stock in transformed_df[stock_column].unique()],
        value=transformed_df[stock_column].unique()[0]
    ),
    
    html.H2('Model Performance'),
    dcc.Graph(id='model-performance')
])

# Define callbacks to update charts
@app.callback(
    [Output('price-chart', 'figure'),
     Output('volume-chart', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_charts(selected_stock):
    filtered_df = transformed_df[transformed_df[stock_column] == selected_stock]
    
    price_chart = go.Figure(data=[
        go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], mode='lines', name='Close')
    ])
    price_chart.update_layout(title='Stock Prices', xaxis_title='Date', yaxis_title='Price')
    
    volume_chart = go.Figure(data=[
        go.Bar(x=filtered_df['Date'], y=filtered_df['Volume'], name='Volume')
    ])
    volume_chart.update_layout(title='Stock Volume', xaxis_title='Date', yaxis_title='Volume')
    
    return price_chart, volume_chart

@app.callback(
    Output('model-performance', 'figure'),
    [Input('stock-dropdown', 'value')]
)
def update_model_performance(selected_stock):
    # Placeholder for model performance
    performance_chart = go.Figure(data=[
        go.Bar(x=['Accuracy', 'Precision', 'Recall'], y=[0.9, 0.8, 0.85])
    ])
    performance_chart.update_layout(title='Model Performance', xaxis_title='Metric', yaxis_title='Value')
    
    return performance_chart

# # Run the app in local
# if __name__ == '__main__':
#     app.run_server(debug=True)

# Run the app using docker
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
