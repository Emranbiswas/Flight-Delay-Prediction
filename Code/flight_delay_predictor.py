import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Flight Delay Prediction App"),
    
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV'),
        multiple=False
    ),
    
    html.Div(id='file-upload'),
    
    html.Div([
        html.Button('Train Model', id='train-button', n_clicks=0),
        html.Div(id='rmse-output')
    ], style={'margin-top': '20px'}),
    
    html.Div([
        html.H2("Make Prediction"),
        html.Div([
            html.Label('Select Airline'),
            dcc.Dropdown(id='airline-dropdown', multi=False)
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label('Select Origin'),
            dcc.Dropdown(id='origin-dropdown', multi=False)
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label('Select Destination'),
            dcc.Dropdown(id='dest-dropdown', multi=False)
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label('Departure Delay (minutes)'),
            dcc.Input(id='dep-delay-input', type='number', placeholder='Departure Delay')
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label('Taxi Out Time (minutes)'),
            dcc.Input(id='taxi-out-input', type='number', placeholder='Taxi Out')
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label('Distance (miles)'),
            dcc.Input(id='distance-input', type='number', placeholder='Distance')
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label('Month'),
            dcc.Input(id='month-input', type='number', placeholder='Month')
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label('Day of Month'),
            dcc.Input(id='dayofmonth-input', type='number', placeholder='Day of Month')
        ], style={'margin-bottom': '10px'}),
        html.Button('Predict Delay', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output', style={'margin-top': '20px'})
    ])
])

@app.callback(
    [Output('airline-dropdown', 'options'),
     Output('origin-dropdown', 'options'),
     Output('dest-dropdown', 'options')],
    [Input('upload-data', 'contents')]
)
def update_dropdowns(contents):
    if contents is None:
        return [], [], []
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    airline_options = [{'label': val, 'value': val} for val in df['AIRLINE'].unique()]
    origin_options = [{'label': val, 'value': val} for val in df['ORIGIN'].unique()]
    dest_options = [{'label': val, 'value': val} for val in df['DEST'].unique()]
    
    return airline_options, origin_options, dest_options

@app.callback(
    [Output('rmse-output', 'children'),
     Output('prediction-output', 'children')],
    [Input('train-button', 'n_clicks'),
     Input('predict-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('airline-dropdown', 'value'),
     State('origin-dropdown', 'value'),
     State('dest-dropdown', 'value'),
     State('dep-delay-input', 'value'),
     State('taxi-out-input', 'value'),
     State('distance-input', 'value'),
     State('month-input', 'value'),
     State('dayofmonth-input', 'value')]
)
def update_rmse_and_prediction(train_clicks, predict_clicks, contents,
                                selected_airline, selected_origin, selected_dest, dep_delay, taxi_out, distance, month, dayofmonth):
    if contents is None:
        return "Please upload a dataset and train the model before making predictions.", None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Automatically select relevant features and target
    target = 'ARR_DELAY'
    features = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_DELAY', 'TAXI_OUT', 'DISTANCE', 'Month', 'DayofMonth']

    # Use a subset of the dataset if it's too large
    #sample_size = min(10000, len(df))
    #df = df.sample( random_state=42)
    
    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split the data
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    rmse_output = f"Flight RMSE: {rmse:.2f} minutes"
    
    # Predict user input data
    if predict_clicks > 0:
        # Prepare the input data
        input_data = {
            'DEP_DELAY': dep_delay if dep_delay is not None else 0,
            'TAXI_OUT': taxi_out if taxi_out is not None else 0,
            'DISTANCE': distance if distance is not None else 0,
            'Month': month if month is not None else 1,
            'DayofMonth': dayofmonth if dayofmonth is not None else 1,
            'AIRLINE': selected_airline,
            'ORIGIN': selected_origin,
            'DEST': selected_dest
        }
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure the input DataFrame has the same column order as during training
        input_df = input_df[features]
        
        # Encode categorical features in the input data
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        try:
            predicted_delay = model.predict(input_df)[0]
            prediction_output = f"Predicted Delay: {predicted_delay:.2f} minutes"
        except Exception as e:
            prediction_output = f"Prediction error: {str(e)}"
    else:
        prediction_output = None

    return rmse_output, prediction_output

if __name__ == '__main__':
    app.run_server(debug=True)
