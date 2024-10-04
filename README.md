# Flight Delay Prediction App

This is a web-based application that predicts flight arrival delays based on various flight parameters such as departure delay, distance, taxi-out time, and more. The app uses machine learning (Linear Regression) to train a model on uploaded flight data and allows users to input flight information to make predictions on delays.

## Features

- **Upload Flight Data**: Upload a CSV file containing historical flight data.
- **Train a Model**: Train a Linear Regression model using the uploaded data.
- **Make Predictions**: Input specific flight information (airline, origin, destination, etc.) to predict the arrival delay.
- **Real-time Feedback**: View model performance using Root Mean Squared Error (RMSE) and predicted delay in real-time.

## Technologies Used

- **Dash**: The web framework used for creating the interactive dashboard.
- **Python**: Core programming language used for the app.
- **Scikit-learn**: Used for implementing the machine learning model (Linear Regression).
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical operations and handling large datasets.

## Prerequisites

Before running this app, ensure you have the following installed:

- Python 3.x

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/Emranbiswas/Flight-Delay-Prediction.git
   
2. **Install the required dependencies**

   ```bash
   pip install dash dash_core_components dash_html_components scikit-learn pandas numpy
   
3. **Run the application**

   ```bash
   python flight_delay_predictor.py

4. **Access the App**

   ```bash
   http://127.0.0.1:8050/

5. **Upload Data**

   Here is the dataset link: (airline_dataset_2023.csv](https://github.com/Emranbiswas/Flight-Delay-Prediction/blob/main/Data/airline_dataset_2023.csv)).

   
