# Airline_Ticket

This project aims to predict the fare of airline tickets based on various features such as source, destination, arrival time, departure time, date of journey, duration, airline, and total stops. The goal is to analyze the data, extract meaningful insights, and develop machine learning models to accurately predict ticket prices.

## Problem Understanding

The main objectives of this project are:

1. Analyze the dataset and extract meaningful insights.
2. Clean and preprocess the data for modeling.
3. Predict the fare of airline tickets using machine learning algorithms.

## Data Collection

The dataset used in this project is stored in an Excel file named "Data_Train.xlsx" taken from Kaggle. It contains information about airline tickets including features like source, destination, arrival time, departure time, date of journey, duration, airline, total stops, and ticket price.

## Data Cleaning and Preparation

1. Checked for missing values in the dataset and dropped rows with missing values.
2. Changed the data types of 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' to DateTime.
3. Extracted additional features such as day, month, and year from the 'Date_of_Journey' column.
4. Extracted hour and minute information from the 'Dep_Time' and 'Arrival_Time' columns.
5. Preprocessed the 'Duration' column to handle cases where hours or minutes were missing.

## Data Visualization

Visualized various aspects of the dataset using Matplotlib, Seaborn, and Plotly libraries. Plots include bar plots, scatter plots, box plots, and histograms to understand relationships between features and the target variable (ticket price).

## Feature Engineering

1. One-hot encoded the 'Source' feature.
2. Applied target-guided encoding to the 'Airline' feature.
3. Encoded the 'Destination' feature using manual encoding.
4. Converted the 'Total_Stops' feature from categorical to numerical.

## Model Building

Implemented multiple machine learning algorithms including:

- Random Forest Regressor
- Support Vector Machine (SVM) with linear and RBF kernels
- Ridge Regression
- K-Nearest Neighbors (KNN)

Evaluated model performance using various metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Hyperparameter Tuning

Tuned the Random Forest model using RandomizedSearchCV to optimize its performance. Selected the best hyperparameters based on cross-validation results.

## Results

The Random Forest model achieved the highest accuracy of 81.63% after hyperparameter tuning. Other models also provided insights into ticket price prediction, with varying levels of accuracy.

## Future Work

Potential future work includes:
- Exploring advanced feature engineering techniques.
- Experimenting with different machine learning algorithms.
- Fine-tuning hyperparameters further to improve model performance.
- Incorporating additional data sources for better predictions.
