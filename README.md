# Predicting California Housing Prices Using Support Vector Regression

## Overview
This project focuses on predicting **median house values** in California districts using **Support Vector Regression (SVR)**. The dataset used is the **California Housing dataset**, which provides a rich set of features collected from the 1990 U.S. census. The goal is to train a robust regression model using SVR with carefully tuned hyperparameters to achieve predictive performance.

This project demonstrates key machine learning techniques, including data preprocessing, hyperparameter tuning, model evaluation, and performance interpretation.

## Skills Demonstrated
Through this project, the following skills were explored and applied:

1. **Data Exploration and Preprocessing**:
   - Inspection of data structure and feature types.
   - Handling missing values (none were found in this dataset).
   - Feature scaling using `StandardScaler` for optimal SVR performance.
2. **Modeling and Hyperparameter Tuning**:
   - Random sampling of training data for efficient hyperparameter tuning.
   - Implementation of **RandomizedSearchCV** to tune SVR hyperparameters (`C`, `epsilon`, and `kernel`).
   - Custom scoring using **Root Mean Squared Error (RMSE)** as the evaluation metric.
3. **Model Evaluation**:
   - Final model evaluation on the test set using multiple error metrics:
     - **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction error.
     - **Mean Absolute Percentage Error (MAPE)**: Represents the average percentage error.
     - **Normalized RMSE (NRMSE)**: Provides a relative error measure by normalizing RMSE with the range of target values.
4. **Performance Metrics Interpretation**:
   - Comparison of error metrics to assess model reliability and robustness.
   - Insights into the impact of hyperparameter choices on prediction accuracy.

## Dataset
The dataset used is the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), which consists of:

- **Number of Instances**: 20,640.
- **Features**:
  - `MedInc`: Median income in block group.
  - `HouseAge`: Median house age in block group.
  - `AveRooms`: Average number of rooms per household.
  - `AveBedrms`: Average number of bedrooms per household.
  - `Population`: Block group population.
  - `AveOccup`: Average number of household members.
  - `Latitude`: Block group latitude.
  - `Longitude`: Block group longitude.
- **Target**: `MedianHouseValue`, representing the median house price in hundreds of thousands of dollars.

## Results
- **Best Parameters**:  
  The best hyperparameters found using **RandomizedSearchCV** were:
  - `C`: 14.25
  - `epsilon`: 0.32
  - `kernel`: `rbf` (Radial Basis Function kernel)

- **Final Model Performance**:
  - **Root Mean Squared Error (RMSE)**: 0.5672 (equivalent to ~$56,720 error in house prices).
  - **Mean Absolute Percentage Error (MAPE)**: 20.40%.
  - **Normalized RMSE (NRMSE)**: 0.1170 (11.7% relative error).

These metrics indicate that the model achieves moderate accuracy and generalization.

## Requirements
Ensure you have Python 3.7 or higher and the following libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
