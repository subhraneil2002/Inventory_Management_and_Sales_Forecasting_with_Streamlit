import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib  # For loading and saving the model


# Cache data loading to prevent reloading on every interaction
@st.cache_data
def load_data(file_path):
    encodings_to_try = ['utf-8', 'iso-8859-1', 'windows-1252', 'utf-16']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            st.success(f"CSV file read successfully using encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            st.warning(f"Failed to read CSV file using encoding: {encoding}")
    return None


# Cache the cleaning process
@st.cache_data
def clean_dataframe(df):
    # Trim whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure 'Quantity' and 'Material Description' columns exist
    if 'Quantity' not in df.columns or 'Material Description' not in df.columns:
        st.error("The required columns are missing from the data.")
        return None

    df['Quantity'] = df['Quantity'].astype(str).str.replace(',', '').astype(float).astype(int)
    return df


# Aggregate data by month
@st.cache_data
def aggregate_monthly_data(df):
    return df.groupby(['Year', 'Month', 'Material Description']).agg({'Quantity': 'sum'}).reset_index()


# Filter frequent products
@st.cache_data
def filter_frequent_products(df, min_orders=48):
    order_counts = df['Material Description'].value_counts()
    frequent_products = order_counts[order_counts >= min_orders].index
    return df[df['Material Description'].isin(frequent_products)]


# Cache the lag feature creation
@st.cache_data
def create_lag_features(df, quantity_column='Quantity', lag_values=[1, 3, 6, 12], ma_windows=[3, 6]):
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df.set_index('Date', inplace=True)
    df = df[[quantity_column]]

    for lag in lag_values:
        df[f'Lag_{lag}'] = df[quantity_column].shift(lag)

    for window in ma_windows:
        df[f'MA_{window}'] = df[quantity_column].rolling(window=window).mean()

    return df


# Load the pre-trained model or train it using GridSearchCV
@st.cache_resource
def load_or_train_model(X_train, y_train, retrain=False):
    model_path = "xgboost_tuned_model.pkl"

    if not retrain:
        try:
            model = joblib.load(model_path)
            st.success("Pre-trained model loaded successfully.")
            return model
        except FileNotFoundError:
            st.warning("No pre-trained model found, training a new model.")

    # Define the XGBoost model
    xgb_model = XGBRegressor()

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9]
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                               n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, model_path)
    st.success("Model trained with GridSearchCV and saved successfully.")
    return best_model


# Main function to drive the Streamlit app
def main():
    st.title("Sales Forecasting App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            cleaned_data = clean_dataframe(data)
            if cleaned_data is not None:
                aggregated_data = aggregate_monthly_data(cleaned_data)
                filtered_data = filter_frequent_products(aggregated_data)

                # Create lag features
                lagged_data = create_lag_features(filtered_data)

                # Assume X and y have been prepared from the lagged_data
                # Replace X_train, y_train with actual data from lagged_data
                X_train, y_train = None, None  # Replace this with actual data preparation logic

                # Load or train the model
                model = load_or_train_model(X_train, y_train, retrain=False)

                if model is not None:
                    # Display results or forecasts
                    st.write("Model ready to make predictions")
                else:
                    st.error("Model could not be loaded.")


if __name__ == '__main__':
    main()
