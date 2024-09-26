import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


# Function to load data with multiple encodings
def load_data(file_path):
    encodings_to_try = ['utf-8', 'iso-8859-1', 'windows-1252', 'utf-16']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            st.success(f"CSV file read successfully using encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            st.warning(f"Failed to read CSV file using encoding: {encoding}")
    st.error("Unable to read CSV file with any of the common encodings.")
    return None


def clean_dataframe(df):
    # Trim whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure 'Quantity' and 'Material Description' columns exist
    if 'Quantity' not in df.columns or 'Material Description' not in df.columns:
        st.error("The required columns are missing from the data.")
        return None

    # Convert Quantity column to an integer type, handling any commas
    df['Quantity'] = df['Quantity'].astype(str).str.replace(',', '').astype(float).astype(int)
    return df


def aggregate_monthly_data(df):
    return df.groupby(['Year', 'Month', 'Material Description']).agg({'Quantity': 'sum'}).reset_index()


def filter_frequent_products(df, min_orders=48):
    order_counts = df['Material Description'].value_counts()
    frequent_products = order_counts[order_counts >= min_orders].index
    return df[df['Material Description'].isin(frequent_products)]


def create_lag_features(df, quantity_column='Quantity', lag_values=[1, 3, 6, 12], ma_windows=[3, 6]):
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df.set_index('Date', inplace=True)
    df = df[[quantity_column]]

    for lag in lag_values:
        df[f'Lag_{lag}'] = df[quantity_column].shift(lag)

    for window in ma_windows:
        df[f'MA_{window}'] = df[quantity_column].rolling(window=window).mean().shift(1)

    df.dropna(inplace=True)
    return df


def tune_xgboost(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9]
    }

    xgb_model = XGBRegressor(objective='reg:squarederror')
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1)
    xgb_grid_search.fit(X, y)

    return xgb_grid_search.best_estimator_, xgb_grid_search.best_params_


def forecast_next_year(model, last_known, num_months=12, feature_names=None):
    forecast = []

    # Forecasting for the fiscal year: April 2024 to March 2025
    for i in range(num_months):
        # Convert last_known to a DataFrame with the correct feature names
        last_known_df = pd.DataFrame([last_known], columns=feature_names)

        # Predict the next value
        next_pred = model.predict(last_known_df)
        forecast.append(np.ceil(next_pred[0]))  # Apply ceiling to the forecasted value

        # Update the lag features and moving averages
        last_known = np.roll(last_known, -1)  # Shift the lag features
        last_known[-6:] = np.append(last_known[-5:], next_pred)  # Update lags and moving average

    # Create forecast dates starting from April 2024
    forecast_dates = pd.date_range(start='2024-04-01', periods=num_months, freq='MS')

    # Create the forecast series with the future dates as the index
    forecast_series = pd.Series(forecast, index=forecast_dates)

    return forecast_series


def plot_forecast(forecast, work_order):
    plt.figure(figsize=(10, 6))
    plt.plot(forecast.index, forecast, marker='o', color='royalblue', label='Forecast')
    for month, quantity in forecast.items():
        plt.text(month, quantity, f'{int(quantity):.2f}', ha='center', va='bottom', fontsize=10, color='black')
    plt.title(f'Forecast for {work_order}', weight='bold', fontsize=15)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Quantity', fontsize=12)
    plt.xticks(forecast.index, forecast.index.strftime('%B %Y'), rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


# Streamlit application
st.title("Sales Forecasting App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Columns in the uploaded file:", df.columns)  # Display the columns
        df_cleaned = clean_dataframe(df)

        if df_cleaned is not None:  # Proceed only if cleaning was successful
            monthly_data = aggregate_monthly_data(df_cleaned)
            data_filtered = filter_frequent_products(monthly_data)

            work_order = st.selectbox("Select Material Description", data_filtered['Material Description'].unique())
            if st.button("Forecast"):
                selected_data = data_filtered[data_filtered['Material Description'] == work_order]

                # Generate lag features
                lag_data = create_lag_features(selected_data)

                # Ensure X (features) and y (target) are aligned
                X = lag_data.drop(columns='Quantity')
                y = lag_data['Quantity']

                if len(X) == len(y):  # Ensure X and y have the same length
                    feature_names = X.columns.tolist()  # Get feature names

                    # Train the model
                    model, best_params = tune_xgboost(X, y)

                    # Use the last known values to predict future sales
                    last_known = X.iloc[-1].values
                    forecast_series = forecast_next_year(model, last_known, num_months=12, feature_names=feature_names)

                    # Display the forecast
                    st.write(f"Forecast for {work_order}:")
                    for month, quantity in forecast_series.items():
                        st.write(f"{month.strftime('%B %Y')}: {int(quantity)}")

                    # Plot the forecast
                    plot_forecast(forecast_series, work_order)
