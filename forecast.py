import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load('xgboost_model.pkl')


def create_lag_features_for_prediction(df, quantity_column='Quantity', lag_values=[1, 3, 6, 12], ma_windows=[3, 6]):
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df.set_index('Date', inplace=True)
    df = df[[quantity_column]]

    for lag in lag_values:
        df[f'Lag_{lag}'] = df[quantity_column].shift(lag)

    for window in ma_windows:
        df[f'MA_{window}'] = df[quantity_column].rolling(window=window).mean().shift(1)

    df.dropna(inplace=True)
    return df


def forecast_next_year(model, last_known, last_date, num_months=12, feature_names=None):
    forecast = []
    for _ in range(num_months):
        last_known_df = pd.DataFrame([last_known], columns=feature_names)
        next_pred = model.predict(last_known_df)
        forecast.append(np.ceil(next_pred[0]))
        last_known = np.roll(last_known, -1)
        last_known[-1] = next_pred[0]

    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_months, freq='MS')
    return pd.Series(forecast, index=future_dates)


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
    df = pd.read_csv(uploaded_file)
    df_cleaned = df[['Year', 'Month', 'Quantity', 'Material Description']]

    monthly_data = df_cleaned.groupby(['Year', 'Month', 'Material Description']).agg({'Quantity': 'sum'}).reset_index()
    work_order = st.selectbox("Select Material Description", monthly_data['Material Description'].unique())

    if st.button("Forecast"):
        selected_data = monthly_data[monthly_data['Material Description'] == work_order]
        lag_data = create_lag_features_for_prediction(selected_data)

        X = lag_data.drop(columns='Quantity')
        last_known = X.iloc[-1].values
        last_date = lag_data.index[-1]
        forecast_series = forecast_next_year(model, last_known, last_date, feature_names=X.columns.tolist())

        st.write(f"Forecast for {work_order}:")
        for month, quantity in forecast_series.items():
            st.write(f"{month.strftime('%B %Y')}: {int(quantity)}")

        plot_forecast(forecast_series, work_order)
