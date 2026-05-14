import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Dashboard title
st.title("Disease Forecast Dashboard")

# Load dataset
df = pd.read_csv("dataset for workshop.csv")

st.sidebar.header("Forecast Settings")

forecast_unit = st.sidebar.selectbox(
    "Forecast by:",
    ["Months", "Years"]
)

forecast_value = st.sidebar.number_input(
    f"Number of {forecast_unit.lower()} to forecast:",
    min_value=1,
    max_value=10 if forecast_unit == "Years" else 120,
    value=12 if forecast_unit == "Months" else 1
)

if forecast_unit == "Years":
    forecast_periods = forecast_value * 12
else:
    forecast_periods = forecast_value
  
# Convert date column
#df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())


# Prophet Forecasting
st.subheader("Facebook Prophet Forecast")

prophet_df = df[['date', 'disease_cases']]
prophet_df.columns = ['ds', 'y']

model = Prophet()
model.fit(prophet_df)

#future = model.make_future_dataframe(periods=30)
future = model.make_future_dataframe(periods=forecast_periods, freq='MS')

forecast = model.predict(future)

fig2 = model.plot(forecast)

st.pyplot(fig2)

# Show forecast table
st.subheader("Forecast Output")
st.write(forecast[['ds', 'yhat']].tail())
