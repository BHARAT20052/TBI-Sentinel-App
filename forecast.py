import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def forecast_vitals(csv_path: str, anomaly_percent: float) -> dict:
    """
    Mocks a forecasting model for patient vitals and generates a plot.
    
    - Reads the mock data (T1, T2, T3, T4 columns).
    - Generates a 48-hour forecast plot (forecast.png).
    - Determines risk based on the severity of the anomaly.
    """
    print(f"Simulating vitals forecast for {csv_path}...")
    
    # 1. Simulate Reading Data
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Create dummy data if file reading fails or is not available
        df = pd.DataFrame({
            'T1': np.random.uniform(70, 100, 100),
            'T2': np.random.uniform(70, 100, 100),
        })

    # 2. Mock Forecasting Plot (48 hours)
    plt.figure(figsize=(10, 5))
    
    # Use one of the time series columns for forecasting simulation
    data_to_forecast = df['T1'].iloc[-50:].values if 'T1' in df.columns else df.iloc[-50:, 0].values

    # Simulate a forecast based on the last 50 points
    forecast_points = 48
    last_val = data_to_forecast[-1]
    
    # Create a subtle trend (e.g., slightly increasing variability due to TBI)
    forecast_trend = np.linspace(last_val, last_val + (anomaly_percent * 0.5), forecast_points)
    forecast_noise = np.random.normal(0, 1.5, forecast_points) * (anomaly_percent / 2) # Higher anomaly = more noise
    forecast_data = forecast_trend + forecast_noise
    
    # Plotting
    plt.plot(np.arange(len(data_to_forecast)), data_to_forecast, label='Historical Data (Last 50 hours)', color='blue')
    plt.plot(np.arange(len(data_to_forecast), len(data_to_forecast) + forecast_points), forecast_data, label='48-Hour Forecast', linestyle='--', color='red')
    
    plt.title('Heart Rate Forecast: Next 48 Hours')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.savefig("forecast.png")
    plt.close()

    # 3. Determine Risk and Forecast Summary
    if anomaly_percent >= 3.0:
        risk_level = "Critical"
        forecast_summary = "High probability of severe neurological deterioration within 48 hours based on volume and vital instability."
    elif anomaly_percent >= 1.5:
        risk_level = "High"
        forecast_summary = "Significant risk of secondary injury; vitals show high volatility."
    else:
        risk_level = "Moderate"
        forecast_summary = "Vitals remain relatively stable, but continuous monitoring is essential."
        
    return {
        "risk": risk_level,
        "forecast": forecast_summary
    }