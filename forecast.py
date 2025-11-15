# forecast.py (Modified)
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np # <-- NEW: Import numpy

def forecast_vitals(csv_path, anomaly_volume):
    df = pd.read_csv(csv_path)
    df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce').dropna() 
    df = df.rename(columns={'timestamp': 'ds', 'heart_rate': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=48, freq='H')
    forecast = m.predict(future)
    
    # --- Enhanced Risk Calculation ---
    historic_mean = df['y'].mean()
    
    # 1. Volume Risk Score (e.g., scale 0-5)
    volume_risk = np.clip(anomaly_volume / 10, 0, 5) 
    
    # 2. Forecast Deviation Risk (e.g., scale 0-5, based on max difference from mean)
    max_deviation = np.max(np.abs(forecast['yhat'].iloc[-48:] - historic_mean))
    deviation_risk = np.clip(max_deviation / 4, 0, 5) # 4 BPM deviation mapped to 1 risk point
    
    # Total risk (weighted)
    total_risk = round(0.6 * volume_risk + 0.4 * deviation_risk, 1) # TBI volume weighted higher
    
    # Map score to a clear category for the LLM
    if total_risk >= 4:
        risk_category = "CRITICAL"
    elif total_risk >= 2.5:
        risk_category = "HIGH"
    elif total_risk >= 1.0:
        risk_category = "MODERATE"
    else:
        risk_category = "LOW"
    # --------------------------------

    # Plotting code remains the same
    plt.figure(figsize=(8,4))
    plt.plot(df['ds'], df['y'], 'k.', label='Historical Data')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.title("48-Hour Heart Rate Forecast")
    plt.legend()
    plt.savefig("forecast.png")
    plt.close()
    
    trend = "Rising" if forecast['yhat'].iloc[-1] > historic_mean else "Stable"
    
    return {
        "risk_score": risk_category, # <-- Return the new risk category
        "forecast": f"{trend} trend detected (Max deviation: {max_deviation:.1f} BPM)"
    }