import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def forecast_vitals(csv_path, anomaly_volume):
    try:
        df = pd.read_csv(csv_path)
        
        # --- ROBUST DATA PREP ---
        # 1. Create a synthetic timestamp column 'ds' (Prophet requirement)
        df['ds'] = pd.to_datetime('2025-11-15') + pd.to_timedelta(df.index, unit='h')
        
        # 2. Convert 'T1' (assumed heart rate) to numeric, dropping any invalid rows
        df['y'] = pd.to_numeric(df['T1'], errors='coerce')
        df = df.dropna(subset=['y']) 
        
        # Ensure we have enough data after cleanup (Prophet requires at least two data points)
        if len(df) < 2:
            raise ValueError("Not enough clean data for Prophet forecasting.")
        # --- END DATA PREP ---

        # Fit model
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=48, freq='H')
        forecast = m.predict(future)
        
        # Calculate risk and trend
        risk = "HIGH" if anomaly_volume > 15 else "LOW"
        
        # Save plot
        plt.figure(figsize=(8,4))
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.title("48-Hour Heart Rate Forecast")
        plt.legend()
        plt.savefig("forecast.png")
        plt.close()
        
        trend = "Rising" if forecast['yhat'].iloc[-1] > df['y'].mean() else "Stable"
        
        # Return SUCCESS dictionary
        return {"risk": risk, "forecast": f"{trend} trend detected"}
        
    except Exception as e:
        # Return a SAFE FAIL dictionary that your app.py can handle
        print(f"Prophet forecast failed: {e}")
        return {"risk": "ERROR", "forecast": "Forecasting failed due to internal data error."}