import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def forecast_vitals(csv_path, anomaly_volume):
    df = pd.read_csv(csv_path)
    
    # --- START OF CODE MODIFICATION ---
    # 1. Create a synthetic timestamp column 'ds' (required for Prophet)
    df['ds'] = pd.to_datetime('2025-11-15') + pd.to_timedelta(df.index, unit='h')
    
    # 2. Rename the chosen vital column ('T1') to 'y'
    #    (If you want to use T2, T3, or T4, change 'T1' below)
    df = df.rename(columns={'T1': 'y'})
    # --- END OF CODE MODIFICATION ---

    df['ds'] = pd.to_datetime(df['ds']) # This line is redundant but can be kept
    
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=48, freq='H')
    # ... (rest of the code remains the same)