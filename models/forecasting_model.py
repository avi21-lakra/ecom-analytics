import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

# models/ -> ecom-olist-analytics/data/processed
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

def run_forecast():
    print("Loading daily_revenue.csv...")
    daily = pd.read_csv(PROC / "daily_revenue.csv", parse_dates=["date"])
    daily = daily.set_index("date").asfreq("D").fillna(0)

    y = daily["total_revenue"]

    print("Training SARIMA model...")
    model = SARIMAX(
        y,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    print(" Forecasting next 30 days...")
    forecast = results.get_forecast(steps=30).predicted_mean.reset_index()
    forecast.columns = ["date", "forecast_revenue"]

    forecast.to_csv(PROC / "revenue_forecast_sarima.csv", index=False)
    print("✅ Forecast saved!")

if __name__ == "__main__":
    run_forecast()
