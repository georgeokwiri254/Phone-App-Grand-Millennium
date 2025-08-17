"""
Fast and Efficient Time Series Forecasting for Hotel Occupancy
Optimized version with simpler models for faster execution
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from pathlib import Path

# Statistical models (simplified)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet for trend/seasonality modeling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Scikit-learn for ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastHotelForecasting:
    """Fast and efficient forecasting for hotel occupancy"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.models = {}
        self.forecasts = {}
    
    def load_data_from_sql(self, table_name: str = 'hotel_data_combined') -> pd.DataFrame:
        """Load data from SQL database"""
        logger.info(f"Loading data from SQL table: {table_name}")
        
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT Date, Occupancy_Pct, Revenue, ADR, RevPAR, Month, DayOfWeek, IsWeekend FROM {table_name} ORDER BY Date"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Remove future dates beyond current date
        current_date = datetime.now().date()
        df = df[df.index.date <= current_date]
        
        logger.info(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def simple_arima_forecast(self, series: pd.Series, steps: int) -> np.ndarray:
        """Simple ARIMA(1,1,1) forecast"""
        try:
            model = ARIMA(series, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=steps)
            return forecast.values
        except:
            # Fallback to simple trend
            return self.trend_forecast(series, steps)
    
    def exponential_smoothing_forecast(self, series: pd.Series, steps: int) -> np.ndarray:
        """Exponential smoothing with seasonality"""
        try:
            model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=7)
            fitted = model.fit()
            forecast = fitted.forecast(steps=steps)
            return forecast.values
        except:
            return self.trend_forecast(series, steps)
    
    def prophet_forecast(self, df: pd.DataFrame, target_col: str, steps: int) -> np.ndarray:
        """Prophet forecast"""
        if not PROPHET_AVAILABLE:
            return self.trend_forecast(df[target_col], steps)
        
        try:
            # Prepare data
            prophet_df = df.reset_index()[[target_col]].reset_index()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = df.index
            
            # Fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1
            )
            model.fit(prophet_df)
            
            # Generate future dates
            future = model.make_future_dataframe(periods=steps)
            forecast = model.predict(future)
            
            return forecast['yhat'].tail(steps).values
        except:
            return self.trend_forecast(df[target_col], steps)
    
    def ml_forecast(self, df: pd.DataFrame, target_col: str, steps: int) -> np.ndarray:
        """Machine learning forecast using Random Forest"""
        try:
            # Create features
            feature_df = df.copy()
            
            # Add lag features (simplified)
            for lag in [1, 7, 14]:
                feature_df[f'{target_col}_lag_{lag}'] = feature_df[target_col].shift(lag)
            
            # Add rolling features
            feature_df[f'{target_col}_ma_7'] = feature_df[target_col].rolling(7).mean()
            feature_df[f'{target_col}_ma_30'] = feature_df[target_col].rolling(30).mean()
            
            # Select features
            feature_cols = [col for col in feature_df.columns if 
                          ('lag_' in col or 'ma_' in col or col in ['Month', 'DayOfWeek', 'IsWeekend'])]
            
            # Remove NaN rows
            feature_df = feature_df.dropna()
            
            if len(feature_df) < 30:  # Not enough data
                return self.trend_forecast(df[target_col], steps)
            
            X = feature_df[feature_cols]
            y = feature_df[target_col]
            
            # Fit model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Generate forecasts
            forecasts = []
            current_df = feature_df.copy()
            
            for step in range(steps):
                # Get latest features
                latest_features = current_df[feature_cols].iloc[-1:].values
                pred = model.predict(latest_features)[0]
                forecasts.append(pred)
                
                # Update dataframe for next prediction
                new_date = current_df.index[-1] + timedelta(days=1)
                new_row = current_df.iloc[-1:].copy()
                new_row.index = [new_date]
                new_row[target_col] = pred
                
                # Update lag features
                for lag in [1, 7, 14]:
                    lag_col = f'{target_col}_lag_{lag}'
                    if lag_col in new_row.columns:
                        if lag == 1:
                            new_row[lag_col] = current_df[target_col].iloc[-1]
                        elif lag <= len(current_df):
                            new_row[lag_col] = current_df[target_col].iloc[-lag]
                
                current_df = pd.concat([current_df, new_row])
            
            return np.array(forecasts)
        except Exception as e:
            logger.warning(f"ML forecast failed: {e}")
            return self.trend_forecast(df[target_col], steps)
    
    def trend_forecast(self, series: pd.Series, steps: int) -> np.ndarray:
        """Simple trend-based forecast as fallback"""
        # Calculate recent trend
        recent_data = series.tail(30)
        trend = np.polyfit(range(len(recent_data)), recent_data.values, 1)[0]
        
        # Generate forecast
        last_value = series.iloc[-1]
        forecasts = []
        
        for i in range(steps):
            forecast = last_value + (trend * (i + 1))
            # Apply bounds for occupancy
            forecast = max(0, min(100, forecast))
            forecasts.append(forecast)
        
        return np.array(forecasts)
    
    def seasonal_naive_forecast(self, series: pd.Series, steps: int, season_length: int = 365) -> np.ndarray:
        """Seasonal naive forecast"""
        # Use last year's data for prediction
        if len(series) >= season_length:
            seasonal_data = series.tail(season_length).values
            forecasts = []
            
            for i in range(steps):
                seasonal_index = i % season_length
                forecasts.append(seasonal_data[seasonal_index])
            
            return np.array(forecasts)
        else:
            return self.trend_forecast(series, steps)
    
    def generate_all_forecasts(self, df: pd.DataFrame, target_col: str, horizons: List[int]) -> Dict[str, pd.DataFrame]:
        """Generate forecasts using all models"""
        logger.info("Generating forecasts using multiple models")
        
        series = df[target_col].dropna()
        all_forecasts = {}
        
        for horizon in horizons:
            logger.info(f"Generating {horizon}-day forecasts")
            
            # Generate future dates
            last_date = df.index.max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
            
            # Generate forecasts
            forecasts = {}
            
            # ARIMA
            forecasts['ARIMA'] = self.simple_arima_forecast(series, horizon)
            
            # Exponential Smoothing
            forecasts['Exponential_Smoothing'] = self.exponential_smoothing_forecast(series, horizon)
            
            # Prophet
            forecasts['Prophet'] = self.prophet_forecast(df, target_col, horizon)
            
            # Machine Learning
            forecasts['ML_Random_Forest'] = self.ml_forecast(df, target_col, horizon)
            
            # Seasonal Naive
            forecasts['Seasonal_Naive'] = self.seasonal_naive_forecast(series, horizon)
            
            # Create DataFrame
            forecast_df = pd.DataFrame(index=future_dates)
            for model_name, forecast_values in forecasts.items():
                if len(forecast_values) == horizon:
                    forecast_df[model_name] = forecast_values
                else:
                    logger.warning(f"{model_name} forecast length mismatch")
            
            # Calculate ensemble (median for robustness)
            forecast_df['Ensemble_Mean'] = forecast_df.mean(axis=1)
            forecast_df['Ensemble_Median'] = forecast_df.median(axis=1)
            
            all_forecasts[f'{horizon}_days'] = forecast_df
        
        return all_forecasts
    
    def validate_forecasts(self, df: pd.DataFrame, target_col: str, test_days: int = 60) -> Dict[str, Dict[str, float]]:
        """Simple validation using recent data"""
        logger.info(f"Validating forecasts using last {test_days} days")
        
        # Split data
        train_df = df.iloc[:-test_days]
        test_df = df.iloc[-test_days:]
        
        # Generate forecasts for validation period
        validation_forecasts = self.generate_all_forecasts(train_df, target_col, [test_days])
        
        # Calculate metrics
        actual_values = test_df[target_col].values
        results = {}
        
        if f'{test_days}_days' in validation_forecasts:
            forecast_df = validation_forecasts[f'{test_days}_days']
            
            for model_name in forecast_df.columns:
                predicted = forecast_df[model_name].values
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(actual_values, predicted))
                mae = mean_absolute_error(actual_values, predicted)
                mape = np.mean(np.abs((actual_values - predicted) / actual_values)) * 100
                
                results[model_name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape
                }
        
        return results
    
    def run_fast_forecast(self, target_col: str = 'Occupancy_Pct') -> Dict:
        """Run complete fast forecasting pipeline"""
        logger.info("Starting fast forecasting pipeline")
        
        # Load data
        df = self.load_data_from_sql()
        
        # Determine forecast horizons based on current date
        current_date = datetime.now()
        horizons = [90, 180, 365]  # 3, 6, 12 months
        
        logger.info(f"Current date: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"Forecast horizons: {horizons} days")
        
        # Validate models
        validation_results = self.validate_forecasts(df, target_col)
        
        # Generate final forecasts
        forecasts = self.generate_all_forecasts(df, target_col, horizons)
        
        # Store results
        self.forecasts = forecasts
        
        return {
            'forecasts': forecasts,
            'validation_results': validation_results,
            'data_info': {
                'shape': df.shape,
                'date_range': (df.index.min(), df.index.max()),
                'current_occupancy': df[target_col].tail(7).mean()
            }
        }
    
    def save_forecasts(self, output_dir: str):
        """Save forecasts to CSV files"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for horizon, forecast_df in self.forecasts.items():
            filename = f"occupancy_forecast_{horizon}_{timestamp}.csv"
            filepath = Path(output_dir) / filename
            forecast_df.to_csv(filepath)
            logger.info(f"Saved {horizon} forecast to {filepath}")
    
    def print_forecast_summary(self, results: Dict):
        """Print a summary of forecast results"""
        print("\n🏨 Hotel Occupancy Forecast Summary")
        print("=" * 60)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        print(f"📅 Current Date: {current_date}")
        print(f"📊 Current 7-day Avg Occupancy: {results['data_info']['current_occupancy']:.1f}%")
        
        print(f"\n📈 Forecast Periods:")
        for horizon, forecast_df in results['forecasts'].items():
            days = horizon.split('_')[0]
            ensemble_avg = forecast_df['Ensemble_Median'].mean()
            ensemble_min = forecast_df['Ensemble_Median'].min()
            ensemble_max = forecast_df['Ensemble_Median'].max()
            
            print(f"  {days:>3} days: Avg {ensemble_avg:5.1f}% | Range {ensemble_min:5.1f}% - {ensemble_max:5.1f}%")
        
        print(f"\n🎯 Model Validation Results:")
        print("Model                | RMSE  | MAE   | MAPE")
        print("-" * 45)
        for model, metrics in results['validation_results'].items():
            print(f"{model:18} | {metrics['RMSE']:5.1f} | {metrics['MAE']:5.1f} | {metrics['MAPE']:4.1f}%")


def main():
    """Main execution"""
    base_dir = "/home/gee_devops254/Downloads/Revenue Architecture"
    db_path = f"{base_dir}/db/revenue.db"
    output_dir = f"{base_dir}/forecasts"
    
    # Run forecasting
    forecaster = FastHotelForecasting(db_path)
    results = forecaster.run_fast_forecast()
    
    # Save results
    forecaster.save_forecasts(output_dir)
    
    # Print summary
    forecaster.print_forecast_summary(results)
    
    return results


if __name__ == "__main__":
    main()