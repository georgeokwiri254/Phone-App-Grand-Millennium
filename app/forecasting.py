"""
Forecasting Functions for Revenue Analytics
Implements ExponentialSmoothing and other time series forecasting methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import logging

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available, using fallback forecasting methods")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class RevenueForecastor:
    """Revenue forecasting using multiple methods"""
    
    def __init__(self):
        self.models = {}
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / 'models'
        self.models_dir.mkdir(exist_ok=True)
    
    def prepare_time_series(self, df: pd.DataFrame, date_col: str, value_col: str, freq: str = 'D') -> pd.Series:
        """
        Prepare time series data for forecasting
        
        Args:
            df: DataFrame with data
            date_col: Name of date column
            value_col: Name of value column
            freq: Frequency (D=daily, M=monthly, W=weekly)
            
        Returns:
            Time series with datetime index
        """
        try:
            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Set date as index and sort
            ts_df = df.set_index(date_col).sort_index()
            
            # Get the time series
            ts = ts_df[value_col]
            
            # Remove any infinite or very large values
            ts = ts.replace([np.inf, -np.inf], np.nan)
            ts = ts.dropna()
            
            # Ensure frequency
            if freq == 'D':
                ts = ts.asfreq('D', fill_value=0)
            elif freq == 'M':
                ts = ts.resample('M').sum()
            elif freq == 'W':
                ts = ts.resample('W').sum()
            
            logger.info(f"Prepared time series: {len(ts)} points, range {ts.index.min()} to {ts.index.max()}")
            return ts
            
        except Exception as e:
            logger.error(f"Error preparing time series: {e}")
            return pd.Series()
    
    def forecast_exponential_smoothing(self, ts: pd.Series, periods: int, 
                                     seasonal: Optional[str] = None, 
                                     seasonal_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Forecast using ExponentialSmoothing
        
        Args:
            ts: Time series data
            periods: Number of periods to forecast
            seasonal: Type of seasonality ('add', 'mul', None)
            seasonal_periods: Number of periods in a season
            
        Returns:
            DataFrame with forecast, lower_ci, upper_ci
        """
        try:
            if not STATSMODELS_AVAILABLE:
                logger.warning("statsmodels not available, using linear trend fallback")
                return self.forecast_linear_trend(ts, periods)
            
            if len(ts) < 10:
                logger.warning("Insufficient data for ExponentialSmoothing, using linear trend")
                return self.forecast_linear_trend(ts, periods)
            
            # Fit ExponentialSmoothing model
            if seasonal and seasonal_periods and len(ts) >= 2 * seasonal_periods:
                model = ExponentialSmoothing(
                    ts,
                    trend='add',
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods
                )
            else:
                # Simple exponential smoothing with trend
                model = ExponentialSmoothing(ts, trend='add')
            
            fitted_model = model.fit(optimized=True)
            
            # Generate forecast
            forecast = fitted_model.forecast(periods)
            
            # Calculate confidence intervals (simple approximation)
            residuals = fitted_model.resid
            mse = np.mean(residuals**2)
            std_error = np.sqrt(mse)
            
            # 95% confidence interval
            confidence_factor = 1.96
            lower_ci = forecast - confidence_factor * std_error
            upper_ci = forecast + confidence_factor * std_error
            
            # Create result DataFrame
            forecast_dates = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=ts.index.freq or 'D'
            )
            
            result_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast.values,
                'lower_ci': lower_ci.values,
                'upper_ci': upper_ci.values
            })
            
            # Save model
            model_path = self.models_dir / f'exp_smoothing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
            joblib.dump(fitted_model, model_path)
            
            logger.info(f"ExponentialSmoothing forecast completed: {periods} periods")
            return result_df
            
        except Exception as e:
            logger.error(f"ExponentialSmoothing failed: {e}")
            return self.forecast_linear_trend(ts, periods)
    
    def forecast_linear_trend(self, ts: pd.Series, periods: int) -> pd.DataFrame:
        """
        Fallback linear trend forecasting
        
        Args:
            ts: Time series data
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecast, lower_ci, upper_ci
        """
        try:
            if len(ts) < 2:
                # Not enough data, return zeros
                forecast_dates = pd.date_range(
                    start=datetime.now().date(),
                    periods=periods,
                    freq='D'
                )
                return pd.DataFrame({
                    'date': forecast_dates,
                    'forecast': [0] * periods,
                    'lower_ci': [0] * periods,
                    'upper_ci': [0] * periods
                })
            
            # Prepare data for linear regression
            X = np.arange(len(ts)).reshape(-1, 1)
            y = ts.values
            
            # Fit linear model
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future X values
            future_X = np.arange(len(ts), len(ts) + periods).reshape(-1, 1)
            
            # Predict
            forecast = model.predict(future_X)
            
            # Calculate confidence intervals
            residuals = y - model.predict(X)
            mse = np.mean(residuals**2)
            std_error = np.sqrt(mse)
            
            confidence_factor = 1.96
            lower_ci = forecast - confidence_factor * std_error
            upper_ci = forecast + confidence_factor * std_error
            
            # Ensure non-negative values for some metrics
            forecast = np.maximum(forecast, 0)
            lower_ci = np.maximum(lower_ci, 0)
            
            # Create result DataFrame
            forecast_dates = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            result_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            })
            
            logger.info(f"Linear trend forecast completed: {periods} periods")
            return result_df
            
        except Exception as e:
            logger.error(f"Linear trend forecasting failed: {e}")
            # Return zero forecast as last resort
            forecast_dates = pd.date_range(
                start=datetime.now().date(),
                periods=periods,
                freq='D'
            )
            return pd.DataFrame({
                'date': forecast_dates,
                'forecast': [0] * periods,
                'lower_ci': [0] * periods,
                'upper_ci': [0] * periods
            })
    
    def forecast_occupancy(self, occupancy_df: pd.DataFrame, target_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Forecast daily occupancy from today to end of current month
        
        Args:
            occupancy_df: DataFrame with occupancy data
            target_date: Date to forecast from (defaults to today)
            
        Returns:
            DataFrame with occupancy forecast
        """
        try:
            if target_date is None:
                target_date = datetime.now().date()
            
            # Calculate end of month
            if target_date.month == 12:
                end_of_month = datetime(target_date.year + 1, 1, 1).date() - timedelta(days=1)
            else:
                end_of_month = datetime(target_date.year, target_date.month + 1, 1).date() - timedelta(days=1)
            
            periods = (end_of_month - target_date).days + 1
            
            if periods <= 0:
                logger.warning("Target date is past end of month")
                return pd.DataFrame()
            
            # Prepare occupancy time series
            occ_ts = self.prepare_time_series(occupancy_df, 'Date', 'Occ%', 'D')
            
            if len(occ_ts) == 0:
                logger.error("No valid occupancy data for forecasting")
                return pd.DataFrame()
            
            # Forecast using ExponentialSmoothing
            forecast_df = self.forecast_exponential_smoothing(occ_ts, periods)
            
            # Ensure occupancy stays within reasonable bounds (0-100%)
            forecast_df['forecast'] = forecast_df['forecast'].clip(0, 100)
            forecast_df['lower_ci'] = forecast_df['lower_ci'].clip(0, 100)
            forecast_df['upper_ci'] = forecast_df['upper_ci'].clip(0, 100)
            
            logger.info(f"Occupancy forecast: {periods} days from {target_date} to {end_of_month}")
            return forecast_df
            
        except Exception as e:
            logger.error(f"Occupancy forecasting failed: {e}")
            return pd.DataFrame()
    
    def forecast_segment_revenue(self, segment_df: pd.DataFrame, months: int = 3) -> pd.DataFrame:
        """
        Forecast Business on the Books Revenue by segment for next N months
        
        Args:
            segment_df: DataFrame with segment data
            months: Number of months to forecast
            
        Returns:
            DataFrame with revenue forecast by segment
        """
        try:
            if 'Business_on_the_Books_Revenue' not in segment_df.columns:
                logger.error("Business_on_the_Books_Revenue column not found")
                return pd.DataFrame()
            
            # Group by month and merged segment
            monthly_revenue = segment_df.groupby(['Month', 'MergedSegment'])['Business_on_the_Books_Revenue'].sum().reset_index()
            
            forecasts = []
            
            # Forecast for each segment
            segments = monthly_revenue['MergedSegment'].unique()
            
            for segment in segments:
                segment_data = monthly_revenue[monthly_revenue['MergedSegment'] == segment]
                
                if len(segment_data) < 2:
                    logger.warning(f"Insufficient data for segment {segment}")
                    continue
                
                # Prepare time series
                segment_ts = self.prepare_time_series(segment_data, 'Month', 'Business_on_the_Books_Revenue', 'M')
                
                if len(segment_ts) == 0:
                    continue
                
                # Forecast
                forecast_df = self.forecast_exponential_smoothing(segment_ts, months)
                forecast_df['segment'] = segment
                
                forecasts.append(forecast_df)
            
            if forecasts:
                result_df = pd.concat(forecasts, ignore_index=True)
                logger.info(f"Segment revenue forecast: {months} months for {len(segments)} segments")
                return result_df
            else:
                logger.error("No successful segment forecasts generated")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Segment revenue forecasting failed: {e}")
            return pd.DataFrame()
    
    def calculate_forecast_metrics(self, actual: pd.Series, predicted: pd.Series) -> dict:
        """
        Calculate forecast accuracy metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            }
            
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {e}")
            return {}

def get_forecaster() -> RevenueForecastor:
    """Get forecaster instance - singleton pattern"""
    if not hasattr(get_forecaster, '_instance'):
        get_forecaster._instance = RevenueForecastor()
    return get_forecaster._instance