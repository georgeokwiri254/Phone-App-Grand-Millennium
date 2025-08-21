#!/usr/bin/env python3
"""
Advanced Time Series Forecasting for Monthly Revenue Analytics
Implements Prophet, ARIMA, and SARIMAX models with seasonality detection
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Time Series Libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AdvancedTimeSeriesForecaster:
    """Advanced time series forecasting with multiple models and validation"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.model_performance = {}
        self.data = None
        self.prepared_data = {}
        
    def load_data(self, df: pd.DataFrame) -> bool:
        """Load and validate monthly data, filtering to actual historical data only"""
        try:
            # Validate required columns
            required_cols = ['Year', 'Month', 'Total_Revenue', 'Avg_ADR', 'Avg_RevPar', 'Avg_Occupancy_Pct']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Create datetime index from Year and Month columns
            df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
            df = df.sort_values('date').reset_index(drop=True)
            
            # CRITICAL: Filter to actual historical data only (up to Aug 2025)
            # Do NOT use projected data from Sep-Dec 2025 for training
            current_date = datetime(2025, 8, 31)  # End of August 2025
            df = df[df['date'] <= current_date].copy()
            
            logger.info(f"🔥 FILTERED DATA: Using only actual historical data up to {current_date.strftime('%Y-%m')}")
            logger.info(f"Training data range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
            
            if len(df) == 0:
                logger.error("No historical data available after filtering")
                return False
            
            # Ensure numeric columns are properly typed
            numeric_cols = ['Total_Revenue', 'Avg_ADR', 'Avg_RevPar', 'Avg_Occupancy_Pct']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Store data
            self.data = df.copy()
            logger.info(f"✅ Loaded {len(df)} ACTUAL monthly records for training")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def prepare_data_for_models(self):
        """Prepare data for different forecasting models"""
        if self.data is None:
            return False
            
        try:
            # Prepare data for each target variable
            targets = ['Total_Revenue', 'Avg_ADR', 'Avg_RevPar', 'Avg_Occupancy_Pct']
            
            for target in targets:
                # Prophet format (ds, y)
                prophet_data = pd.DataFrame({
                    'ds': self.data['date'],
                    'y': self.data[target]
                })
                
                # ARIMA/SARIMAX format (time series)
                ts_data = self.data.set_index('date')[target]
                
                self.prepared_data[target] = {
                    'prophet': prophet_data,
                    'timeseries': ts_data,
                    'original': self.data[['date', target]].rename(columns={target: 'y'})
                }
            
            logger.info("Data prepared for all forecasting models")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """Check stationarity using ADF and KPSS tests"""
        try:
            # ADF Test (null hypothesis: non-stationary)
            adf_result = adfuller(series.dropna())
            adf_stationary = adf_result[1] <= 0.05
            
            # KPSS Test (null hypothesis: stationary)
            kpss_result = kpss(series.dropna(), regression='c')
            kpss_stationary = kpss_result[1] > 0.05
            
            return {
                'adf_pvalue': adf_result[1],
                'adf_stationary': adf_stationary,
                'kpss_pvalue': kpss_result[1],
                'kpss_stationary': kpss_stationary,
                'is_stationary': adf_stationary and kpss_stationary
            }
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return {'is_stationary': False}
    
    def fit_prophet_model(self, target: str, forecast_periods: int = 15) -> bool:
        """Fit Prophet model with automatic seasonality detection"""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available")
            return False
            
        try:
            data = self.prepared_data[target]['prophet'].copy()
            
            # Initialize Prophet with stronger seasonality detection for hotel patterns
            model = Prophet(
                yearly_seasonality=True,  # Capture winter/summer seasonality
                weekly_seasonality=False,  # Monthly data doesn't need weekly
                daily_seasonality=False,
                seasonality_mode='multiplicative',  # Better for hotel revenue data
                interval_width=0.95,
                changepoint_prior_scale=0.1,  # Allow more flexibility for seasonal changes
                seasonality_prior_scale=15.0,  # Stronger seasonality (Dubai peak/low seasons)
                holidays_prior_scale=10.0  # Strong holiday effects
            )
            
            # Add custom seasonalities for Dubai hotel patterns
            model.add_seasonality(
                name='peak_winter_season', 
                period=365.25, 
                fourier_order=8,  # Higher order to capture complex patterns
                condition_name='is_peak_season'
            )
            
            # Add seasonal indicators to training data
            data['is_peak_season'] = data['ds'].dt.month.isin([1, 2, 3, 10, 11, 12])
            
            # Fit model
            model.fit(data)
            
            # Generate future dates
            future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
            
            # Add seasonal indicators to future data
            future['is_peak_season'] = future['ds'].dt.month.isin([1, 2, 3, 10, 11, 12])
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Store model and forecast
            self.models[f'prophet_{target}'] = model
            self.forecasts[f'prophet_{target}'] = forecast
            
            logger.info(f"Prophet model fitted for {target}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model for {target}: {e}")
            return False
    
    def fit_arima_model(self, target: str, forecast_periods: int = 15) -> bool:
        """Fit ARIMA model with automatic order selection"""
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available")
            return False
            
        try:
            series = self.prepared_data[target]['timeseries'].copy()
            
            # Check stationarity
            stationarity = self.check_stationarity(series)
            
            # Simple ARIMA order selection (can be enhanced with grid search)
            if stationarity['is_stationary']:
                order = (1, 0, 1)  # AR(1), no differencing, MA(1)
            else:
                order = (1, 1, 1)  # AR(1), first difference, MA(1)
            
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.get_forecast(steps=forecast_periods)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            
            # Store results
            forecast_df = pd.DataFrame({
                'forecast': forecast_mean,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1]
            })
            
            self.models[f'arima_{target}'] = fitted_model
            self.forecasts[f'arima_{target}'] = forecast_df
            
            logger.info(f"ARIMA{order} model fitted for {target}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model for {target}: {e}")
            return False
    
    def fit_sarimax_model(self, target: str, forecast_periods: int = 15) -> bool:
        """Fit SARIMAX model with seasonal components"""
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available")
            return False
            
        try:
            series = self.prepared_data[target]['timeseries'].copy()
            
            # SARIMAX with seasonal components (monthly data, yearly season = 12)
            # Using conservative parameters for stability
            order = (1, 1, 1)  # Non-seasonal: AR(1), I(1), MA(1)
            seasonal_order = (1, 1, 1, 12)  # Seasonal: AR(1), I(1), MA(1), period=12
            
            # Fit SARIMAX model
            model = SARIMAX(
                series, 
                order=order, 
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            # Generate forecast
            forecast = fitted_model.get_forecast(steps=forecast_periods)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            
            # Store results
            forecast_df = pd.DataFrame({
                'forecast': forecast_mean,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1]
            })
            
            self.models[f'sarimax_{target}'] = fitted_model
            self.forecasts[f'sarimax_{target}'] = forecast_df
            
            logger.info(f"SARIMAX{order}x{seasonal_order} model fitted for {target}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting SARIMAX model for {target}: {e}")
            return False
    
    def generate_forecast_dates(self, periods: int) -> pd.DatetimeIndex:
        """Generate future monthly dates starting from September 2025"""
        # Start forecast from September 2025 (after August 2025 actuals)
        start_date = datetime(2025, 9, 1)
        return pd.date_range(start=start_date, periods=periods, freq='MS')
    
    def fit_all_models(self, target: str = 'Total_Revenue', forecast_periods: int = 15) -> Dict:
        """Fit all available models for a target variable"""
        results = {}
        
        # Prepare data if not already done
        if not self.prepared_data:
            if not self.prepare_data_for_models():
                return results
        
        # Fit Prophet
        if self.fit_prophet_model(target, forecast_periods):
            results['prophet'] = True
            
        # Fit ARIMA
        if self.fit_arima_model(target, forecast_periods):
            results['arima'] = True
            
        # Fit SARIMAX
        if self.fit_sarimax_model(target, forecast_periods):
            results['sarimax'] = True
            
        return results
    
    def get_forecast_summary(self, target: str = 'Total_Revenue') -> pd.DataFrame:
        """Get forecast summary from all fitted models starting from Sep 2025"""
        try:
            # Define forecast start date (Sep 2025) - NOT based on last historical date
            forecast_start_date = datetime(2025, 9, 1)
            
            summaries = []
            
            # Prophet forecast
            if f'prophet_{target}' in self.forecasts:
                prophet_forecast = self.forecasts[f'prophet_{target}']
                # Filter to get only forecasts from Sep 2025 onwards
                future_forecast = prophet_forecast[prophet_forecast['ds'] >= forecast_start_date]
                
                for _, row in future_forecast.iterrows():
                    summaries.append({
                        'date': row['ds'],
                        'model': 'Prophet',
                        'forecast': row['yhat'],
                        'lower_ci': row['yhat_lower'],
                        'upper_ci': row['yhat_upper']
                    })
            
            # ARIMA forecast
            if f'arima_{target}' in self.forecasts:
                arima_forecast = self.forecasts[f'arima_{target}']
                # Generate dates starting from Sep 2025
                future_dates = self.generate_forecast_dates(len(arima_forecast))
                
                for date, (_, row) in zip(future_dates, arima_forecast.iterrows()):
                    summaries.append({
                        'date': date,
                        'model': 'ARIMA',
                        'forecast': row['forecast'],
                        'lower_ci': row['lower_ci'],
                        'upper_ci': row['upper_ci']
                    })
            
            # SARIMAX forecast
            if f'sarimax_{target}' in self.forecasts:
                sarimax_forecast = self.forecasts[f'sarimax_{target}']
                # Generate dates starting from Sep 2025
                future_dates = self.generate_forecast_dates(len(sarimax_forecast))
                
                for date, (_, row) in zip(future_dates, sarimax_forecast.iterrows()):
                    summaries.append({
                        'date': date,
                        'model': 'SARIMAX',
                        'forecast': row['forecast'],
                        'lower_ci': row['lower_ci'],
                        'upper_ci': row['upper_ci']
                    })
            
            forecast_df = pd.DataFrame(summaries)
            
            if not forecast_df.empty:
                logger.info(f"📅 Forecast period: {forecast_df['date'].min().strftime('%Y-%m')} to {forecast_df['date'].max().strftime('%Y-%m')}")
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error getting forecast summary: {e}")
            return pd.DataFrame()
    
    def create_forecast_visualization(self, target: str = 'Total_Revenue') -> go.Figure:
        """Create interactive forecast visualization"""
        try:
            # Get historical data
            historical = self.data[['date', target]].copy()
            historical.columns = ['date', 'value']
            
            # Get forecasts
            forecast_summary = self.get_forecast_summary(target)
            
            # Create figure
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f'{target} Forecast Comparison']
            )
            
            # Plot historical data
            fig.add_trace(
                go.Scatter(
                    x=historical['date'],
                    y=historical['value'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Plot forecasts by model
            models = forecast_summary['model'].unique()
            colors = ['red', 'green', 'orange', 'purple']
            
            for i, model in enumerate(models):
                model_data = forecast_summary[forecast_summary['model'] == model]
                
                # Main forecast line
                fig.add_trace(
                    go.Scatter(
                        x=model_data['date'],
                        y=model_data['forecast'],
                        mode='lines+markers',
                        name=f'{model} Forecast',
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
                
                # Confidence intervals
                fig.add_trace(
                    go.Scatter(
                        x=model_data['date'],
                        y=model_data['upper_ci'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=model_data['date'],
                        y=model_data['lower_ci'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor=colors[i % len(colors)].replace('1)', '0.2)'),
                        line=dict(width=0),
                        name=f'{model} CI',
                        hoverinfo='skip'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f'{target} Forecasting Comparison',
                xaxis_title='Date',
                yaxis_title=target,
                hovermode='x unified',
                showlegend=True,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating forecast visualization: {e}")
            return go.Figure()

def create_forecaster() -> AdvancedTimeSeriesForecaster:
    """Create and return forecaster instance"""
    return AdvancedTimeSeriesForecaster()