"""
Improved Time Series Forecasting for Hotel Occupancy and Revenue
Implements multiple forecasting models with proper validation and ensemble methods
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prophet for trend/seasonality modeling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Scikit-learn for ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HotelForecastingEngine:
    """Advanced forecasting engine for hotel occupancy and revenue metrics"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.models = {}
        self.forecasts = {}
        self.validation_results = {}
        self.ensemble_weights = {}
        
    def load_data_from_sql(self, table_name: str = 'hotel_data_combined') -> pd.DataFrame:
        """Load the cleaned data from SQL database"""
        logger.info(f"Loading data from SQL table: {table_name}")
        
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM {table_name} ORDER BY Date"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        logger.info(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def check_stationarity(self, series: pd.Series, series_name: str) -> Dict[str, Any]:
        """Check stationarity using Augmented Dickey-Fuller test"""
        logger.info(f"Checking stationarity for {series_name}")
        
        result = adfuller(series.dropna())
        stationarity_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        logger.info(f"{series_name} - ADF p-value: {result[1]:.4f}, Stationary: {stationarity_result['is_stationary']}")
        return stationarity_result
    
    def make_stationary(self, series: pd.Series, method: str = 'diff') -> pd.Series:
        """Make time series stationary"""
        if method == 'diff':
            return series.diff().dropna()
        elif method == 'log_diff':
            return np.log(series).diff().dropna()
        elif method == 'seasonal_diff':
            return series.diff(365).dropna()  # Annual seasonal differencing
        else:
            return series
    
    def prepare_data_for_modeling(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for time series modeling"""
        logger.info(f"Preparing data for modeling - target: {target_col}")
        
        # Ensure we have the target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Get target series
        target_series = df[target_col].copy()
        
        # Check for stationarity
        stationarity = self.check_stationarity(target_series, target_col)
        
        # Make stationary if needed
        if not stationarity['is_stationary']:
            logger.info(f"Making {target_col} stationary using differencing")
            # Try simple differencing first
            diff_series = self.make_stationary(target_series, 'diff')
            diff_stationarity = self.check_stationarity(diff_series, f"{target_col}_diff")
            
            if not diff_stationarity['is_stationary']:
                # Try seasonal differencing
                target_series = self.make_stationary(target_series, 'seasonal_diff')
                logger.info(f"Applied seasonal differencing to {target_col}")
        
        return df, target_series
    
    def split_data_time_aware(self, df: pd.DataFrame, test_size: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data maintaining temporal order"""
        split_date = df.index[-test_size]
        train_df = df[df.index < split_date]
        test_df = df[df.index >= split_date]
        
        logger.info(f"Train period: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} days)")
        logger.info(f"Test period: {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} days)")
        
        return train_df, test_df
    
    def fit_arima_model(self, series: pd.Series, order: Tuple[int, int, int] = None) -> ARIMA:
        """Fit ARIMA model with automatic order selection if not specified"""
        if order is None:
            # Simple heuristic for order selection
            # In practice, you'd use auto_arima or grid search
            order = (2, 1, 2)  # Conservative default
        
        logger.info(f"Fitting ARIMA model with order {order}")
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        return fitted_model
    
    def fit_sarima_model(self, series: pd.Series, 
                         order: Tuple[int, int, int] = (2, 1, 2),
                         seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 365)) -> SARIMAX:
        """Fit SARIMA model for seasonal data"""
        logger.info(f"Fitting SARIMA model with order {order} and seasonal_order {seasonal_order}")
        
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        return fitted_model
    
    def fit_ets_model(self, series: pd.Series) -> ETSModel:
        """Fit Exponential Smoothing (ETS) model"""
        logger.info("Fitting ETS model")
        
        model = ETSModel(series, trend='add', seasonal='add', seasonal_periods=365)
        fitted_model = model.fit()
        
        return fitted_model
    
    def fit_prophet_model(self, df: pd.DataFrame, target_col: str) -> Optional[Any]:
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping Prophet model")
            return None
        
        logger.info("Fitting Prophet model")
        
        # Prepare data for Prophet
        prophet_df = df.reset_index()[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
        
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Add custom regressors if available
        regressor_cols = ['IsWeekend', 'Month', 'Quarter']
        for col in regressor_cols:
            if col in df.columns:
                model.add_regressor(col)
                prophet_df[col] = df[col].values
        
        fitted_model = model.fit(prophet_df)
        return fitted_model
    
    def fit_ml_models(self, df: pd.DataFrame, target_col: str, n_lags: int = 30) -> Dict[str, Any]:
        """Fit machine learning models using lagged features"""
        logger.info("Fitting ML models (Random Forest, Gradient Boosting)")
        
        # Create lagged features
        feature_df = df.copy()
        
        # Add more lag features
        for lag in range(1, n_lags + 1):
            feature_df[f'{target_col}_lag_{lag}'] = feature_df[target_col].shift(lag)
        
        # Add rolling features
        for window in [7, 14, 30]:
            feature_df[f'{target_col}_rolling_mean_{window}'] = feature_df[target_col].rolling(window).mean()
            feature_df[f'{target_col}_rolling_std_{window}'] = feature_df[target_col].rolling(window).std()
        
        # Select features
        feature_cols = [col for col in feature_df.columns if 
                       ('lag_' in col or 'rolling_' in col or 'MA_' in col or 
                        col in ['Month', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear'])]
        
        # Remove rows with NaN values
        feature_df = feature_df.dropna()
        
        X = feature_df[feature_cols]
        y = feature_df[target_col]
        
        models = {}
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X, y)
        models['gradient_boosting'] = gb_model
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        models['linear_regression'] = lr_model
        
        # Store feature columns for prediction
        models['feature_cols'] = feature_cols
        models['feature_df'] = feature_df
        
        return models
    
    def generate_forecasts(self, models: Dict[str, Any], df: pd.DataFrame, 
                          target_col: str, horizons: List[int] = [90, 180, 365]) -> Dict[str, pd.DataFrame]:
        """Generate forecasts for different horizons"""
        logger.info(f"Generating forecasts for horizons: {horizons}")
        
        forecasts = {}
        last_date = df.index.max()
        
        for horizon in horizons:
            horizon_forecasts = {}
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
            
            # ARIMA forecast
            if 'arima' in models:
                try:
                    arima_forecast = models['arima'].forecast(steps=horizon)
                    horizon_forecasts['ARIMA'] = arima_forecast
                except Exception as e:
                    logger.error(f"ARIMA forecast failed: {e}")
            
            # SARIMA forecast
            if 'sarima' in models:
                try:
                    sarima_forecast = models['sarima'].forecast(steps=horizon)
                    horizon_forecasts['SARIMA'] = sarima_forecast
                except Exception as e:
                    logger.error(f"SARIMA forecast failed: {e}")
            
            # ETS forecast
            if 'ets' in models:
                try:
                    ets_forecast = models['ets'].forecast(steps=horizon)
                    horizon_forecasts['ETS'] = ets_forecast
                except Exception as e:
                    logger.error(f"ETS forecast failed: {e}")
            
            # Prophet forecast
            if 'prophet' in models and models['prophet'] is not None:
                try:
                    future_df = models['prophet'].make_future_dataframe(periods=horizon)
                    prophet_forecast = models['prophet'].predict(future_df)
                    horizon_forecasts['Prophet'] = prophet_forecast['yhat'].iloc[-horizon:].values
                except Exception as e:
                    logger.error(f"Prophet forecast failed: {e}")
            
            # ML models forecast
            if 'ml_models' in models:
                try:
                    ml_forecasts = self.generate_ml_forecasts(models['ml_models'], df, target_col, horizon)
                    horizon_forecasts.update(ml_forecasts)
                except Exception as e:
                    logger.error(f"ML forecast failed: {e}")
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame(index=future_dates)
            for model_name, forecast_values in horizon_forecasts.items():
                forecast_df[model_name] = forecast_values
            
            # Calculate ensemble forecast (simple average)
            forecast_df['Ensemble'] = forecast_df.mean(axis=1)
            
            forecasts[f'{horizon}_days'] = forecast_df
            
            logger.info(f"Generated {len(horizon_forecasts)} model forecasts for {horizon}-day horizon")
        
        return forecasts
    
    def generate_ml_forecasts(self, ml_models: Dict[str, Any], df: pd.DataFrame, 
                            target_col: str, horizon: int) -> Dict[str, np.ndarray]:
        """Generate forecasts using ML models"""
        ml_forecasts = {}
        feature_cols = ml_models['feature_cols']
        
        # Start with the last known data
        current_df = df.copy()
        predictions = {model_name: [] for model_name in ['random_forest', 'gradient_boosting', 'linear_regression']}
        
        for step in range(horizon):
            # Create features for current step
            try:
                # Get the latest features
                latest_features = current_df[feature_cols].iloc[-1:].values
                
                # Make predictions
                for model_name in predictions.keys():
                    if model_name in ml_models:
                        pred = ml_models[model_name].predict(latest_features)[0]
                        predictions[model_name].append(pred)
                
                # Update the dataframe with the ensemble prediction for next step
                ensemble_pred = np.mean([predictions[m][-1] for m in predictions.keys()])
                new_row = current_df.iloc[-1:].copy()
                new_row.index = [current_df.index[-1] + timedelta(days=1)]
                new_row[target_col] = ensemble_pred
                
                # Update lag features
                for lag in range(1, 31):  # Assuming max 30 lags
                    lag_col = f'{target_col}_lag_{lag}'
                    if lag_col in new_row.columns:
                        if lag == 1:
                            new_row[lag_col] = current_df[target_col].iloc[-1]
                        else:
                            new_row[lag_col] = current_df[f'{target_col}_lag_{lag-1}'].iloc[-1]
                
                current_df = pd.concat([current_df, new_row])
                
            except Exception as e:
                logger.warning(f"ML forecast step {step} failed: {e}")
                # Use last prediction or mean
                for model_name in predictions.keys():
                    last_pred = predictions[model_name][-1] if predictions[model_name] else df[target_col].mean()
                    predictions[model_name].append(last_pred)
        
        # Convert to proper format
        for model_name, preds in predictions.items():
            ml_forecasts[f'ML_{model_name}'] = np.array(preds)
        
        return ml_forecasts
    
    def validate_models(self, df: pd.DataFrame, target_col: str, test_size: int = 90) -> Dict[str, Dict[str, float]]:
        """Validate models using time-aware cross-validation"""
        logger.info("Validating models using time-aware cross-validation")
        
        train_df, test_df = self.split_data_time_aware(df, test_size)
        
        # Prepare data
        train_df, train_series = self.prepare_data_for_modeling(train_df, target_col)
        
        # Fit all models
        models = {}
        
        # Statistical models
        try:
            models['arima'] = self.fit_arima_model(train_series)
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
        
        try:
            models['sarima'] = self.fit_sarima_model(train_series)
        except Exception as e:
            logger.error(f"SARIMA fitting failed: {e}")
        
        try:
            models['ets'] = self.fit_ets_model(train_series)
        except Exception as e:
            logger.error(f"ETS fitting failed: {e}")
        
        # Prophet
        try:
            models['prophet'] = self.fit_prophet_model(train_df, target_col)
        except Exception as e:
            logger.error(f"Prophet fitting failed: {e}")
        
        # ML models
        try:
            models['ml_models'] = self.fit_ml_models(train_df, target_col)
        except Exception as e:
            logger.error(f"ML models fitting failed: {e}")
        
        # Generate forecasts
        forecasts = self.generate_forecasts(models, train_df, target_col, [test_size])
        
        # Calculate validation metrics
        validation_results = {}
        actual_values = test_df[target_col].values
        
        if f'{test_size}_days' in forecasts:
            forecast_df = forecasts[f'{test_size}_days']
            
            for model_name in forecast_df.columns:
                try:
                    predicted_values = forecast_df[model_name].values
                    
                    # Ensure same length
                    min_length = min(len(actual_values), len(predicted_values))
                    actual_trimmed = actual_values[:min_length]
                    predicted_trimmed = predicted_values[:min_length]
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(actual_trimmed, predicted_trimmed))
                    mae = mean_absolute_error(actual_trimmed, predicted_trimmed)
                    mape = mean_absolute_percentage_error(actual_trimmed, predicted_trimmed) * 100
                    
                    validation_results[model_name] = {
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape
                    }
                    
                    logger.info(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Validation failed for {model_name}: {e}")
        
        self.models = models
        self.validation_results = validation_results
        
        return validation_results
    
    def run_comprehensive_forecast(self, target_col: str = 'Occupancy_Pct', 
                                  horizons: List[int] = [90, 180, 365]) -> Dict[str, Any]:
        """Run comprehensive forecasting pipeline"""
        logger.info(f"Starting comprehensive forecast for {target_col}")
        
        # Load data
        df = self.load_data_from_sql()
        
        # Validate models
        validation_results = self.validate_models(df, target_col)
        
        # Generate final forecasts using full dataset
        df_prepared, series_prepared = self.prepare_data_for_modeling(df, target_col)
        forecasts = self.generate_forecasts(self.models, df_prepared, target_col, horizons)
        
        # Store results
        self.forecasts = forecasts
        
        return {
            'forecasts': forecasts,
            'validation_results': validation_results,
            'models': self.models,
            'data_info': {
                'data_shape': df.shape,
                'date_range': (df.index.min(), df.index.max()),
                'target_column': target_col
            }
        }
    
    def save_forecasts_to_csv(self, output_dir: str):
        """Save forecasts to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for horizon, forecast_df in self.forecasts.items():
            filename = f"forecast_{horizon}_{timestamp}.csv"
            file_path = output_path / filename
            forecast_df.to_csv(file_path)
            logger.info(f"Saved {horizon} forecast to {file_path}")
    
    def plot_forecasts(self, target_col: str, save_plots: bool = True, output_dir: str = None):
        """Plot forecast results"""
        if not self.forecasts:
            logger.error("No forecasts available to plot")
            return
        
        # Load historical data for plotting
        df = self.load_data_from_sql()
        
        fig, axes = plt.subplots(len(self.forecasts), 1, figsize=(15, 5*len(self.forecasts)))
        if len(self.forecasts) == 1:
            axes = [axes]
        
        for i, (horizon, forecast_df) in enumerate(self.forecasts.items()):
            ax = axes[i]
            
            # Plot historical data (last 365 days)
            historical_data = df[target_col].iloc[-365:]
            ax.plot(historical_data.index, historical_data.values, label='Historical', color='blue', alpha=0.7)
            
            # Plot forecasts
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for j, model_name in enumerate(forecast_df.columns):
                color = colors[j % len(colors)]
                ax.plot(forecast_df.index, forecast_df[model_name], 
                       label=f'{model_name} Forecast', color=color, alpha=0.8)
            
            ax.set_title(f'{target_col} Forecast - {horizon}')
            ax.set_ylabel(target_col)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots and output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = Path(output_dir) / f"forecasts_plot_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved forecast plot to {plot_path}")
        
        plt.show()


def main():
    """Main execution function"""
    # Configuration
    base_dir = "/home/gee_devops254/Downloads/Revenue Architecture"
    db_path = os.path.join(base_dir, "db", "revenue.db")
    output_dir = os.path.join(base_dir, "forecasts")
    
    # Create forecasting engine
    engine = HotelForecastingEngine(db_path)
    
    # Run comprehensive forecast for Occupancy
    logger.info("Running comprehensive forecasting for Occupancy %")
    results = engine.run_comprehensive_forecast(
        target_col='Occupancy_Pct',
        horizons=[90, 180, 365]  # 3, 6, 12 months
    )
    
    # Save results
    engine.save_forecasts_to_csv(output_dir)
    
    # Print validation results
    print("\n📊 Model Validation Results (Occupancy %):")
    print("=" * 60)
    for model_name, metrics in results['validation_results'].items():
        print(f"{model_name:15} | RMSE: {metrics['RMSE']:6.2f} | MAE: {metrics['MAE']:6.2f} | MAPE: {metrics['MAPE']:5.1f}%")
    
    # Print forecast summary
    print(f"\n🔮 Generated forecasts for {len(results['forecasts'])} horizons:")
    for horizon, forecast_df in results['forecasts'].items():
        print(f"  {horizon}: {len(forecast_df)} days, {len(forecast_df.columns)} models")
        print(f"    Best model prediction: {forecast_df['Ensemble'].mean():.1f}% avg occupancy")
    
    return results


if __name__ == "__main__":
    import os
    main()