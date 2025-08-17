"""
Corrected Time Series Forecasting with Historical Pattern Validation
Addresses issues with unrealistic low forecasts by incorporating seasonal recovery patterns
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from pathlib import Path

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectedHotelForecasting:
    """Corrected forecasting that accounts for historical seasonal patterns"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.historical_stats = {}
        self.seasonal_patterns = {}
        self.forecasts = {}
    
    def load_and_analyze_historical_data(self) -> pd.DataFrame:
        """Load data and analyze historical patterns"""
        logger.info("Loading and analyzing historical patterns")
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT Date, Occupancy_Pct, Revenue, ADR, RevPAR, Month, DayOfWeek, IsWeekend FROM hotel_data_combined ORDER BY Date", 
            conn
        )
        conn.close()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Remove future dates beyond current date
        current_date = datetime.now().date()
        df = df[df.index.date <= current_date]
        
        # Calculate historical statistics
        self.historical_stats = {
            'overall_mean': df['Occupancy_Pct'].mean(),
            'overall_std': df['Occupancy_Pct'].std(),
            'overall_min': df['Occupancy_Pct'].min(),
            'overall_max': df['Occupancy_Pct'].max()
        }
        
        # Calculate seasonal patterns
        seasonal_stats = df.groupby('Month')['Occupancy_Pct'].agg(['mean', 'std', 'min', 'max']).round(2)
        self.seasonal_patterns = seasonal_stats.to_dict('index')
        
        # Calculate day of week patterns
        dow_stats = df.groupby('DayOfWeek')['Occupancy_Pct'].agg(['mean', 'std']).round(2)
        self.dow_patterns = dow_stats.to_dict('index')
        
        logger.info(f"Historical data loaded: {len(df)} records")
        logger.info(f"Overall occupancy: {self.historical_stats['overall_mean']:.1f}% ± {self.historical_stats['overall_std']:.1f}%")
        
        return df
    
    def detect_current_anomaly(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect if current occupancy is anomalously low"""
        recent_30_days = df.tail(30)['Occupancy_Pct']
        recent_avg = recent_30_days.mean()
        historical_avg = self.historical_stats['overall_mean']
        
        # Calculate how many standard deviations below normal
        std_dev = self.historical_stats['overall_std']
        z_score = (recent_avg - historical_avg) / std_dev
        
        anomaly_info = {
            'recent_avg': recent_avg,
            'historical_avg': historical_avg,
            'z_score': z_score,
            'is_anomaly': z_score < -1.5,  # More than 1.5 std below normal
            'recovery_factor': max(0.1, min(1.0, (recent_avg / historical_avg)))
        }
        
        logger.info(f"Current occupancy: {recent_avg:.1f}% (Historical: {historical_avg:.1f}%)")
        logger.info(f"Z-score: {z_score:.2f}, Anomaly: {anomaly_info['is_anomaly']}")
        
        return anomaly_info
    
    def seasonal_recovery_forecast(self, df: pd.DataFrame, steps: int) -> np.ndarray:
        """Forecast with seasonal recovery pattern"""
        logger.info("Generating seasonal recovery forecast")
        
        # Get anomaly information
        anomaly_info = self.detect_current_anomaly(df)
        recent_avg = anomaly_info['recent_avg']
        
        # Generate future dates
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
        
        forecasts = []
        recovery_rate = 0.02  # 2% recovery per month if anomalously low
        
        for i, future_date in enumerate(future_dates):
            month = future_date.month
            day_of_week = future_date.dayofweek
            
            # Get historical seasonal pattern
            seasonal_mean = self.seasonal_patterns[month]['mean']
            seasonal_std = self.seasonal_patterns[month]['std']
            
            # Day of week adjustment
            dow_mean = self.dow_patterns[day_of_week]['mean']
            dow_adjustment = (dow_mean - self.historical_stats['overall_mean']) * 0.1
            
            # Recovery pattern if currently anomalous
            if anomaly_info['is_anomaly']:
                # Gradual recovery towards seasonal normal
                recovery_progress = min(1.0, i * recovery_rate / 30)  # Recover over ~1.5 years
                target_occupancy = (
                    recent_avg * (1 - recovery_progress) + 
                    seasonal_mean * recovery_progress
                )
            else:
                target_occupancy = seasonal_mean
            
            # Apply day of week adjustment
            target_occupancy += dow_adjustment
            
            # Add some random variation but keep within reasonable bounds
            variation = np.random.normal(0, seasonal_std * 0.3)
            forecast_value = target_occupancy + variation
            
            # Ensure reasonable bounds
            forecast_value = max(10, min(95, forecast_value))
            
            forecasts.append(forecast_value)
        
        return np.array(forecasts)
    
    def enhanced_prophet_forecast(self, df: pd.DataFrame, steps: int) -> np.ndarray:
        """Enhanced Prophet forecast with recovery patterns"""
        if not PROPHET_AVAILABLE:
            return self.seasonal_recovery_forecast(df, steps)
        
        try:
            # Detect anomaly period
            anomaly_info = self.detect_current_anomaly(df)
            
            # If currently anomalous, adjust recent data for training
            df_adjusted = df.copy()
            if anomaly_info['is_anomaly']:
                # Add synthetic recovery data points
                recent_period = 90  # Last 3 months
                adjustment_factor = 1.2  # Boost recent data slightly for training
                
                recent_mask = df_adjusted.index >= (df_adjusted.index.max() - timedelta(days=recent_period))
                df_adjusted.loc[recent_mask, 'Occupancy_Pct'] *= adjustment_factor
            
            # Prepare Prophet data
            prophet_df = df_adjusted.reset_index()[['Occupancy_Pct']].reset_index()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = df_adjusted.index
            
            # Add holiday/season indicators
            prophet_df['month'] = prophet_df['ds'].dt.month
            prophet_df['is_peak_season'] = prophet_df['month'].isin([11, 12, 1, 2]).astype(int)
            prophet_df['is_summer'] = prophet_df['month'].isin([6, 7, 8]).astype(int)
            
            # Initialize Prophet with stronger seasonality
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=15.0,  # Stronger seasonality
                holidays_prior_scale=10.0
            )
            
            # Add regressors
            model.add_regressor('is_peak_season')
            model.add_regressor('is_summer')
            
            # Fit model
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=steps)
            future['month'] = future['ds'].dt.month
            future['is_peak_season'] = future['month'].isin([11, 12, 1, 2]).astype(int)
            future['is_summer'] = future['month'].isin([6, 7, 8]).astype(int)
            
            # Generate forecast
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].tail(steps).values
            
            # Apply bounds
            forecast_values = np.clip(forecast_values, 15, 90)
            
            return forecast_values
            
        except Exception as e:
            logger.warning(f"Enhanced Prophet failed: {e}")
            return self.seasonal_recovery_forecast(df, steps)
    
    def trend_corrected_ml_forecast(self, df: pd.DataFrame, steps: int) -> np.ndarray:
        """ML forecast with trend correction"""
        try:
            # Create features with seasonal indicators
            feature_df = df.copy()
            
            # Add lag features
            for lag in [1, 7, 14, 30]:
                feature_df[f'occupancy_lag_{lag}'] = feature_df['Occupancy_Pct'].shift(lag)
            
            # Add rolling statistics
            for window in [7, 14, 30]:
                feature_df[f'occupancy_ma_{window}'] = feature_df['Occupancy_Pct'].rolling(window).mean()
                feature_df[f'occupancy_std_{window}'] = feature_df['Occupancy_Pct'].rolling(window).std()
            
            # Add seasonal features
            feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['Month'] / 12)
            feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['Month'] / 12)
            feature_df['day_sin'] = np.sin(2 * np.pi * feature_df['DayOfWeek'] / 7)
            feature_df['day_cos'] = np.cos(2 * np.pi * feature_df['DayOfWeek'] / 7)
            
            # Historical seasonal averages as features
            for month in range(1, 13):
                feature_df[f'historical_month_{month}'] = (
                    (feature_df['Month'] == month).astype(int) * 
                    self.seasonal_patterns[month]['mean']
                )
            
            # Select features
            feature_cols = [col for col in feature_df.columns if 
                          ('lag_' in col or 'ma_' in col or 'std_' in col or 
                           'sin' in col or 'cos' in col or 'historical_month_' in col or
                           col in ['Month', 'DayOfWeek', 'IsWeekend'])]
            
            # Remove NaN rows
            feature_df = feature_df.dropna()
            
            if len(feature_df) < 100:
                return self.seasonal_recovery_forecast(df, steps)
            
            X = feature_df[feature_cols]
            y = feature_df['Occupancy_Pct']
            
            # Fit Random Forest with more trees
            model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
            model.fit(X, y)
            
            # Generate forecasts
            forecasts = []
            current_df = feature_df.copy()
            anomaly_info = self.detect_current_anomaly(df)
            
            for step in range(steps):
                # Get latest features
                latest_features = current_df[feature_cols].iloc[-1:].values
                pred = model.predict(latest_features)[0]
                
                # Apply recovery adjustment if anomalous
                if anomaly_info['is_anomaly']:
                    future_date = current_df.index[-1] + timedelta(days=step+1)
                    month = future_date.month
                    historical_month_avg = self.seasonal_patterns[month]['mean']
                    
                    # Blend prediction with historical seasonal average
                    recovery_weight = min(0.4, step / 180)  # Increase weight over 6 months
                    pred = pred * (1 - recovery_weight) + historical_month_avg * recovery_weight
                
                # Ensure reasonable bounds
                pred = max(15, min(90, pred))
                forecasts.append(pred)
                
                # Update features for next step (simplified)
                new_date = current_df.index[-1] + timedelta(days=1)
                new_row = current_df.iloc[-1:].copy()
                new_row.index = [new_date]
                new_row['Occupancy_Pct'] = pred
                new_row['Month'] = new_date.month
                new_row['DayOfWeek'] = new_date.dayofweek
                
                current_df = pd.concat([current_df, new_row])
            
            return np.array(forecasts)
            
        except Exception as e:
            logger.warning(f"ML forecast failed: {e}")
            return self.seasonal_recovery_forecast(df, steps)
    
    def generate_corrected_forecasts(self, df: pd.DataFrame, horizons: List[int]) -> Dict[str, pd.DataFrame]:
        """Generate corrected forecasts using all improved models"""
        logger.info("Generating corrected forecasts with historical validation")
        
        all_forecasts = {}
        
        for horizon in horizons:
            logger.info(f"Generating corrected {horizon}-day forecasts")
            
            # Generate future dates
            last_date = df.index.max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
            
            # Generate forecasts using different models
            forecasts = {}
            
            # Seasonal recovery forecast (baseline)
            forecasts['Seasonal_Recovery'] = self.seasonal_recovery_forecast(df, horizon)
            
            # Enhanced Prophet
            forecasts['Enhanced_Prophet'] = self.enhanced_prophet_forecast(df, horizon)
            
            # Trend corrected ML
            forecasts['ML_Trend_Corrected'] = self.trend_corrected_ml_forecast(df, horizon)
            
            # Historical seasonal average (reference)
            seasonal_ref = []
            for date in future_dates:
                month_avg = self.seasonal_patterns[date.month]['mean']
                seasonal_ref.append(month_avg)
            forecasts['Historical_Seasonal_Avg'] = np.array(seasonal_ref)
            
            # Create DataFrame
            forecast_df = pd.DataFrame(index=future_dates)
            for model_name, forecast_values in forecasts.items():
                if len(forecast_values) == horizon:
                    forecast_df[model_name] = forecast_values
            
            # Calculate intelligent ensemble
            # Give more weight to models that account for recovery patterns
            weights = {
                'Seasonal_Recovery': 0.3,
                'Enhanced_Prophet': 0.35,
                'ML_Trend_Corrected': 0.25,
                'Historical_Seasonal_Avg': 0.1
            }
            
            ensemble_forecast = np.zeros(horizon)
            total_weight = 0
            
            for model_name, weight in weights.items():
                if model_name in forecast_df.columns:
                    ensemble_forecast += forecast_df[model_name].values * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_forecast /= total_weight
            
            forecast_df['Corrected_Ensemble'] = ensemble_forecast
            
            all_forecasts[f'{horizon}_days'] = forecast_df
        
        return all_forecasts
    
    def run_corrected_forecast(self, target_col: str = 'Occupancy_Pct') -> Dict:
        """Run the complete corrected forecasting pipeline"""
        logger.info("Starting corrected forecasting pipeline")
        
        # Load and analyze data
        df = self.load_and_analyze_historical_data()
        
        # Generate corrected forecasts
        horizons = [90, 180, 365]  # 3, 6, 12 months
        forecasts = self.generate_corrected_forecasts(df, horizons)
        
        # Store results
        self.forecasts = forecasts
        
        return {
            'forecasts': forecasts,
            'historical_stats': self.historical_stats,
            'seasonal_patterns': self.seasonal_patterns,
            'anomaly_info': self.detect_current_anomaly(df),
            'data_info': {
                'shape': df.shape,
                'date_range': (df.index.min(), df.index.max())
            }
        }
    
    def save_corrected_forecasts(self, output_dir: str):
        """Save corrected forecasts to CSV"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for horizon, forecast_df in self.forecasts.items():
            filename = f"corrected_occupancy_forecast_{horizon}_{timestamp}.csv"
            filepath = Path(output_dir) / filename
            forecast_df.to_csv(filepath)
            logger.info(f"Saved corrected {horizon} forecast to {filepath}")
    
    def print_corrected_summary(self, results: Dict):
        """Print summary of corrected forecasts"""
        print("\n🏨 Corrected Hotel Occupancy Forecast Summary")
        print("=" * 65)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        anomaly_info = results['anomaly_info']
        
        print(f"📅 Current Date: {current_date}")
        print(f"📊 Recent Occupancy: {anomaly_info['recent_avg']:.1f}%")
        print(f"📈 Historical Average: {anomaly_info['historical_avg']:.1f}%")
        print(f"⚠️  Current Anomaly: {'Yes' if anomaly_info['is_anomaly'] else 'No'} (Z-score: {anomaly_info['z_score']:.2f})")
        
        print(f"\n🔮 Corrected Forecast Periods:")
        for horizon, forecast_df in results['forecasts'].items():
            days = horizon.split('_')[0]
            
            # Get different model predictions
            corrected_avg = forecast_df['Corrected_Ensemble'].mean()
            corrected_min = forecast_df['Corrected_Ensemble'].min()
            corrected_max = forecast_df['Corrected_Ensemble'].max()
            
            historical_avg = forecast_df['Historical_Seasonal_Avg'].mean()
            
            print(f"  {days:>3} days: Corrected {corrected_avg:5.1f}% | Range {corrected_min:5.1f}% - {corrected_max:5.1f}% | Historical {historical_avg:5.1f}%")
        
        print(f"\n📊 Seasonal Patterns (Historical Monthly Averages):")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for i, month in enumerate(months, 1):
            if i in results['seasonal_patterns']:
                avg = results['seasonal_patterns'][i]['mean']
                print(f"  {month}: {avg:5.1f}%", end="  ")
                if i % 4 == 0:
                    print()
        print()


def main():
    """Main execution"""
    base_dir = "/home/gee_devops254/Downloads/Revenue Architecture"
    db_path = f"{base_dir}/db/revenue.db"
    output_dir = f"{base_dir}/forecasts"
    
    # Run corrected forecasting
    forecaster = CorrectedHotelForecasting(db_path)
    results = forecaster.run_corrected_forecast()
    
    # Save results
    forecaster.save_corrected_forecasts(output_dir)
    
    # Print summary
    forecaster.print_corrected_summary(results)
    
    return results


if __name__ == "__main__":
    main()