"""
Advanced Forecasting Module for Revenue Analytics
Implements 3-month forecasting, year-end predictions, and machine learning analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import logging
from pathlib import Path

# Machine Learning imports
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Machine learning libraries not available")

logger = logging.getLogger(__name__)

class AdvancedForecastor:
    """Advanced forecasting with budget weighting and ML analysis"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # Revenue targets
        self.current_year_budget = 38_000_000  # AED 38 million budget
        self.last_year_total = 33_000_000      # AED 33 million last year
        
    def get_next_three_months(self, current_date: datetime = None) -> List[Tuple[datetime, str]]:
        """Get the next 3 months from current date"""
        if current_date is None:
            current_date = datetime.now()
        
        months = []
        for i in range(3):
            if current_date.month + i > 12:
                next_month = datetime(current_date.year + 1, (current_date.month + i) % 12, 1)
            else:
                next_month = datetime(current_date.year, current_date.month + i, 1)
            
            month_name = next_month.strftime('%B')
            months.append((next_month, month_name))
        
        return months
    
    def forecast_three_months_weighted(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame,
                                      current_budget_weight: float = 0.7, current_historical_weight: float = 0.3,
                                      future_budget_weight: float = 0.4, future_historical_weight: float = 0.6) -> pd.DataFrame:
        """
        Forecast next 3 months with customizable budget weighting
        Allows adjustment of weights between budget and historical data
        """
        try:
            current_date = datetime.now()
            next_months = self.get_next_three_months(current_date)
            
            forecasts = []
            
            for month_date, month_name in next_months:
                month_forecast = self._forecast_single_month_weighted(
                    segment_df, occupancy_df, month_date, month_name,
                    current_budget_weight, current_historical_weight,
                    future_budget_weight, future_historical_weight
                )
                forecasts.append(month_forecast)
            
            if forecasts:
                result = pd.concat(forecasts, ignore_index=True)
                logger.info(f"Generated 3-month weighted forecast for {len(next_months)} months")
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in 3-month weighted forecasting: {e}")
            return pd.DataFrame()
    
    def _forecast_single_month_weighted(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame, 
                                      month_date: datetime, month_name: str,
                                      current_budget_weight: float, current_historical_weight: float,
                                      future_budget_weight: float, future_historical_weight: float) -> pd.DataFrame:
        """Forecast single month with customizable weighted approach"""
        try:
            # Get current month data
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            # Use provided weights based on month
            if month_date.month == current_month:
                budget_weight = current_budget_weight
                historical_weight = current_historical_weight
            else:
                budget_weight = future_budget_weight
                historical_weight = future_historical_weight
            
            # Get budget data for the month
            budget_revenue = self._get_monthly_budget(segment_df, month_date)
            
            # Get historical data
            historical_revenue = self._get_historical_monthly_revenue(segment_df, month_date)
            
            # Calculate weighted forecast
            if historical_revenue > 0:
                weighted_revenue = (budget_weight * budget_revenue + 
                                  historical_weight * historical_revenue)
            else:
                weighted_revenue = budget_revenue
            
            # Get segment breakdown
            segments_forecast = self._forecast_segments_for_month(segment_df, month_date, weighted_revenue)
            
            # Calculate occupancy forecast
            occupancy_forecast = self._forecast_occupancy_for_month(occupancy_df, month_date)
            
            result = pd.DataFrame({
                'month': [month_name],
                'month_date': [month_date],
                'total_revenue_forecast': [weighted_revenue],
                'budget_weight': [budget_weight],
                'historical_weight': [historical_weight],
                'budget_component': [budget_revenue * budget_weight],
                'historical_component': [historical_revenue * historical_weight],
                'avg_occupancy_forecast': [occupancy_forecast]
            })
            
            # Add segment details
            for _, segment_row in segments_forecast.iterrows():
                result[f"{segment_row['segment']}_revenue"] = [segment_row['forecast_revenue']]
            
            return result
            
        except Exception as e:
            logger.error(f"Error forecasting month {month_name}: {e}")
            return pd.DataFrame()
    
    def _get_monthly_budget(self, segment_df: pd.DataFrame, month_date: datetime) -> float:
        """Get budget allocation for a specific month from actual data"""
        try:
            # Find the closest available month in the data for similar budget pattern
            month_str = month_date.strftime('%Y-%m-01')
            
            # First try to find exact month match
            month_data = segment_df[segment_df['Month'] == month_str]
            if not month_data.empty and 'Budget_This_Year_Revenue' in segment_df.columns:
                monthly_budget = month_data['Budget_This_Year_Revenue'].sum()
                if monthly_budget > 0:
                    return monthly_budget
            
            # If no exact match, use the same month number from available data
            target_month = month_date.month
            available_months = segment_df['Month'].unique()
            
            for available_month in available_months:
                try:
                    available_date = pd.to_datetime(available_month)
                    if available_date.month == target_month:
                        month_data = segment_df[segment_df['Month'] == available_month]
                        if not month_data.empty:
                            monthly_budget = month_data['Budget_This_Year_Revenue'].sum()
                            if monthly_budget > 0:
                                return monthly_budget
                except:
                    continue
            
            # Fallback: use average budget from all available months
            if 'Budget_This_Year_Revenue' in segment_df.columns:
                all_months = []
                for month in segment_df['Month'].unique():
                    month_data = segment_df[segment_df['Month'] == month]
                    month_total = month_data['Budget_This_Year_Revenue'].sum()
                    if month_total > 0:
                        all_months.append(month_total)
                
                if all_months:
                    return np.mean(all_months)
            
            # Final fallback: use target budget divided by 12
            return self.current_year_budget / 12
            
        except Exception as e:
            logger.error(f"Error getting monthly budget: {e}")
            return self.current_year_budget / 12
    
    def _get_historical_monthly_revenue(self, segment_df: pd.DataFrame, month_date: datetime) -> float:
        """Get historical revenue for the same month using Full_Month_Last_Year_Revenue"""
        try:
            # Find the same month number from available data
            target_month = month_date.month
            available_months = segment_df['Month'].unique()
            
            for available_month in available_months:
                try:
                    available_date = pd.to_datetime(available_month)
                    if available_date.month == target_month:
                        month_data = segment_df[segment_df['Month'] == available_month]
                        if not month_data.empty and 'Full_Month_Last_Year_Revenue' in segment_df.columns:
                            monthly_ly_revenue = month_data['Full_Month_Last_Year_Revenue'].sum()
                            if monthly_ly_revenue > 0:
                                return monthly_ly_revenue
                except:
                    continue
            
            # If no exact month match, use average from all available months
            if 'Full_Month_Last_Year_Revenue' in segment_df.columns:
                all_months = []
                for month in segment_df['Month'].unique():
                    month_data = segment_df[segment_df['Month'] == month]
                    month_total = month_data['Full_Month_Last_Year_Revenue'].sum()
                    if month_total > 0:
                        all_months.append(month_total)
                
                if all_months:
                    return np.mean(all_months)
            
            # Fallback: use estimated monthly from last year total
            return self.last_year_total / 12
            
        except Exception as e:
            logger.error(f"Error getting historical revenue: {e}")
            return self.last_year_total / 12
    
    def _forecast_segments_for_month(self, segment_df: pd.DataFrame, month_date: datetime, 
                                   total_revenue: float) -> pd.DataFrame:
        """Forecast segment breakdown for a month"""
        try:
            # Get historical segment proportions
            if 'MergedSegment' in segment_df.columns and 'Business_on_the_Books_Revenue' in segment_df.columns:
                segment_totals = segment_df.groupby('MergedSegment')['Business_on_the_Books_Revenue'].sum()
                total_historical = segment_totals.sum()
                
                if total_historical > 0:
                    segment_proportions = segment_totals / total_historical
                    
                    # Apply proportions to forecasted total
                    forecasts = []
                    for segment, proportion in segment_proportions.items():
                        forecast_revenue = total_revenue * proportion
                        forecasts.append({
                            'segment': segment,
                            'proportion': proportion,
                            'forecast_revenue': forecast_revenue
                        })
                    
                    return pd.DataFrame(forecasts)
            
            # Fallback: equal distribution among available segments
            segments = segment_df['MergedSegment'].unique() if 'MergedSegment' in segment_df.columns else ['Unknown']
            equal_share = total_revenue / len(segments)
            
            return pd.DataFrame([
                {'segment': segment, 'proportion': 1/len(segments), 'forecast_revenue': equal_share}
                for segment in segments
            ])
            
        except Exception as e:
            logger.error(f"Error forecasting segments: {e}")
            return pd.DataFrame()
    
    def _forecast_occupancy_for_month(self, occupancy_df: pd.DataFrame, month_date: datetime) -> float:
        """Forecast average occupancy for a month"""
        try:
            if 'Date' in occupancy_df.columns and 'Occ%' in occupancy_df.columns:
                occupancy_df['Date'] = pd.to_datetime(occupancy_df['Date'])
                
                # Find same month from historical data
                target_month = month_date.month
                historical_occ = occupancy_df[occupancy_df['Date'].dt.month == target_month]
                
                if not historical_occ.empty:
                    return historical_occ['Occ%'].mean()
                
                # Fallback to overall average
                return occupancy_df['Occ%'].mean()
            
            return 75.0  # Default assumption
            
        except Exception as e:
            logger.error(f"Error forecasting occupancy: {e}")
            return 75.0
    
    def predict_year_end_revenue(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame) -> Dict:
        """
        Predict how much revenue we'll close the year with
        Compare against 38M budget and 33M last year
        """
        try:
            current_date = datetime.now()
            current_month = current_date.month
            
            # Get year-to-date revenue
            current_year = current_date.year
            ytd_revenue = self._get_year_to_date_revenue(segment_df, current_year)
            
            # Get remaining months forecast
            remaining_months = 12 - current_month
            if remaining_months > 0:
                remaining_forecast = self._forecast_remaining_year(segment_df, occupancy_df, remaining_months)
            else:
                remaining_forecast = 0
            
            # Calculate total year-end prediction
            year_end_prediction = ytd_revenue + remaining_forecast
            
            # Calculate performance metrics
            budget_variance = year_end_prediction - self.current_year_budget
            budget_performance = (year_end_prediction / self.current_year_budget) * 100
            
            vs_last_year = year_end_prediction - self.last_year_total
            vs_last_year_growth = ((year_end_prediction - self.last_year_total) / self.last_year_total) * 100
            
            return {
                'year_end_prediction': year_end_prediction,
                'ytd_revenue': ytd_revenue,
                'remaining_forecast': remaining_forecast,
                'budget_target': self.current_year_budget,
                'budget_variance': budget_variance,
                'budget_performance_pct': budget_performance,
                'last_year_total': self.last_year_total,
                'vs_last_year': vs_last_year,
                'vs_last_year_growth_pct': vs_last_year_growth,
                'months_remaining': remaining_months,
                'current_month': current_month
            }
            
        except Exception as e:
            logger.error(f"Error predicting year-end revenue: {e}")
            return {}
    
    def _get_year_to_date_revenue(self, segment_df: pd.DataFrame, year: int) -> float:
        """Get year-to-date revenue from Business on the Books data"""
        try:
            current_month = datetime.now().month
            ytd_revenue = 0
            
            # Sum up all completed months in the data using Business_on_the_Books_Revenue
            available_months = segment_df['Month'].unique()
            
            for month_str in available_months:
                try:
                    month_date = pd.to_datetime(month_str)
                    # Only include months up to current month
                    if month_date.year == year and month_date.month <= current_month:
                        month_data = segment_df[segment_df['Month'] == month_str]
                        if not month_data.empty and 'Business_on_the_Books_Revenue' in segment_df.columns:
                            month_revenue = month_data['Business_on_the_Books_Revenue'].sum()
                            ytd_revenue += month_revenue
                except:
                    continue
            
            # If we have YTD data, return it
            if ytd_revenue > 0:
                return ytd_revenue
            
            # Fallback: estimate based on current month
            estimated_monthly = self.current_year_budget / 12
            return estimated_monthly * current_month * 0.6  # Conservative YTD estimate
            
        except Exception as e:
            logger.error(f"Error getting YTD revenue: {e}")
            return 0
    
    def _forecast_remaining_year(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame, 
                               remaining_months: int) -> float:
        """Forecast revenue for remaining months of the year"""
        try:
            current_month = datetime.now().month
            
            # Calculate monthly forecast based on budget and historical data
            total_forecast = 0
            
            for month_offset in range(remaining_months):
                future_month = current_month + month_offset + 1
                if future_month > 12:
                    future_month = future_month - 12
                
                # Get budget for this month
                monthly_budget = self._get_monthly_budget(segment_df, datetime(2025, future_month, 1))
                
                # Get historical performance
                historical_revenue = self._get_historical_monthly_revenue(segment_df, datetime(2025, future_month, 1))
                
                # Weight: 60% historical, 40% budget for future months
                if historical_revenue > 0:
                    monthly_forecast = 0.6 * historical_revenue + 0.4 * monthly_budget
                else:
                    monthly_forecast = monthly_budget
                
                total_forecast += monthly_forecast
            
            return total_forecast
            
        except Exception as e:
            logger.error(f"Error forecasting remaining year: {e}")
            return (self.current_year_budget / 12) * remaining_months
    
    def predict_current_month_close(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame) -> Dict:
        """
        Predict current month close using BOB, MTD pickup, last year remainder, and budget weighting
        """
        try:
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            current_day = current_date.day
            
            # Get current month segment data
            month_str = f"{current_year}-{current_month:02d}-01"
            month_data = segment_df[segment_df['Month'] == month_str]
            
            if month_data.empty:
                return {'error': 'No current month data available'}
            
            # Get key metrics
            business_on_books = month_data['Business_on_the_Books_Revenue'].sum()
            mtd_pickup = month_data['Month_to_Date_Revenue'].sum() if 'Month_to_Date_Revenue' in month_data.columns else 0
            budget_month = month_data['Budget_This_Year_Revenue'].sum()
            last_year_full_month = month_data['Full_Month_Last_Year_Revenue'].sum()
            last_year_same_time = month_data['Business_on_the_Books_Same_Time_Last_Year_Revenue'].sum()
            
            # Calculate last year remainder (what they achieved in remaining days)
            last_year_remainder = last_year_full_month - last_year_same_time
            
            # Method 1: Current BOB + Last Year Remainder approach
            bob_plus_remainder = business_on_books + last_year_remainder
            
            # Method 2: Occupancy-based daily revenue prediction
            occupancy_prediction = self._predict_from_occupancy_data(occupancy_df, current_date)
            
            # Method 3: Budget-weighted prediction (smaller weight as requested)
            # 60% last year pattern, 30% BOB trend, 10% budget
            weighted_prediction = (
                0.6 * bob_plus_remainder + 
                0.3 * (business_on_books * (last_year_full_month / max(last_year_same_time, 1))) +
                0.1 * budget_month
            )
            
            # Final prediction (average of methods with more weight on BOB+remainder)
            final_prediction = (
                0.5 * bob_plus_remainder + 
                0.3 * weighted_prediction + 
                0.2 * occupancy_prediction
            )
            
            return {
                'business_on_books': business_on_books,
                'mtd_pickup': mtd_pickup,
                'budget_month': budget_month,
                'last_year_full_month': last_year_full_month,
                'last_year_same_time': last_year_same_time,
                'last_year_remainder': last_year_remainder,
                'bob_plus_remainder_prediction': bob_plus_remainder,
                'occupancy_prediction': occupancy_prediction,
                'weighted_prediction': weighted_prediction,
                'final_prediction': final_prediction,
                'current_day': current_day,
                'days_remaining': self._get_days_remaining_in_month(current_date)
            }
            
        except Exception as e:
            logger.error(f"Error predicting current month close: {e}")
            return {}
    
    def _get_current_month_data(self, segment_df: pd.DataFrame, year: int, month: int) -> float:
        """Get current month to date revenue"""
        try:
            # Find the current month's data
            month_str = f"{year}-{month:02d}-01"
            month_data = segment_df[segment_df['Month'] == month_str]
            
            if not month_data.empty:
                if 'Month_to_Date_Revenue' in segment_df.columns:
                    return month_data['Month_to_Date_Revenue'].sum()
                elif 'Business_on_the_Books_Revenue' in segment_df.columns:
                    return month_data['Business_on_the_Books_Revenue'].sum()
            
            # If exact month not found, try to find any current year month data
            current_year_data = segment_df[segment_df['Month'].str.startswith(str(year))]
            if not current_year_data.empty and 'Month_to_Date_Revenue' in segment_df.columns:
                # Use the most recent month's MTD data
                return current_year_data['Month_to_Date_Revenue'].sum() / len(current_year_data['Month'].unique())
            
            return 0
        except Exception as e:
            logger.error(f"Error getting current month data: {e}")
            return 0
    
    def _get_same_time_last_year(self, segment_df: pd.DataFrame, month: int, day: int) -> float:
        """Get revenue for same time last year"""
        try:
            # Find the same month from available data
            available_months = segment_df['Month'].unique()
            
            for available_month in available_months:
                try:
                    available_date = pd.to_datetime(available_month)
                    if available_date.month == month:
                        month_data = segment_df[segment_df['Month'] == available_month]
                        if not month_data.empty and 'Business_on_the_Books_Same_Time_Last_Year_Revenue' in segment_df.columns:
                            return month_data['Business_on_the_Books_Same_Time_Last_Year_Revenue'].sum()
                except:
                    continue
            
            return 0
        except Exception as e:
            logger.error(f"Error getting same time last year: {e}")
            return 0
    
    def _get_full_month_last_year(self, segment_df: pd.DataFrame, month: int) -> float:
        """Get full month revenue from last year"""
        try:
            # Find the same month from available data
            available_months = segment_df['Month'].unique()
            
            for available_month in available_months:
                try:
                    available_date = pd.to_datetime(available_month)
                    if available_date.month == month:
                        month_data = segment_df[segment_df['Month'] == available_month]
                        if not month_data.empty and 'Full_Month_Last_Year_Revenue' in segment_df.columns:
                            return month_data['Full_Month_Last_Year_Revenue'].sum()
                except:
                    continue
            
            return 0
        except Exception as e:
            logger.error(f"Error getting full month last year: {e}")
            return 0
    
    def _predict_from_occupancy_data(self, occupancy_df: pd.DataFrame, current_date: datetime) -> float:
        """Predict remaining month revenue using occupancy data patterns"""
        try:
            if 'Date' in occupancy_df.columns and 'Revenue' in occupancy_df.columns:
                # Make a copy to avoid modifying original
                occupancy_df_copy = occupancy_df.copy()
                occupancy_df_copy['Date'] = pd.to_datetime(occupancy_df_copy['Date'], errors='coerce')
                
                # Get current month data up to today
                current_month_start = datetime(current_date.year, current_date.month, 1)
                current_month_data = occupancy_df_copy[
                    (occupancy_df_copy['Date'] >= current_month_start) & 
                    (occupancy_df_copy['Date'] <= current_date)
                ]
                
                if not current_month_data.empty:
                    # Get revenue already achieved this month
                    current_month_revenue = current_month_data['Revenue'].sum()
                    
                    # Calculate average daily revenue from recent days (last 7 days or available)
                    recent_data = current_month_data.tail(min(7, len(current_month_data)))
                    clean_revenue = recent_data['Revenue'].replace(0, np.nan).dropna()
                    
                    if not clean_revenue.empty:
                        avg_daily_revenue = clean_revenue.median()
                        days_remaining = self._get_days_remaining_in_month(current_date)
                        
                        # Predict total month = current achieved + (avg daily * remaining days)
                        total_month_prediction = current_month_revenue + (avg_daily_revenue * days_remaining)
                        return total_month_prediction
            
            return 0
            
        except Exception as e:
            logger.error(f"Error in occupancy prediction: {e}")
            return 0
    
    def _get_days_remaining_in_month(self, current_date: datetime) -> int:
        """Get number of days remaining in current month"""
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1)
        
        last_day = next_month - timedelta(days=1)
        return (last_day.date() - current_date.date()).days
    
    def analyze_revenue_drivers_multiple_models(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame) -> Dict:
        """
        Use multiple ML models to determine what influences Business on the Books Revenue the most
        """
        try:
            if not ML_AVAILABLE:
                return {'error': 'Machine learning libraries not available'}
            
            # Prepare features from segment data
            features_df = self._prepare_ml_features(segment_df, occupancy_df)
            
            if features_df.empty:
                return {'error': 'No data available for ML analysis'}
            
            # Define target variable
            target_col = 'Business_on_the_Books_Revenue'
            if target_col not in features_df.columns:
                return {'error': f'Target variable {target_col} not found'}
            
            # Prepare features and target
            X, y = self._prepare_features_target(features_df, target_col)
            
            if X.empty or len(y) == 0:
                return {'error': 'No valid features or target data'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize models
            models = {
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                ),
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42)
            }
            
            model_results = {}
            
            # Train and evaluate each model
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation scores
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    
                    # Feature importance (if available)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                    elif hasattr(model, 'coef_'):
                        # For linear models, use absolute coefficients as importance
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': np.abs(model.coef_)
                        }).sort_values('importance', ascending=False)
                    
                    model_results[model_name] = {
                        'model': model,
                        'performance': {
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse,
                            'r2_score': r2,
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std()
                        },
                        'feature_importance': feature_importance,
                        'predictions': y_pred
                    }
                    
                except Exception as model_error:
                    logger.warning(f"Error training {model_name}: {model_error}")
                    model_results[model_name] = {'error': str(model_error)}
            
            # Find best model
            best_model_name = None
            best_r2 = -np.inf
            
            for model_name, result in model_results.items():
                if 'performance' in result and result['performance']['r2_score'] > best_r2:
                    best_r2 = result['performance']['r2_score']
                    best_model_name = model_name
            
            return {
                'model_results': model_results,
                'best_model': best_model_name,
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': X.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-model analysis: {e}")
            return {'error': str(e)}
    
    def _prepare_ml_features(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning"""
        try:
            # Start with segment data
            features_df = segment_df.copy()
            
            # Add time-based features
            if 'Month' in features_df.columns:
                features_df['Month'] = pd.to_datetime(features_df['Month'])
                features_df['month_num'] = features_df['Month'].dt.month
                features_df['quarter'] = features_df['Month'].dt.quarter
                features_df['is_peak_season'] = features_df['month_num'].isin([12, 1, 2, 3])  # Winter months
            
            # Encode categorical variables
            categorical_cols = ['Segment', 'MergedSegment']
            for col in categorical_cols:
                if col in features_df.columns:
                    le = LabelEncoder()
                    features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].fillna('Unknown'))
            
            # Add occupancy features if available
            if not occupancy_df.empty and 'Date' in occupancy_df.columns:
                # Aggregate occupancy by month
                occupancy_df['Date'] = pd.to_datetime(occupancy_df['Date'])
                occupancy_df['month_year'] = occupancy_df['Date'].dt.to_period('M')
                
                monthly_occ = occupancy_df.groupby('month_year').agg({
                    'Occ%': 'mean',
                    'ADR': 'mean',
                    'RevPar': 'mean',
                    'Revenue': 'sum'
                }).reset_index()
                
                monthly_occ['month_year'] = monthly_occ['month_year'].dt.to_timestamp()
                
                # Merge with segment data
                features_df['month_year'] = features_df['Month'].dt.to_period('M').dt.to_timestamp()
                features_df = features_df.merge(
                    monthly_occ, 
                    on='month_year', 
                    how='left',
                    suffixes=('', '_occ')
                )
            
            # Select numeric features
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target variable and irrelevant columns
            exclude_cols = ['Business_on_the_Books_Revenue']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            result_df = features_df[feature_cols + ['Business_on_the_Books_Revenue']].copy()
            
            # Remove rows with missing target
            result_df = result_df.dropna(subset=['Business_on_the_Books_Revenue'])
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return pd.DataFrame()
    
    def _prepare_features_target(self, features_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML"""
        try:
            # Get target
            y = features_df[target_col].copy()
            
            # Get features (exclude target)
            X = features_df.drop(columns=[target_col])
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features and target: {e}")
            return pd.DataFrame(), pd.Series()
    
    def generate_correlation_matrix(self, segment_df: pd.DataFrame, occupancy_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate correlation matrix for revenue drivers"""
        try:
            features_df = self._prepare_ml_features(segment_df, occupancy_df)
            
            if features_df.empty:
                return None
            
            # Select relevant columns for correlation
            correlation_cols = [
                'Business_on_the_Books_Revenue',
                'Business_on_the_Books_Rooms',
                'Business_on_the_Books_ADR',
                'month_num',
                'quarter',
                'is_peak_season'
            ]
            
            # Add occupancy columns if available
            occ_cols = ['Occ%', 'ADR', 'RevPar', 'Revenue_occ']
            for col in occ_cols:
                if col in features_df.columns:
                    correlation_cols.append(col)
            
            # Filter to available columns
            available_cols = [col for col in correlation_cols if col in features_df.columns]
            
            if len(available_cols) < 2:
                return None
            
            correlation_matrix = features_df[available_cols].corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {e}")
            return None
    
    def analyze_segment_performance(self, segment_df: pd.DataFrame) -> Dict:
        """
        Analyze segment performance compared to full month last year
        Flag segments that are performing well or poorly
        """
        try:
            if segment_df.empty:
                return {'error': 'No segment data available'}
            
            analysis_results = []
            
            # Required columns for analysis
            required_cols = [
                'MergedSegment', 'Business_on_the_Books_Revenue', 
                'Full_Month_Last_Year_Revenue', 'Budget_This_Year_Revenue',
                'Business_on_the_Books_Same_Time_Last_Year_Revenue'
            ]
            
            missing_cols = [col for col in required_cols if col not in segment_df.columns]
            if missing_cols:
                return {'error': f'Missing required columns: {missing_cols}'}
            
            # Group by segment for analysis
            segment_groups = segment_df.groupby('MergedSegment').agg({
                'Business_on_the_Books_Revenue': 'sum',
                'Full_Month_Last_Year_Revenue': 'sum',
                'Budget_This_Year_Revenue': 'sum',
                'Business_on_the_Books_Same_Time_Last_Year_Revenue': 'sum',
                'Month_to_Date_Revenue': 'sum'
            }).reset_index()
            
            for _, segment in segment_groups.iterrows():
                segment_name = segment['MergedSegment']
                current_bob = segment['Business_on_the_Books_Revenue']
                full_month_ly = segment['Full_Month_Last_Year_Revenue']
                budget = segment['Budget_This_Year_Revenue']
                same_time_ly = segment['Business_on_the_Books_Same_Time_Last_Year_Revenue']
                mtd_revenue = segment.get('Month_to_Date_Revenue', 0)
                
                # Calculate performance metrics
                vs_full_month_ly = ((current_bob - full_month_ly) / full_month_ly * 100) if full_month_ly > 0 else 0
                vs_budget = ((current_bob - budget) / budget * 100) if budget > 0 else 0
                vs_same_time_ly = ((current_bob - same_time_ly) / same_time_ly * 100) if same_time_ly > 0 else 0
                
                # Determine performance status
                performance_flags = []
                
                # Flag based on vs full month last year
                if vs_full_month_ly >= 20:
                    performance_flags.append("🟢 Excellent growth vs last year full month")
                elif vs_full_month_ly >= 10:
                    performance_flags.append("🟡 Good growth vs last year full month")
                elif vs_full_month_ly >= 0:
                    performance_flags.append("🟠 Moderate growth vs last year full month")
                elif vs_full_month_ly >= -10:
                    performance_flags.append("🔴 Slight decline vs last year full month")
                else:
                    performance_flags.append("🚨 Significant decline vs last year full month")
                
                # Flag based on vs budget
                if vs_budget >= 10:
                    performance_flags.append("💚 Exceeding budget significantly")
                elif vs_budget >= 0:
                    performance_flags.append("✅ Meeting/exceeding budget")
                elif vs_budget >= -10:
                    performance_flags.append("⚠️ Slightly below budget")
                else:
                    performance_flags.append("❌ Significantly below budget")
                
                # Overall assessment
                if vs_full_month_ly >= 15 and vs_budget >= 5:
                    overall_status = "🌟 Top Performer"
                elif vs_full_month_ly >= 5 and vs_budget >= 0:
                    overall_status = "✅ Good Performer"
                elif vs_full_month_ly >= -5 and vs_budget >= -5:
                    overall_status = "⚠️ Average Performer"
                elif vs_full_month_ly >= -15 or vs_budget >= -15:
                    overall_status = "🔴 Below Expectations"
                else:
                    overall_status = "🚨 Critical Attention Needed"
                
                analysis_results.append({
                    'segment': segment_name,
                    'current_bob_revenue': current_bob,
                    'full_month_ly_revenue': full_month_ly,
                    'budget_revenue': budget,
                    'same_time_ly_revenue': same_time_ly,
                    'mtd_revenue': mtd_revenue,
                    'vs_full_month_ly_pct': vs_full_month_ly,
                    'vs_budget_pct': vs_budget,
                    'vs_same_time_ly_pct': vs_same_time_ly,
                    'performance_flags': performance_flags,
                    'overall_status': overall_status
                })
            
            # Sort by performance (vs full month last year)
            analysis_results.sort(key=lambda x: x['vs_full_month_ly_pct'], reverse=True)
            
            # Summary statistics
            total_segments = len(analysis_results)
            top_performers = len([r for r in analysis_results if "Top Performer" in r['overall_status']])
            good_performers = len([r for r in analysis_results if "Good Performer" in r['overall_status']])
            attention_needed = len([r for r in analysis_results if "Critical Attention Needed" in r['overall_status']])
            
            return {
                'analysis_results': analysis_results,
                'summary': {
                    'total_segments': total_segments,
                    'top_performers': top_performers,
                    'good_performers': good_performers,
                    'attention_needed': attention_needed,
                    'top_performer_pct': (top_performers / total_segments * 100) if total_segments > 0 else 0,
                    'attention_needed_pct': (attention_needed / total_segments * 100) if total_segments > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in segment performance analysis: {e}")
            return {'error': str(e)}
    
    def forecast_current_month_occupancy_revenue(self, occupancy_df: pd.DataFrame) -> Dict:
        """
        Forecast occupancy and revenue for current month using moving average
        """
        try:
            if occupancy_df.empty:
                return {'error': 'No occupancy data available'}
            
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            
            # Get current month data
            occupancy_df['Date'] = pd.to_datetime(occupancy_df['Date'])
            current_month_data = occupancy_df[
                (occupancy_df['Date'].dt.month == current_month) & 
                (occupancy_df['Date'].dt.year == current_year)
            ].copy()
            
            if current_month_data.empty:
                return {'error': 'No current month data available'}
            
            # Get last 7 days for moving average - should be actual completed days (excluding today)
            # Sort by date and get the most recent ACTUAL entries (up to yesterday)
            yesterday_date = current_date - timedelta(days=1)
            
            # Get last 7 days of actual data up to yesterday
            actual_data = occupancy_df[
                (occupancy_df['Date'].dt.date <= yesterday_date.date()) &
                (occupancy_df['Date'].dt.date >= (yesterday_date - timedelta(days=6)).date())
            ].copy()
            
            latest_data = actual_data.sort_values('Date').tail(7).copy()
            
            if len(latest_data) < 3:
                return {'error': 'Insufficient current month data for moving average'}
            
            # Calculate moving averages with proper data cleaning
            # Remove any invalid values
            latest_clean = latest_data[
                (latest_data['Occ%'] >= 0) & (latest_data['Occ%'] <= 100) &
                (latest_data['Revenue'] >= 0) & 
                (latest_data['ADR'] >= 0) & (latest_data['ADR'] <= 2000)
            ]
            
            if len(latest_clean) < 3:
                return {'error': 'Insufficient clean data for moving average'}
            
            # Calculate moving averages
            avg_occupancy = latest_clean['Occ%'].mean()
            avg_revenue = latest_clean['Revenue'].mean()
            avg_adr = latest_clean['ADR'].mean()
            
            # Get remaining days in month
            if current_month == 12:
                next_month = datetime(current_year + 1, 1, 1)
            else:
                next_month = datetime(current_year, current_month + 1, 1)
            
            last_day_of_month = next_month - timedelta(days=1)
            days_remaining = (last_day_of_month.date() - current_date.date()).days
            
            # Generate forecast for remaining days (from today to month end)
            forecast_dates = pd.date_range(
                start=current_date,
                end=last_day_of_month,
                freq='D'
            )
            
            forecast_data = []
            for date in forecast_dates:
                # Add some variation based on day of week
                dow_factor = self._get_day_of_week_factor(date.weekday())
                
                forecasted_occ = avg_occupancy * dow_factor
                forecasted_revenue = avg_revenue * dow_factor
                forecasted_adr = avg_adr * dow_factor
                
                # Ensure reasonable bounds
                forecasted_occ = max(0, min(100, forecasted_occ))
                forecasted_revenue = max(0, forecasted_revenue)
                forecasted_adr = max(0, forecasted_adr)
                
                forecast_data.append({
                    'date': date,
                    'forecasted_occupancy': forecasted_occ,
                    'forecasted_revenue': forecasted_revenue,
                    'forecasted_adr': forecasted_adr,
                    'day_of_week': date.strftime('%A')
                })
            
            # Calculate month-end projections
            # Get actual revenue up to yesterday (confirmed/closed days)
            actual_revenue_to_date = current_month_data[
                current_month_data['Date'].dt.date <= yesterday_date.date()
            ]['Revenue'].sum()
            
            # Projected revenue for remaining days (from today to month end)
            remaining_revenue_forecast = sum([f['forecasted_revenue'] for f in forecast_data])
            
            # Total month-end forecast = actual confirmed revenue + projected remaining revenue
            month_end_revenue_forecast = actual_revenue_to_date + remaining_revenue_forecast
            
            current_month_avg_occ = current_month_data['Occ%'].mean()
            forecast_avg_occ = sum([f['forecasted_occupancy'] for f in forecast_data]) / len(forecast_data) if forecast_data else 0
            month_avg_occ_forecast = (current_month_avg_occ + forecast_avg_occ) / 2
            
            return {
                'forecast_data': forecast_data,
                'moving_averages': {
                    'avg_occupancy_7d': avg_occupancy,
                    'avg_revenue_7d': avg_revenue,
                    'avg_adr_7d': avg_adr
                },
                'month_projections': {
                    'actual_revenue_to_date': actual_revenue_to_date,
                    'remaining_revenue_forecast': remaining_revenue_forecast,
                    'month_end_revenue_forecast': month_end_revenue_forecast,
                    'current_month_avg_occ': current_month_avg_occ,
                    'month_avg_occ_forecast': month_avg_occ_forecast
                },
                'days_remaining': days_remaining,
                'forecast_period': f"{forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}" if forecast_data else "No forecast period"
            }
            
        except Exception as e:
            logger.error(f"Error in current month occupancy/revenue forecast: {e}")
            return {'error': str(e)}
    
    def _get_day_of_week_factor(self, weekday: int) -> float:
        """
        Get day of week adjustment factor for forecasting
        0=Monday, 6=Sunday
        """
        # Hotel industry typically sees higher occupancy on weekends
        dow_factors = {
            0: 0.95,  # Monday
            1: 1.0,   # Tuesday
            2: 1.0,   # Wednesday
            3: 1.05,  # Thursday
            4: 1.1,   # Friday
            5: 1.15,  # Saturday
            6: 1.05   # Sunday
        }
        return dow_factors.get(weekday, 1.0)

def get_advanced_forecaster() -> AdvancedForecastor:
    """Get advanced forecaster instance - singleton pattern"""
    if not hasattr(get_advanced_forecaster, '_instance'):
        get_advanced_forecaster._instance = AdvancedForecastor()
    return get_advanced_forecaster._instance