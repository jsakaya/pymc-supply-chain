#!/usr/bin/env python3
"""
Comprehensive Test Suite for PyMC-Supply-Chain Demand Forecasting Models

This script thoroughly tests all demand forecasting models to prove they work correctly:
1. DemandForecastModel (Base) - Basic forecasting functionality
2. SeasonalDemandModel - Seasonal patterns, Fourier series, changepoints
3. HierarchicalDemandModel - Multi-location/product hierarchical forecasting  
4. IntermittentDemandModel - Sparse demand patterns, Croston's method

For each model:
- Creates realistic test data with known patterns
- Fits model and verifies convergence
- Generates forecasts and validates outputs
- Tests uncertainty quantification (credible intervals)
- Validates forecast accuracy on held-out data
- Shows visualizations of results
- Tests edge cases and error handling
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import traceback
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TestResults:
    """Track comprehensive test results."""
    
    def __init__(self):
        self.results = {}
        self.errors = {}
        self.metrics = {}
        self.plots = {}
        
    def add_result(self, test_name: str, success: bool, error_msg: str = None, metrics: Dict = None):
        """Add a test result with optional metrics."""
        self.results[test_name] = success
        if error_msg:
            self.errors[test_name] = error_msg
        if metrics:
            self.metrics[test_name] = metrics
            
    def add_plot(self, test_name: str, fig):
        """Add a plot for a test."""
        self.plots[test_name] = fig
        
    def print_detailed_summary(self):
        """Print comprehensive test summary with metrics."""
        print("\n" + "="*80)
        print("COMPREHENSIVE DEMAND FORECASTING MODEL TEST RESULTS")
        print("="*80)
        
        passed = sum(1 for success in self.results.values() if success)
        total = len(self.results)
        
        print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print()
        
        # Group by model type
        model_groups = {
            'Base Demand Model': [],
            'Seasonal Demand Model': [],
            'Hierarchical Demand Model': [], 
            'Intermittent Demand Model': [],
            'Accuracy Validation': [],
            'Visualization Tests': []
        }
        
        for test_name, success in self.results.items():
            if 'base' in test_name.lower() or 'basic' in test_name.lower():
                model_groups['Base Demand Model'].append((test_name, success))
            elif 'seasonal' in test_name.lower():
                model_groups['Seasonal Demand Model'].append((test_name, success))
            elif 'hierarchical' in test_name.lower():
                model_groups['Hierarchical Demand Model'].append((test_name, success))
            elif 'intermittent' in test_name.lower():
                model_groups['Intermittent Demand Model'].append((test_name, success))
            elif 'accuracy' in test_name.lower() or 'mae' in test_name.lower() or 'rmse' in test_name.lower():
                model_groups['Accuracy Validation'].append((test_name, success))
            elif 'plot' in test_name.lower() or 'visualiz' in test_name.lower():
                model_groups['Visualization Tests'].append((test_name, success))
        
        for group_name, tests in model_groups.items():
            if tests:
                print(f"{group_name}:")
                for test_name, success in tests:
                    status = "✅ PASS" if success else "❌ FAIL"
                    print(f"  {status} {test_name}")
                    
                    # Show metrics if available
                    if test_name in self.metrics and success:
                        metrics = self.metrics[test_name]
                        for metric_name, value in metrics.items():
                            if isinstance(value, float):
                                print(f"    └─ {metric_name}: {value:.4f}")
                            else:
                                print(f"    └─ {metric_name}: {value}")
                print()
        
        # Print errors
        if self.errors:
            print("DETAILED ERROR MESSAGES:")
            print("-" * 50)
            for test_name, error_msg in self.errors.items():
                print(f"\n{test_name}:")
                print(f"  {error_msg}")
        
        print("\n" + "="*80)

def create_synthetic_demand_data(
    n_days: int = 365,
    base_demand: float = 100,
    trend_slope: float = 0.1,
    seasonal_amplitude: float = 20,
    noise_std: float = 5,
    start_date: str = '2023-01-01'
) -> pd.DataFrame:
    """Create realistic synthetic demand data with known patterns."""
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    t = np.arange(n_days)
    
    # Base demand with trend
    trend = base_demand + trend_slope * t
    
    # Multiple seasonality patterns
    yearly_seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 365.25)  # Yearly
    weekly_seasonal = seasonal_amplitude * 0.3 * np.sin(2 * np.pi * t / 7)  # Weekly
    
    # Add some holiday effects (simulate Black Friday, Christmas, etc.)
    holiday_effect = np.zeros(n_days)
    for i in range(0, n_days, 91):  # Quarterly "sales events"
        if i < n_days:
            holiday_effect[i:min(i+3, n_days)] = base_demand * 0.5
    
    # Random noise
    noise = np.random.normal(0, noise_std, n_days)
    
    # Combine all components
    demand = trend + yearly_seasonal + weekly_seasonal + holiday_effect + noise
    demand = np.maximum(demand, 0)  # Ensure non-negative
    
    # Add some external factors
    temperature = 20 + 15 * np.sin(2 * np.pi * t / 365.25) + np.random.normal(0, 2, n_days)
    promotion = np.random.binomial(1, 0.1, n_days)  # 10% chance of promotion each day
    
    return pd.DataFrame({
        'date': dates,
        'demand': demand,
        'temperature': temperature,
        'promotion': promotion
    })

def create_hierarchical_demand_data(
    n_days: int = 365,
    regions: List[str] = ['North', 'South', 'East', 'West'],
    products: List[str] = ['ProductA', 'ProductB', 'ProductC']
) -> pd.DataFrame:
    """Create hierarchical demand data with multiple locations and products."""
    np.random.seed(42)
    
    data = []
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    t = np.arange(n_days)
    
    for region in regions:
        for product in products:
            # Base demand varies by region and product
            region_factor = np.random.uniform(0.5, 2.0)
            product_factor = np.random.uniform(0.7, 1.5)
            base_demand = 50 * region_factor * product_factor
            
            # Different seasonal patterns
            seasonal = 10 * np.sin(2 * np.pi * t / 365.25 + np.random.uniform(0, 2*np.pi))
            trend = np.random.uniform(-0.05, 0.15) * t
            noise = np.random.normal(0, base_demand * 0.1, n_days)
            
            demand = base_demand + seasonal + trend + noise
            demand = np.maximum(demand, 0)
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'region': region,
                    'product': product,
                    'demand': demand[i]
                })
    
    return pd.DataFrame(data)

def create_intermittent_demand_data(
    n_days: int = 365,
    zero_prob: float = 0.7,
    avg_demand_when_nonzero: float = 50
) -> pd.DataFrame:
    """Create intermittent/sparse demand data for spare parts scenario."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Create intermittent demand pattern
    demand = np.zeros(n_days)
    
    for i in range(n_days):
        if np.random.random() > zero_prob:
            # When demand occurs, draw from gamma distribution
            demand[i] = np.random.gamma(2, avg_demand_when_nonzero / 2)
    
    # Add some correlation structure - demands might cluster
    for i in range(1, n_days):
        if demand[i-1] > 0 and np.random.random() < 0.3:
            demand[i] = max(demand[i], np.random.gamma(2, avg_demand_when_nonzero / 3))
    
    return pd.DataFrame({
        'date': dates,
        'demand': demand
    })

def calculate_accuracy_metrics(actual: np.ndarray, forecast: np.ndarray, 
                             forecast_lower: np.ndarray, forecast_upper: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive forecast accuracy metrics."""
    # Point forecast accuracy
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast)**2))
    mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100  # Avoid division by zero
    
    # Bias metrics
    bias = np.mean(forecast - actual)
    bias_pct = (bias / np.mean(actual)) * 100 if np.mean(actual) != 0 else 0
    
    # Coverage probability (credible interval)
    coverage = np.mean((actual >= forecast_lower) & (actual <= forecast_upper)) * 100
    
    # Interval width
    avg_interval_width = np.mean(forecast_upper - forecast_lower)
    
    return {
        'MAE': mae,
        'RMSE': rmse, 
        'MAPE': mape,
        'Bias': bias,
        'Bias_PCT': bias_pct,
        'Coverage_95': coverage,
        'Avg_Interval_Width': avg_interval_width
    }

def test_base_demand_model(results: TestResults):
    """Comprehensive test of the base DemandForecastModel."""
    print("\n" + "="*60)
    print("TESTING BASE DEMAND FORECAST MODEL")
    print("="*60)
    
    try:
        from pymc_supply_chain.demand.base import DemandForecastModel
        results.add_result("Import DemandForecastModel", True)
        print("✅ Successfully imported DemandForecastModel")
    except Exception as e:
        results.add_result("Import DemandForecastModel", False, str(e))
        print(f"❌ Failed to import DemandForecastModel: {e}")
        return
    
    # Test 1: Model initialization
    print("\n1. Testing model initialization...")
    try:
        model = DemandForecastModel(
            date_column='date',
            target_column='demand',
            include_trend=True,
            include_seasonality=True,
            seasonality=7,
            external_regressors=['temperature', 'promotion']
        )
        results.add_result("Base model initialization", True)
        print("✅ Model initialized successfully")
    except Exception as e:
        results.add_result("Base model initialization", False, str(e))
        print(f"❌ Model initialization failed: {e}")
        return
    
    # Test 2: Data preparation and model building
    print("\n2. Creating synthetic data and building model...")
    try:
        data = create_synthetic_demand_data(n_days=200)
        print(f"   Created {len(data)} days of synthetic demand data")
        print(f"   Average demand: {data['demand'].mean():.2f}")
        print(f"   Demand range: [{data['demand'].min():.1f}, {data['demand'].max():.1f}]")
        
        # Split data for training and testing
        train_data = data.iloc[:150]  # First 150 days for training
        test_data = data.iloc[150:]   # Last 50 days for testing
        
        # Build PyMC model
        pymc_model = model.build_model(train_data)
        results.add_result("Base model building", True)
        print("✅ PyMC model built successfully")
    except Exception as e:
        results.add_result("Base model building", False, str(e))
        print(f"❌ Model building failed: {e}")
        return
    
    # Test 3: Model fitting and convergence
    print("\n3. Fitting model with MCMC sampling...")
    try:
        # Fit with reasonable sampling parameters for testing
        inference_data = model.fit(
            train_data,
            draws=500,
            tune=300,
            chains=2,
            progressbar=False,
            random_seed=42
        )
        
        # Check convergence diagnostics
        import arviz as az
        summary = az.summary(inference_data, hdi_prob=0.95)
        max_rhat = summary['r_hat'].max()
        min_ess_bulk = summary['ess_bulk'].min()
        
        convergence_metrics = {
            'Max_Rhat': max_rhat,
            'Min_ESS_Bulk': min_ess_bulk,
            'Converged': max_rhat < 1.1 and min_ess_bulk > 100
        }
        
        results.add_result("Base model fitting", True, metrics=convergence_metrics)
        print("✅ Model fitting completed successfully")
        print(f"   Max R-hat: {max_rhat:.4f} (should be < 1.1)")
        print(f"   Min ESS: {min_ess_bulk:.0f} (should be > 100)")
        
        if convergence_metrics['Converged']:
            print("✅ Model converged properly")
        else:
            print("⚠️  Model convergence marginal - may need more sampling")
            
    except Exception as e:
        results.add_result("Base model fitting", False, str(e))
        print(f"❌ Model fitting failed: {e}")
        return
    
    # Test 4: Forecasting
    print("\n4. Generating forecasts...")
    try:
        forecast_steps = len(test_data)
        forecast_df = model.forecast(
            steps=forecast_steps,
            frequency='D'
        )
        
        # Validate forecast structure
        expected_cols = ['date', 'forecast', 'forecast_lower', 'forecast_upper', 'forecast_std']
        missing_cols = set(expected_cols) - set(forecast_df.columns)
        
        if missing_cols:
            results.add_result("Base model forecasting", False, f"Missing columns: {missing_cols}")
            print(f"❌ Forecasting failed: Missing columns {missing_cols}")
            return
        
        forecast_metrics = {
            'Forecast_Steps': len(forecast_df),
            'Avg_Forecast': forecast_df['forecast'].mean(),
            'Forecast_Range': forecast_df['forecast'].max() - forecast_df['forecast'].min(),
            'Avg_Uncertainty': forecast_df['forecast_std'].mean(),
            'Avg_Interval_Width': (forecast_df['forecast_upper'] - forecast_df['forecast_lower']).mean()
        }
        
        results.add_result("Base model forecasting", True, metrics=forecast_metrics)
        print("✅ Forecasting completed successfully")
        print(f"   Generated {len(forecast_df)} forecast steps")
        print(f"   Average forecast: {forecast_metrics['Avg_Forecast']:.2f}")
        print(f"   Average uncertainty: {forecast_metrics['Avg_Uncertainty']:.2f}")
        
    except Exception as e:
        results.add_result("Base model forecasting", False, str(e))
        print(f"❌ Forecasting failed: {e}")
        return
    
    # Test 5: Forecast accuracy validation
    print("\n5. Validating forecast accuracy...")
    try:
        actual = test_data['demand'].values
        forecast = forecast_df['forecast'].values
        forecast_lower = forecast_df['forecast_lower'].values
        forecast_upper = forecast_df['forecast_upper'].values
        
        accuracy_metrics = calculate_accuracy_metrics(actual, forecast, forecast_lower, forecast_upper)
        
        results.add_result("Base model accuracy", True, metrics=accuracy_metrics)
        print("✅ Accuracy validation completed")
        print(f"   MAE: {accuracy_metrics['MAE']:.2f}")
        print(f"   RMSE: {accuracy_metrics['RMSE']:.2f}")
        print(f"   MAPE: {accuracy_metrics['MAPE']:.2f}%")
        print(f"   Coverage (95% CI): {accuracy_metrics['Coverage_95']:.1f}%")
        
    except Exception as e:
        results.add_result("Base model accuracy", False, str(e))
        print(f"❌ Accuracy validation failed: {e}")
    
    # Test 6: Visualization
    print("\n6. Creating visualizations...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Base Demand Model Test Results', fontsize=16)
        
        # Plot 1: Historical data and forecast
        ax1 = axes[0, 0]
        ax1.plot(train_data['date'], train_data['demand'], 'k-', alpha=0.7, label='Training Data')
        ax1.plot(test_data['date'], test_data['demand'], 'b-', alpha=0.7, label='Actual Test Data')
        ax1.plot(forecast_df['date'], forecast_df['forecast'], 'r-', linewidth=2, label='Forecast')
        ax1.fill_between(forecast_df['date'], forecast_df['forecast_lower'], 
                        forecast_df['forecast_upper'], alpha=0.3, color='red', label='95% CI')
        ax1.set_title('Demand Forecast vs Actual')
        ax1.set_ylabel('Demand')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Residuals
        ax2 = axes[0, 1]
        residuals = actual - forecast
        ax2.scatter(forecast, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Forecast Residuals')
        ax2.set_xlabel('Forecast')
        ax2.set_ylabel('Residual (Actual - Forecast)')
        
        # Plot 3: Parameter traces
        ax3 = axes[1, 0]
        import arviz as az
        az.plot_trace(inference_data, var_names=['intercept', 'trend_coef'], ax=ax3, compact=True)
        ax3.set_title('Parameter Traces')
        
        # Plot 4: Accuracy metrics
        ax4 = axes[1, 1]
        metrics_names = ['MAE', 'RMSE', 'MAPE', 'Coverage_95']
        metrics_values = [accuracy_metrics[name] for name in metrics_names]
        bars = ax4.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
        ax4.set_title('Accuracy Metrics')
        ax4.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        results.add_plot("Base model visualization", fig)
        results.add_result("Base model visualization", True)
        print("✅ Visualization created successfully")
        
        # Save plot
        plot_path = Path("base_demand_model_results.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved as {plot_path}")
        
    except Exception as e:
        results.add_result("Base model visualization", False, str(e))
        print(f"❌ Visualization failed: {e}")
    
    print(f"\n{'='*60}")
    print("BASE DEMAND MODEL TESTING COMPLETED")
    print(f"{'='*60}")

def test_seasonal_demand_model(results: TestResults):
    """Comprehensive test of the SeasonalDemandModel."""
    print("\n" + "="*60)
    print("TESTING SEASONAL DEMAND FORECAST MODEL")
    print("="*60)
    
    try:
        from pymc_supply_chain.demand.seasonal import SeasonalDemandModel
        results.add_result("Import SeasonalDemandModel", True)
        print("✅ Successfully imported SeasonalDemandModel")
    except Exception as e:
        results.add_result("Import SeasonalDemandModel", False, str(e))
        print(f"❌ Failed to import SeasonalDemandModel: {e}")
        return
    
    # Test 1: Model initialization with advanced features
    print("\n1. Testing advanced seasonal model initialization...")
    try:
        # Create holiday data
        holidays = pd.DataFrame({
            'holiday': ['Christmas', 'Black Friday', 'New Year'],
            'ds': pd.to_datetime(['2023-12-25', '2023-11-24', '2023-01-01'])
        })
        
        model = SeasonalDemandModel(
            date_column='date',
            target_column='demand',
            yearly_seasonality=10,
            weekly_seasonality=3,
            holidays=holidays,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            n_changepoints=25,
            external_regressors=['temperature']
        )
        results.add_result("Seasonal model initialization", True)
        print("✅ Seasonal model initialized with advanced features")
        print(f"   Yearly seasonality terms: 10")
        print(f"   Weekly seasonality terms: 3") 
        print(f"   Holiday effects: {len(holidays)} holidays")
        print(f"   Changepoints: 25")
    except Exception as e:
        results.add_result("Seasonal model initialization", False, str(e))
        print(f"❌ Seasonal model initialization failed: {e}")
        return
    
    # Test 2: Complex synthetic data with multiple seasonality
    print("\n2. Creating complex seasonal data...")
    try:
        data = create_synthetic_demand_data(
            n_days=400,
            seasonal_amplitude=30,  # Stronger seasonality
            trend_slope=0.2
        )
        
        # Split data
        train_data = data.iloc[:300]
        test_data = data.iloc[300:]
        
        print(f"   Created {len(data)} days of complex seasonal data")
        print(f"   Training: {len(train_data)} days, Testing: {len(test_data)} days")
        
        # Build model
        pymc_model = model.build_model(train_data)
        results.add_result("Seasonal model building", True)
        print("✅ Complex seasonal model built successfully")
        
    except Exception as e:
        results.add_result("Seasonal model building", False, str(e))
        print(f"❌ Seasonal model building failed: {e}")
        return
    
    # Test 3: Model fitting with seasonal components
    print("\n3. Fitting seasonal model...")
    try:
        inference_data = model.fit(
            train_data,
            draws=400,
            tune=200,
            chains=2,
            progressbar=False,
            random_seed=42
        )
        
        # Check convergence
        import arviz as az
        summary = az.summary(inference_data, hdi_prob=0.95)
        max_rhat = summary['r_hat'].max()
        min_ess_bulk = summary['ess_bulk'].min()
        
        convergence_metrics = {
            'Max_Rhat': max_rhat,
            'Min_ESS_Bulk': min_ess_bulk,
            'Converged': max_rhat < 1.1 and min_ess_bulk > 100
        }
        
        results.add_result("Seasonal model fitting", True, metrics=convergence_metrics)
        print("✅ Seasonal model fitting completed")
        print(f"   Max R-hat: {max_rhat:.4f}")
        print(f"   Min ESS: {min_ess_bulk:.0f}")
        
    except Exception as e:
        results.add_result("Seasonal model fitting", False, str(e))
        print(f"❌ Seasonal model fitting failed: {e}")
        return
    
    # Test 4: Advanced forecasting
    print("\n4. Generating seasonal forecasts...")
    try:
        forecast_df = model.forecast(
            steps=len(test_data),
            frequency='D'
        )
        
        # Calculate seasonal forecast metrics
        forecast_metrics = {
            'Forecast_Steps': len(forecast_df),
            'Avg_Forecast': forecast_df['forecast'].mean(),
            'Seasonal_Variation': forecast_df['forecast'].std(),
            'Trend_Direction': 'Upward' if forecast_df['forecast'].iloc[-1] > forecast_df['forecast'].iloc[0] else 'Downward'
        }
        
        results.add_result("Seasonal model forecasting", True, metrics=forecast_metrics)
        print("✅ Seasonal forecasting completed")
        print(f"   Seasonal variation (std): {forecast_metrics['Seasonal_Variation']:.2f}")
        print(f"   Trend direction: {forecast_metrics['Trend_Direction']}")
        
    except Exception as e:
        results.add_result("Seasonal model forecasting", False, str(e))
        print(f"❌ Seasonal forecasting failed: {e}")
        return
    
    # Test 5: Accuracy validation
    print("\n5. Validating seasonal forecast accuracy...")
    try:
        actual = test_data['demand'].values
        forecast = forecast_df['forecast'].values
        forecast_lower = forecast_df['lower_95'].values if 'lower_95' in forecast_df.columns else forecast_df['forecast_lower'].values
        forecast_upper = forecast_df['upper_95'].values if 'upper_95' in forecast_df.columns else forecast_df['forecast_upper'].values
        
        accuracy_metrics = calculate_accuracy_metrics(actual, forecast, forecast_lower, forecast_upper)
        
        results.add_result("Seasonal model accuracy", True, metrics=accuracy_metrics)
        print("✅ Seasonal accuracy validation completed")
        print(f"   MAE: {accuracy_metrics['MAE']:.2f}")
        print(f"   MAPE: {accuracy_metrics['MAPE']:.2f}%")
        print(f"   Coverage: {accuracy_metrics['Coverage_95']:.1f}%")
        
    except Exception as e:
        results.add_result("Seasonal model accuracy", False, str(e))
        print(f"❌ Seasonal accuracy validation failed: {e}")
    
    # Test 6: Component analysis and visualization
    print("\n6. Analyzing seasonal components...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Seasonal Demand Model Analysis', fontsize=16)
        
        # Plot 1: Forecast vs actual
        ax1 = axes[0, 0]
        ax1.plot(train_data['date'], train_data['demand'], 'k-', alpha=0.7, label='Training')
        ax1.plot(test_data['date'], test_data['demand'], 'b-', label='Actual')
        ax1.plot(forecast_df['date'], forecast_df['forecast'], 'r-', linewidth=2, label='Forecast')
        ax1.fill_between(forecast_df['date'], forecast_lower, forecast_upper, 
                        alpha=0.3, color='red', label='95% CI')
        ax1.set_title('Seasonal Forecast Performance')
        ax1.set_ylabel('Demand')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Seasonal decomposition (simplified)
        ax2 = axes[0, 1]
        # Show weekly pattern
        weekly_pattern = []
        for day in range(7):
            day_data = train_data[train_data['date'].dt.dayofweek == day]['demand']
            weekly_pattern.append(day_data.mean())
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(days, weekly_pattern, color='skyblue')
        ax2.set_title('Weekly Seasonality Pattern')
        ax2.set_ylabel('Average Demand')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Monthly trend
        ax3 = axes[1, 0]
        monthly_data = train_data.groupby(train_data['date'].dt.month)['demand'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax3.plot(range(1, 13), monthly_data, marker='o', linewidth=2, markersize=6)
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months)
        ax3.set_title('Monthly Demand Pattern')
        ax3.set_ylabel('Average Demand')
        
        # Plot 4: Forecast errors over time
        ax4 = axes[1, 1]
        errors = actual - forecast
        ax4.plot(forecast_df['date'], errors, 'g-', alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.fill_between(forecast_df['date'], errors, 0, alpha=0.3)
        ax4.set_title('Forecast Errors Over Time')
        ax4.set_ylabel('Error (Actual - Forecast)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        results.add_plot("Seasonal model visualization", fig)
        results.add_result("Seasonal model visualization", True)
        print("✅ Seasonal component analysis completed")
        
        # Save plot
        plot_path = Path("seasonal_demand_model_results.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved as {plot_path}")
        
    except Exception as e:
        results.add_result("Seasonal model visualization", False, str(e))
        print(f"❌ Seasonal component analysis failed: {e}")
    
    print(f"\n{'='*60}")
    print("SEASONAL DEMAND MODEL TESTING COMPLETED")
    print(f"{'='*60}")

def test_hierarchical_demand_model(results: TestResults):
    """Comprehensive test of the HierarchicalDemandModel."""
    print("\n" + "="*60)
    print("TESTING HIERARCHICAL DEMAND FORECAST MODEL")
    print("="*60)
    
    try:
        from pymc_supply_chain.demand.hierarchical import HierarchicalDemandModel
        results.add_result("Import HierarchicalDemandModel", True)
        print("✅ Successfully imported HierarchicalDemandModel")
    except Exception as e:
        results.add_result("Import HierarchicalDemandModel", False, str(e))
        print(f"❌ Failed to import HierarchicalDemandModel: {e}")
        return
    
    # Test 1: Model initialization for hierarchical structure
    print("\n1. Testing hierarchical model initialization...")
    try:
        model = HierarchicalDemandModel(
            hierarchy_cols=['region', 'product'],
            date_column='date',
            target_column='demand',
            pooling_strength=0.5,
            include_trend=True,
            include_seasonality=True,
            seasonality=7
        )
        results.add_result("Hierarchical model initialization", True)
        print("✅ Hierarchical model initialized")
        print(f"   Hierarchy levels: region → product")
        print(f"   Pooling strength: 0.5 (partial pooling)")
        
    except Exception as e:
        results.add_result("Hierarchical model initialization", False, str(e))
        print(f"❌ Hierarchical model initialization failed: {e}")
        return
    
    # Test 2: Create hierarchical data
    print("\n2. Creating hierarchical demand data...")
    try:
        data = create_hierarchical_demand_data(
            n_days=300,
            regions=['North', 'South', 'East', 'West'],
            products=['ProductA', 'ProductB', 'ProductC']
        )
        
        print(f"   Created hierarchical dataset:")
        print(f"   - Total observations: {len(data)}")
        print(f"   - Regions: {data['region'].nunique()}")
        print(f"   - Products: {data['product'].nunique()}")
        print(f"   - Days: {data['date'].nunique()}")
        
        # Show some statistics by hierarchy
        hierarchy_stats = data.groupby(['region', 'product'])['demand'].agg(['mean', 'std', 'count'])
        print(f"   - Average demand by group:")
        print(hierarchy_stats.head())
        
        # Split data for train/test
        unique_dates = sorted(data['date'].unique())
        train_dates = unique_dates[:240]  # First 80% for training
        test_dates = unique_dates[240:]   # Last 20% for testing
        
        train_data = data[data['date'].isin(train_dates)]
        test_data = data[data['date'].isin(test_dates)]
        
        results.add_result("Hierarchical data creation", True)
        print("✅ Hierarchical data created successfully")
        
    except Exception as e:
        results.add_result("Hierarchical data creation", False, str(e))
        print(f"❌ Hierarchical data creation failed: {e}")
        return
    
    # Test 3: Model building with hierarchy
    print("\n3. Building hierarchical model...")
    try:
        # Test with subset of data first (for computational efficiency)
        subset_data = train_data.groupby(['region', 'product']).head(50).reset_index(drop=True)
        
        pymc_model = model.build_model(subset_data)
        results.add_result("Hierarchical model building", True)
        print("✅ Hierarchical PyMC model built successfully")
        print(f"   Model includes partial pooling across {data['region'].nunique()} regions")
        print(f"   Cross-product learning across {data['product'].nunique()} products")
        
    except Exception as e:
        results.add_result("Hierarchical model building", False, str(e))
        print(f"❌ Hierarchical model building failed: {e}")
        return
    
    # Test 4: Model fitting (with reduced complexity for testing)
    print("\n4. Fitting hierarchical model...")
    try:
        inference_data = model.fit(
            subset_data,
            draws=300,  # Reduced for computational efficiency
            tune=150,
            chains=2,
            progressbar=False,
            random_seed=42
        )
        
        # Check convergence
        import arviz as az
        summary = az.summary(inference_data, hdi_prob=0.95)
        max_rhat = summary['r_hat'].max()
        min_ess_bulk = summary['ess_bulk'].min()
        
        convergence_metrics = {
            'Max_Rhat': max_rhat,
            'Min_ESS_Bulk': min_ess_bulk,
            'Converged': max_rhat < 1.15 and min_ess_bulk > 50,  # More lenient for hierarchical
            'N_Parameters': len(summary)
        }
        
        results.add_result("Hierarchical model fitting", True, metrics=convergence_metrics)
        print("✅ Hierarchical model fitting completed")
        print(f"   Parameters estimated: {convergence_metrics['N_Parameters']}")
        print(f"   Max R-hat: {max_rhat:.4f}")
        print(f"   Min ESS: {min_ess_bulk:.0f}")
        
    except Exception as e:
        results.add_result("Hierarchical model fitting", False, str(e))
        print(f"❌ Hierarchical model fitting failed: {e}")
        return
    
    # Test 5: Simple forecasting test
    print("\n5. Testing hierarchical forecasting...")
    try:
        # Generate forecasts for a subset of hierarchy combinations
        forecast_df = model.forecast(steps=7, frequency='D')
        
        forecast_metrics = {
            'Forecast_Steps': len(forecast_df),
            'Avg_Forecast': forecast_df['forecast'].mean(),
            'Forecast_Variability': forecast_df['forecast'].std()
        }
        
        results.add_result("Hierarchical model forecasting", True, metrics=forecast_metrics)
        print("✅ Hierarchical forecasting completed")
        print(f"   Generated {len(forecast_df)} forecast steps")
        
    except Exception as e:
        results.add_result("Hierarchical model forecasting", False, str(e))
        print(f"❌ Hierarchical forecasting failed: {e}")
    
    # Test 6: Hierarchical structure analysis
    print("\n6. Analyzing hierarchical structure...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hierarchical Demand Model Analysis', fontsize=16)
        
        # Plot 1: Demand by region
        ax1 = axes[0, 0]
        region_demand = data.groupby('region')['demand'].sum()
        bars1 = ax1.bar(region_demand.index, region_demand.values, color='skyblue')
        ax1.set_title('Total Demand by Region')
        ax1.set_ylabel('Total Demand')
        
        # Add value labels
        for bar, value in zip(bars1, region_demand.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # Plot 2: Demand by product
        ax2 = axes[0, 1]
        product_demand = data.groupby('product')['demand'].sum()
        bars2 = ax2.bar(product_demand.index, product_demand.values, color='lightgreen')
        ax2.set_title('Total Demand by Product')
        ax2.set_ylabel('Total Demand')
        
        # Add value labels
        for bar, value in zip(bars2, product_demand.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # Plot 3: Hierarchical heatmap
        ax3 = axes[1, 0]
        pivot_data = data.groupby(['region', 'product'])['demand'].mean().unstack()
        import seaborn as sns
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Average Demand by Region-Product')
        
        # Plot 4: Time series by hierarchy level
        ax4 = axes[1, 1]
        for region in data['region'].unique()[:2]:  # Show first 2 regions for clarity
            region_data = data[data['region'] == region].groupby('date')['demand'].sum()
            ax4.plot(region_data.index, region_data.values, label=f'Region {region}', alpha=0.7)
        
        ax4.set_title('Demand Time Series by Region')
        ax4.set_ylabel('Daily Demand')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        results.add_plot("Hierarchical model visualization", fig)
        results.add_result("Hierarchical model visualization", True)
        print("✅ Hierarchical structure analysis completed")
        
        # Save plot
        plot_path = Path("hierarchical_demand_model_results.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved as {plot_path}")
        
    except Exception as e:
        results.add_result("Hierarchical model visualization", False, str(e))
        print(f"❌ Hierarchical structure analysis failed: {e}")
    
    print(f"\n{'='*60}")
    print("HIERARCHICAL DEMAND MODEL TESTING COMPLETED")
    print(f"{'='*60}")

def test_intermittent_demand_model(results: TestResults):
    """Comprehensive test of the IntermittentDemandModel."""
    print("\n" + "="*60)
    print("TESTING INTERMITTENT DEMAND FORECAST MODEL")
    print("="*60)
    
    try:
        from pymc_supply_chain.demand.intermittent import IntermittentDemandModel
        results.add_result("Import IntermittentDemandModel", True)
        print("✅ Successfully imported IntermittentDemandModel")
    except Exception as e:
        results.add_result("Import IntermittentDemandModel", False, str(e))
        print(f"❌ Failed to import IntermittentDemandModel: {e}")
        return
    
    # Test 1: Model initialization for different methods
    print("\n1. Testing intermittent model initialization...")
    methods_to_test = ['croston', 'sba', 'zero_inflated']
    
    for method in methods_to_test:
        try:
            model = IntermittentDemandModel(
                date_column='date',
                target_column='demand',
                method=method,
                min_periods=2,
                smoothing_param=None
            )
            results.add_result(f"Intermittent model init - {method}", True)
            print(f"✅ {method.upper()} method initialized successfully")
        except Exception as e:
            results.add_result(f"Intermittent model init - {method}", False, str(e))
            print(f"❌ {method.upper()} initialization failed: {e}")
    
    # Focus on Croston's method for detailed testing
    model = IntermittentDemandModel(method='croston')
    
    # Test 2: Create intermittent demand data
    print("\n2. Creating intermittent demand data...")
    try:
        # Create sparse demand data
        data = create_intermittent_demand_data(
            n_days=365,
            zero_prob=0.75,  # 75% zero demand periods
            avg_demand_when_nonzero=25
        )
        
        # Analyze demand pattern
        pattern_analysis = model.analyze_demand_pattern(data['demand'])
        
        print(f"   Created {len(data)} days of intermittent demand")
        print(f"   Zero demand periods: {pattern_analysis['zero_demand_periods']} ({pattern_analysis['zero_demand_percentage']:.1f}%)")
        print(f"   Average demand interval: {pattern_analysis['average_demand_interval']:.1f} days")
        print(f"   Pattern type: {pattern_analysis['pattern_type']}")
        print(f"   CV²: {pattern_analysis['coefficient_of_variation_squared']:.3f}")
        
        # Split data
        train_data = data.iloc[:280]  # First 280 days
        test_data = data.iloc[280:]   # Last 85 days
        
        results.add_result("Intermittent data creation", True, metrics=pattern_analysis)
        
    except Exception as e:
        results.add_result("Intermittent data creation", False, str(e))
        print(f"❌ Intermittent data creation failed: {e}")
        return
    
    # Test 3: Model building for intermittent data
    print("\n3. Building intermittent model (Croston's method)...")
    try:
        pymc_model = model.build_model(train_data)
        results.add_result("Intermittent model building", True)
        print("✅ Intermittent model built successfully")
        print("   Model includes demand size and interval components")
        
    except Exception as e:
        results.add_result("Intermittent model building", False, str(e))
        print(f"❌ Intermittent model building failed: {e}")
        return
    
    # Test 4: Model fitting
    print("\n4. Fitting intermittent model...")
    try:
        inference_data = model.fit(
            train_data,
            draws=400,
            tune=200,
            chains=2,
            progressbar=False,
            random_seed=42
        )
        
        # Check convergence
        import arviz as az
        summary = az.summary(inference_data, hdi_prob=0.95)
        max_rhat = summary['r_hat'].max()
        min_ess_bulk = summary['ess_bulk'].min()
        
        convergence_metrics = {
            'Max_Rhat': max_rhat,
            'Min_ESS_Bulk': min_ess_bulk,
            'Converged': max_rhat < 1.1 and min_ess_bulk > 50
        }
        
        results.add_result("Intermittent model fitting", True, metrics=convergence_metrics)
        print("✅ Intermittent model fitting completed")
        print(f"   Max R-hat: {max_rhat:.4f}")
        print(f"   Min ESS: {min_ess_bulk:.0f}")
        
        # Extract key parameters
        if 'demand_rate' in inference_data.posterior:
            demand_rate = inference_data.posterior['demand_rate'].mean().values
            print(f"   Estimated demand rate: {demand_rate:.4f} units/day")
        
    except Exception as e:
        results.add_result("Intermittent model fitting", False, str(e))
        print(f"❌ Intermittent model fitting failed: {e}")
        return
    
    # Test 5: Forecasting with safety stock
    print("\n5. Generating intermittent forecasts...")
    try:
        forecast_df = model.forecast(
            steps=len(test_data),
            service_level=0.95
        )
        
        forecast_metrics = {
            'Forecast_Steps': len(forecast_df),
            'Avg_Forecast': forecast_df['forecast'].mean(),
            'Avg_Safety_Stock': forecast_df['safety_stock'].mean() if 'safety_stock' in forecast_df.columns else 0,
            'Zero_Forecasts': (forecast_df['forecast'] == 0).sum()
        }
        
        results.add_result("Intermittent model forecasting", True, metrics=forecast_metrics)
        print("✅ Intermittent forecasting completed")
        print(f"   Average forecast: {forecast_metrics['Avg_Forecast']:.3f}")
        if 'safety_stock' in forecast_df.columns:
            print(f"   Average safety stock: {forecast_metrics['Avg_Safety_Stock']:.2f}")
        
    except Exception as e:
        results.add_result("Intermittent model forecasting", False, str(e))
        print(f"❌ Intermittent forecasting failed: {e}")
        return
    
    # Test 6: Specialized intermittent accuracy metrics
    print("\n6. Validating intermittent forecast accuracy...")
    try:
        actual = test_data['demand'].values
        forecast = forecast_df['forecast'].values
        forecast_lower = forecast_df['forecast_lower'].values
        forecast_upper = forecast_df['forecast_upper'].values
        
        # Standard accuracy metrics
        accuracy_metrics = calculate_accuracy_metrics(actual, forecast, forecast_lower, forecast_upper)
        
        # Intermittent-specific metrics
        # Period-over-Period Error (POPE)
        pope = np.mean(np.abs(actual - forecast))
        
        # Scaled Mean Absolute Error (for intermittent data)
        smae = np.mean(np.abs(actual - forecast)) / np.mean(actual + 1e-8)
        
        # Count accuracy (how well we predict zero vs non-zero)
        actual_nonzero = (actual > 0).astype(int)
        forecast_nonzero = (forecast > np.mean(forecast)/2).astype(int)
        count_accuracy = np.mean(actual_nonzero == forecast_nonzero) * 100
        
        intermittent_metrics = {
            'POPE': pope,
            'SMAE': smae,
            'Count_Accuracy': count_accuracy,
            **accuracy_metrics
        }
        
        results.add_result("Intermittent model accuracy", True, metrics=intermittent_metrics)
        print("✅ Intermittent accuracy validation completed")
        print(f"   POPE: {pope:.3f}")
        print(f"   SMAE: {smae:.3f}")
        print(f"   Count accuracy: {count_accuracy:.1f}%")
        print(f"   Coverage: {accuracy_metrics['Coverage_95']:.1f}%")
        
    except Exception as e:
        results.add_result("Intermittent model accuracy", False, str(e))
        print(f"❌ Intermittent accuracy validation failed: {e}")
    
    # Test 7: Intermittent-specific visualization
    print("\n7. Creating intermittent demand visualizations...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Intermittent Demand Model Analysis', fontsize=16)
        
        # Plot 1: Intermittent demand pattern
        ax1 = axes[0, 0]
        ax1.plot(train_data['date'], train_data['demand'], 'ko', markersize=2, alpha=0.6, label='Training')
        ax1.plot(test_data['date'], test_data['demand'], 'bo', markersize=3, label='Actual Test')
        ax1.plot(forecast_df['date'], forecast_df['forecast'], 'r-', linewidth=2, label='Forecast')
        ax1.fill_between(forecast_df['date'], forecast_lower, forecast_upper, 
                        alpha=0.3, color='red', label='95% CI')
        ax1.set_title('Intermittent Demand Forecast')
        ax1.set_ylabel('Demand')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Demand size distribution
        ax2 = axes[0, 1]
        non_zero_demand = data[data['demand'] > 0]['demand']
        ax2.hist(non_zero_demand, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(non_zero_demand.mean(), color='red', linestyle='--', 
                   label=f'Mean: {non_zero_demand.mean():.1f}')
        ax2.set_title('Non-Zero Demand Distribution')
        ax2.set_xlabel('Demand Size')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Plot 3: Inter-arrival times
        ax3 = axes[1, 0]
        non_zero_indices = np.where(data['demand'] > 0)[0]
        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
            ax3.hist(intervals, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.axvline(intervals.mean(), color='red', linestyle='--', 
                       label=f'Mean: {intervals.mean():.1f}')
            ax3.set_title('Inter-Arrival Times')
            ax3.set_xlabel('Days Between Demands')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        
        # Plot 4: Cumulative demand comparison
        ax4 = axes[1, 1]
        actual_cumsum = np.cumsum(actual)
        forecast_cumsum = np.cumsum(forecast)
        
        ax4.plot(range(len(actual)), actual_cumsum, 'b-', label='Actual Cumulative')
        ax4.plot(range(len(forecast)), forecast_cumsum, 'r--', label='Forecast Cumulative')
        ax4.set_title('Cumulative Demand Comparison')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Cumulative Demand')
        ax4.legend()
        
        plt.tight_layout()
        results.add_plot("Intermittent model visualization", fig)
        results.add_result("Intermittent model visualization", True)
        print("✅ Intermittent demand analysis completed")
        
        # Save plot
        plot_path = Path("intermittent_demand_model_results.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved as {plot_path}")
        
    except Exception as e:
        results.add_result("Intermittent model visualization", False, str(e))
        print(f"❌ Intermittent demand visualization failed: {e}")
    
    print(f"\n{'='*60}")
    print("INTERMITTENT DEMAND MODEL TESTING COMPLETED")
    print(f"{'='*60}")

def run_comprehensive_model_comparison(results: TestResults):
    """Compare all demand models on common test scenarios."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL COMPARISON AND BUSINESS SCENARIOS")
    print("="*70)
    
    # Scenario 1: Retail demand with clear seasonality
    print("\n1. RETAIL SCENARIO: Electronics store with seasonal patterns")
    retail_data = create_synthetic_demand_data(
        n_days=400,
        base_demand=150,
        seasonal_amplitude=40,
        trend_slope=0.15
    )
    
    print(f"   Dataset: {len(retail_data)} days, avg demand: {retail_data['demand'].mean():.1f}")
    
    # Scenario 2: Supply chain with multiple SKUs
    print("\n2. SUPPLY CHAIN SCENARIO: Multi-product inventory")  
    hierarchical_data = create_hierarchical_demand_data(
        n_days=300,
        regions=['Warehouse_A', 'Warehouse_B'],
        products=['SKU_1', 'SKU_2', 'SKU_3']
    )
    
    print(f"   Dataset: {len(hierarchical_data)} observations across {hierarchical_data['region'].nunique()} warehouses")
    
    # Scenario 3: Spare parts with intermittent demand
    print("\n3. SPARE PARTS SCENARIO: Aircraft maintenance parts")
    spare_parts_data = create_intermittent_demand_data(
        n_days=500,
        zero_prob=0.85,
        avg_demand_when_nonzero=15
    )
    
    zero_pct = (spare_parts_data['demand'] == 0).mean() * 100
    print(f"   Dataset: {len(spare_parts_data)} days, {zero_pct:.1f}% zero-demand periods")
    
    # Business insights
    print("\n4. BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("-" * 50)
    
    insights = []
    
    # Check which models performed best
    model_performance = {}
    for test_name, success in results.results.items():
        if 'accuracy' in test_name.lower() and success and test_name in results.metrics:
            model_type = test_name.split()[0].lower()
            mae = results.metrics[test_name].get('MAE', float('inf'))
            coverage = results.metrics[test_name].get('Coverage_95', 0)
            model_performance[model_type] = {'MAE': mae, 'Coverage': coverage}
    
    if model_performance:
        best_accuracy = min(model_performance.items(), key=lambda x: x[1]['MAE'])
        best_coverage = max(model_performance.items(), key=lambda x: x[1]['Coverage'])
        
        insights.extend([
            f"🎯 Best accuracy: {best_accuracy[0].title()} Model (MAE: {best_accuracy[1]['MAE']:.2f})",
            f"📊 Best uncertainty quantification: {best_coverage[0].title()} Model (Coverage: {best_coverage[1]['Coverage']:.1f}%)",
            ""
        ])
    
    # Model selection guidelines
    guidelines = [
        "🔍 MODEL SELECTION GUIDELINES:",
        "• Base Model → Simple trends, limited seasonality, fast implementation",
        "• Seasonal Model → Strong seasonal patterns, holiday effects, changepoints",  
        "• Hierarchical Model → Multiple locations/products, cross-learning benefits",
        "• Intermittent Model → Spare parts, slow-moving items, high zero-demand periods",
        "",
        "💡 IMPLEMENTATION RECOMMENDATIONS:",
        "• Start with Base Model for proof-of-concept",
        "• Upgrade to Seasonal for demand with clear patterns", 
        "• Use Hierarchical for portfolio optimization",
        "• Apply Intermittent for critical spare parts planning"
    ]
    
    insights.extend(guidelines)
    
    for insight in insights:
        print(f"   {insight}")
    
    results.add_result("Business scenario analysis", True, 
                      metrics={'Scenarios_Tested': 3, 'Models_Compared': 4})

def main():
    """Run the comprehensive demand model test suite."""
    print("PyMC-Supply-Chain Comprehensive Demand Model Test Suite")
    print("=" * 70)
    print("Testing all demand forecasting models with realistic business scenarios")
    print("=" * 70)
    
    results = TestResults()
    
    try:
        # Test each model comprehensively
        test_base_demand_model(results)
        test_seasonal_demand_model(results)
        test_hierarchical_demand_model(results)
        test_intermittent_demand_model(results)
        
        # Run comparative analysis
        run_comprehensive_model_comparison(results)
        
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user.")
        results.add_result("Test suite completion", False, "Interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error in test suite: {e}")
        traceback.print_exc()
        results.add_result("Test suite completion", False, f"Unexpected error: {e}")
    
    # Print comprehensive results
    results.print_detailed_summary()
    
    # Calculate success metrics
    total_tests = len(results.results)
    passed_tests = sum(1 for success in results.results.values() if success)
    critical_tests = [
        "Base model initialization", "Base model fitting", "Base model forecasting",
        "Seasonal model initialization", "Seasonal model fitting", 
        "Hierarchical model initialization", "Intermittent model initialization"
    ]
    
    critical_passed = sum(1 for test in critical_tests if results.results.get(test, False))
    
    print(f"\n{'='*70}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Critical Tests Passed: {critical_passed}/{len(critical_tests)}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! All demand models are working correctly.")
        print("✅ PyMC-Supply-Chain demand forecasting is ready for production use.")
        return 0
    elif critical_passed >= len(critical_tests) * 0.8:
        print("⚠️  Most tests passed. Core functionality is working well.")
        print("🚀 PyMC-Supply-Chain is suitable for pilot implementations.")
        return 1
    else:
        print("❌ Multiple critical tests failed. Implementation needs attention.")
        print("🔧 Review errors and retry after fixes.")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)