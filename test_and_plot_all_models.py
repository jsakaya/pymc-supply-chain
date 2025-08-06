#!/usr/bin/env python3
"""
Comprehensive Test and Visualization Script for PyMC-Supply-Chain Demand Models

This script demonstrates all 4 demand forecasting models with realistic synthetic data,
comprehensive evaluation metrics, and beautiful visualizations.

Models tested:
1. DemandForecastModel - Regular demand with trend/seasonality
2. SeasonalDemandModel - Strong seasonal patterns with Fourier components
3. HierarchicalDemandModel - Multi-location hierarchical data
4. IntermittentDemandModel - Sparse/intermittent demand patterns

Features:
- Realistic synthetic data generation
- 80/20 train/test splits
- Multiple accuracy metrics (MAE, RMSE, MAPE, WAPE)
- Beautiful visualizations with uncertainty bands
- Model component decomposition
- Performance comparison table
- High-quality PNG exports
"""

import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", message=".*convergence.*")
warnings.filterwarnings("ignore", message=".*Rhat.*")
warnings.filterwarnings("ignore", message=".*effective sample size.*")

try:
    from pymc_supply_chain.demand import (
        DemandForecastModel,
        SeasonalDemandModel, 
        HierarchicalDemandModel,
        IntermittentDemandModel
    )
    logger.info("✅ Successfully imported all demand models")
except ImportError as e:
    logger.error(f"❌ Failed to import models: {e}")
    sys.exit(1)

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class ModelTester:
    """Comprehensive tester for all demand models."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(random_seed)
        self.results = {}
        self.metrics = {}
        
    def generate_regular_demand_data(
        self, 
        n_periods: int = 365,
        start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate realistic demand data with trend and seasonality."""
        logger.info("📊 Generating regular demand data...")
        
        dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
        t = np.arange(n_periods)
        
        # Base level
        base_demand = 100
        
        # Trend (slight upward)
        trend = 0.05 * t
        
        # Seasonal patterns
        yearly_seasonality = 20 * np.sin(2 * np.pi * t / 365.25)
        weekly_seasonality = 5 * np.sin(2 * np.pi * t / 7)
        
        # Random noise
        noise = np.random.normal(0, 8, n_periods)
        
        # External regressor (promotion effect)
        promotion = np.random.binomial(1, 0.1, n_periods) * 30
        
        # Combine all components
        demand = base_demand + trend + yearly_seasonality + weekly_seasonality + promotion + noise
        demand = np.maximum(demand, 0)  # Ensure non-negative
        
        return pd.DataFrame({
            'date': dates,
            'demand': demand,
            'promotion': promotion
        })
    
    def generate_seasonal_demand_data(
        self,
        n_periods: int = 730,  # 2 years
        start_date: str = "2022-01-01"
    ) -> pd.DataFrame:
        """Generate data with strong seasonal patterns."""
        logger.info("📊 Generating seasonal demand data...")
        
        dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
        t = np.arange(n_periods)
        
        # Base level
        base_demand = 200
        
        # Strong trend with changepoints
        trend = 0.08 * t
        # Add some changepoints
        changepoint_1 = 200
        changepoint_2 = 500
        trend += np.where(t > changepoint_1, 0.03 * (t - changepoint_1), 0)
        trend += np.where(t > changepoint_2, -0.02 * (t - changepoint_2), 0)
        
        # Complex seasonality
        yearly_main = 50 * np.sin(2 * np.pi * t / 365.25)
        yearly_harmonic = 15 * np.sin(4 * np.pi * t / 365.25)
        weekly = 20 * np.sin(2 * np.pi * t / 7)
        weekend_effect = 10 * np.sin(2 * np.pi * t / 7 + np.pi/2)
        
        # Holiday effects (simulate Christmas, Summer vacation, etc.)
        holidays = np.zeros(n_periods)
        christmas_periods = [d for d in range(n_periods) if (dates[d].month == 12 and dates[d].day in range(20, 32))]
        summer_periods = [d for d in range(n_periods) if (dates[d].month in [7, 8])]
        
        holidays[christmas_periods] = 80
        holidays[summer_periods] = 30
        
        # Random noise
        noise = np.random.normal(0, 12, n_periods)
        
        # Combine components
        demand = base_demand + trend + yearly_main + yearly_harmonic + weekly + weekend_effect + holidays + noise
        demand = np.maximum(demand, 0)
        
        return pd.DataFrame({
            'date': dates,
            'demand': demand
        })
    
    def generate_hierarchical_data(
        self,
        n_periods: int = 365,
        start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate multi-location hierarchical demand data."""
        logger.info("📊 Generating hierarchical demand data...")
        
        dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
        
        # Hierarchy: 3 regions, 2 stores per region, 2 products per store
        regions = ['North', 'South', 'West']
        stores = ['Store_A', 'Store_B']  
        products = ['Product_X', 'Product_Y']
        
        data = []
        
        # Generate hierarchical effects
        region_effects = {'North': 1.2, 'South': 0.9, 'West': 1.1}
        store_effects = {'Store_A': 1.1, 'Store_B': 0.95}
        product_effects = {'Product_X': 1.3, 'Product_Y': 0.8}
        
        t = np.arange(n_periods)
        
        for region in regions:
            for store in stores:
                for product in products:
                    # Base demand
                    base = 50
                    
                    # Hierarchy multipliers
                    region_mult = region_effects[region]
                    store_mult = store_effects[store]
                    product_mult = product_effects[product]
                    
                    # Trend (varies by location)
                    trend = 0.02 * t * region_mult
                    
                    # Seasonality (shared but with different amplitudes)
                    seasonality = (15 * region_mult * np.sin(2 * np.pi * t / 365.25) +
                                  5 * np.sin(2 * np.pi * t / 7))
                    
                    # Location-specific noise
                    noise = np.random.normal(0, 3, n_periods) * store_mult
                    
                    # Combine
                    demand = base * region_mult * store_mult * product_mult + trend + seasonality + noise
                    demand = np.maximum(demand, 0)
                    
                    for i, date in enumerate(dates):
                        data.append({
                            'date': date,
                            'region': region,
                            'store': store,
                            'product': product,
                            'demand': demand[i]
                        })
        
        return pd.DataFrame(data)
    
    def generate_intermittent_data(
        self,
        n_periods: int = 365,
        start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate sparse/intermittent demand data."""
        logger.info("📊 Generating intermittent demand data...")
        
        dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
        
        # Intermittent demand parameters
        demand_probability = 0.15  # 15% chance of demand on any given day
        average_demand_size = 25
        demand_variability = 8
        
        demand = np.zeros(n_periods)
        
        # Generate demand events
        for i in range(n_periods):
            if np.random.random() < demand_probability:
                # Non-zero demand
                demand_size = np.random.gamma(
                    shape=(average_demand_size / demand_variability) ** 2,
                    scale=demand_variability ** 2 / average_demand_size
                )
                demand[i] = max(1, demand_size)
            # else: demand remains 0
        
        # Add some seasonality to demand probability
        t = np.arange(n_periods)
        seasonal_prob_adj = 1 + 0.3 * np.sin(2 * np.pi * t / 365.25)  # Higher in winter
        
        # Apply seasonal adjustment
        for i in range(n_periods):
            if demand[i] == 0 and np.random.random() < demand_probability * (seasonal_prob_adj[i] - 1):
                demand_size = np.random.gamma(
                    shape=(average_demand_size / demand_variability) ** 2,
                    scale=demand_variability ** 2 / average_demand_size
                )
                demand[i] = max(1, demand_size)
        
        return pd.DataFrame({
            'date': dates,
            'demand': demand
        })
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/test sets."""
        split_idx = int(len(df) * train_ratio)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "WAPE": np.nan}
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # MAPE: avoid division by zero
        actual_nonzero = actual[actual != 0]
        predicted_nonzero = predicted[actual != 0]
        if len(actual_nonzero) > 0:
            mape = np.mean(np.abs((actual_nonzero - predicted_nonzero) / actual_nonzero)) * 100
        else:
            mape = np.nan
            
        # WAPE: Weighted Absolute Percentage Error
        if np.sum(actual) != 0:
            wape = np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100
        else:
            wape = np.nan
        
        return {
            "MAE": mae,
            "RMSE": rmse, 
            "MAPE": mape,
            "WAPE": wape
        }
    
    def test_base_demand_model(self) -> Dict[str, Any]:
        """Test the base DemandForecastModel."""
        logger.info("🔬 Testing DemandForecastModel...")
        
        try:
            # Generate data
            data = self.generate_regular_demand_data()
            train_data, test_data = self.split_data(data)
            
            # Initialize and fit model
            model = DemandForecastModel(
                date_column="date",
                target_column="demand",
                include_trend=True,
                include_seasonality=True,
                external_regressors=["promotion"]
            )
            
            logger.info("  Fitting model...")
            model.fit(train_data, progressbar=False, draws=500, tune=500, chains=2)
            
            # Generate forecasts
            logger.info("  Generating forecasts...")
            forecast = model.forecast(steps=len(test_data), frequency='D')
            
            # Calculate metrics
            metrics = self.calculate_metrics(test_data['demand'].values, forecast['forecast'].values)
            
            return {
                'model': model,
                'train_data': train_data,
                'test_data': test_data,
                'forecast': forecast,
                'metrics': metrics,
                'model_type': 'Base Demand Model'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in base demand model: {e}")
            return {'error': str(e)}
    
    def test_seasonal_model(self) -> Dict[str, Any]:
        """Test the SeasonalDemandModel."""
        logger.info("🔬 Testing SeasonalDemandModel...")
        
        try:
            # Generate data
            data = self.generate_seasonal_demand_data()
            train_data, test_data = self.split_data(data)
            
            # Initialize model
            model = SeasonalDemandModel(
                date_column="date",
                target_column="demand", 
                yearly_seasonality=10,
                weekly_seasonality=3,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=5.0
            )
            
            logger.info("  Fitting model...")
            model.fit(train_data, progressbar=False, draws=500, tune=500, chains=2)
            
            # Generate forecasts
            logger.info("  Generating forecasts...")
            forecast = model.forecast(steps=len(test_data), frequency='D')
            
            # Calculate metrics
            metrics = self.calculate_metrics(test_data['demand'].values, forecast['forecast'].values)
            
            return {
                'model': model,
                'train_data': train_data,
                'test_data': test_data,
                'forecast': forecast,
                'metrics': metrics,
                'model_type': 'Seasonal Model'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in seasonal model: {e}")
            return {'error': str(e)}
    
    def test_hierarchical_model(self) -> Dict[str, Any]:
        """Test the HierarchicalDemandModel."""
        logger.info("🔬 Testing HierarchicalDemandModel...")
        
        try:
            # Generate data
            data = self.generate_hierarchical_data()
            
            # For simplicity, test on one product-store combination
            subset = data[(data['region'] == 'North') & 
                         (data['store'] == 'Store_A') & 
                         (data['product'] == 'Product_X')].copy()
            
            train_data, test_data = self.split_data(subset)
            
            # Initialize model (simplified hierarchy for demo)
            model = HierarchicalDemandModel(
                hierarchy_cols=['region'],
                date_column="date",
                target_column="demand",
                pooling_strength=0.3
            )
            
            logger.info("  Fitting model...")
            model.fit(train_data, progressbar=False, draws=500, tune=500, chains=2)
            
            # Generate forecasts
            logger.info("  Generating forecasts...")
            forecast = model.forecast(steps=len(test_data), frequency='D')
            
            # Calculate metrics
            metrics = self.calculate_metrics(test_data['demand'].values, forecast['forecast'].values)
            
            return {
                'model': model,
                'train_data': train_data,
                'test_data': test_data,
                'forecast': forecast,
                'metrics': metrics,
                'model_type': 'Hierarchical Model',
                'full_data': data
            }
            
        except Exception as e:
            logger.error(f"❌ Error in hierarchical model: {e}")
            return {'error': str(e)}
    
    def test_intermittent_model(self) -> Dict[str, Any]:
        """Test the IntermittentDemandModel."""
        logger.info("🔬 Testing IntermittentDemandModel...")
        
        try:
            # Generate data
            data = self.generate_intermittent_data()
            train_data, test_data = self.split_data(data)
            
            # Initialize model
            model = IntermittentDemandModel(
                date_column="date",
                target_column="demand",
                method="croston"
            )
            
            # Analyze demand pattern
            pattern_analysis = model.analyze_demand_pattern(train_data['demand'])
            logger.info(f"  Demand pattern: {pattern_analysis['pattern_type']}")
            logger.info(f"  Zero demand percentage: {pattern_analysis['zero_demand_percentage']:.1f}%")
            
            logger.info("  Fitting model...")
            model.fit(train_data, progressbar=False, draws=500, tune=500, chains=2)
            
            # Generate forecasts
            logger.info("  Generating forecasts...")
            forecast = model.forecast(steps=len(test_data), frequency='D')
            
            # Calculate metrics
            metrics = self.calculate_metrics(test_data['demand'].values, forecast['forecast'].values)
            
            return {
                'model': model,
                'train_data': train_data,
                'test_data': test_data,
                'forecast': forecast,
                'metrics': metrics,
                'model_type': 'Intermittent Model',
                'pattern_analysis': pattern_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Error in intermittent model: {e}")
            return {'error': str(e)}
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization of all models."""
        logger.info("🎨 Creating comprehensive visualizations...")
        
        # Create figure with subplots for all models
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('PyMC-Supply-Chain Demand Models: Comprehensive Performance Analysis', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        model_names = ['base', 'seasonal', 'hierarchical', 'intermittent']
        titles = ['Base Demand Model', 'Seasonal Demand Model', 
                 'Hierarchical Demand Model', 'Intermittent Demand Model']
        
        for idx, (model_name, title) in enumerate(zip(model_names, titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            result = self.results.get(model_name)
            if result is None or 'error' in result:
                ax.text(0.5, 0.5, f"❌ Error in {title}", 
                       ha='center', va='center', fontsize=14, color='red',
                       transform=ax.transAxes)
                ax.set_title(title, fontsize=14, fontweight='bold')
                continue
            
            # Plot training data
            train_data = result['train_data']
            test_data = result['test_data'] 
            forecast = result['forecast']
            metrics = result['metrics']
            
            # Plot historical data
            ax.plot(train_data['date'], train_data['demand'], 
                   'o-', alpha=0.7, markersize=2, linewidth=1,
                   label='Training Data', color='gray')
            
            # Plot test data
            ax.plot(test_data['date'], test_data['demand'],
                   'o-', alpha=0.8, markersize=3, linewidth=1.5,
                   label='Test Data (Actual)', color='black')
            
            # Plot forecast
            if 'date' in forecast.columns:
                forecast_dates = forecast['date']
            else:
                # Create dates for forecast
                last_date = test_data['date'].iloc[-1] if len(test_data) > 0 else train_data['date'].iloc[-1]
                forecast_dates = pd.date_range(start=last_date, periods=len(forecast)+1, freq='D')[1:]
                
            ax.plot(forecast_dates, forecast['forecast'],
                   'o-', linewidth=2, markersize=3,
                   label='Forecast', color='red')
            
            # Plot uncertainty bands
            if 'forecast_lower' in forecast.columns and 'forecast_upper' in forecast.columns:
                ax.fill_between(forecast_dates, 
                              forecast['forecast_lower'],
                              forecast['forecast_upper'],
                              alpha=0.3, color='red', label='95% Confidence Interval')
            
            # Formatting
            ax.set_title(f'{title}\n' + 
                        f'MAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}, MAPE: {metrics.get("MAPE", np.nan):.1f}%',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Demand')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the plot
        plt.savefig('/Users/sakaya/projects/pymc-marketing/pymc-supply-chain/comprehensive_model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        logger.info("✅ Saved comprehensive comparison plot")
        
        plt.show()
    
    def create_metrics_comparison_table(self):
        """Create a comparison table of model performance metrics."""
        logger.info("📊 Creating metrics comparison table...")
        
        # Collect metrics from all models
        metrics_data = []
        
        for model_name in ['base', 'seasonal', 'hierarchical', 'intermittent']:
            result = self.results.get(model_name)
            if result and 'error' not in result:
                metrics = result['metrics']
                metrics_data.append({
                    'Model': result['model_type'],
                    'MAE': f"{metrics['MAE']:.2f}",
                    'RMSE': f"{metrics['RMSE']:.2f}",
                    'MAPE': f"{metrics.get('MAPE', np.nan):.1f}%" if not np.isnan(metrics.get('MAPE', np.nan)) else 'N/A',
                    'WAPE': f"{metrics.get('WAPE', np.nan):.1f}%" if not np.isnan(metrics.get('WAPE', np.nan)) else 'N/A'
                })
            else:
                metrics_data.append({
                    'Model': model_name.title() + ' Model',
                    'MAE': 'Error',
                    'RMSE': 'Error', 
                    'MAPE': 'Error',
                    'WAPE': 'Error'
                })
        
        # Create table
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create a figure for the table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df_metrics.values, 
                        colLabels=df_metrics.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Header styling
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E1E1E1')
            else:
                cell.set_facecolor('#F7F7F7')
        
        plt.title('Model Performance Comparison\nAccuracy Metrics on Test Data', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add explanatory text
        plt.figtext(0.5, 0.02, 
                   'MAE: Mean Absolute Error | RMSE: Root Mean Square Error | MAPE: Mean Absolute Percentage Error | WAPE: Weighted Absolute Percentage Error',
                   ha='center', fontsize=10, style='italic')
        
        plt.savefig('/Users/sakaya/projects/pymc-marketing/pymc-supply-chain/model_metrics_comparison.png',
                   dpi=300, bbox_inches='tight')
        logger.info("✅ Saved metrics comparison table")
        
        plt.show()
        
        return df_metrics
    
    def create_detailed_analysis_plots(self):
        """Create detailed individual model analysis plots."""
        logger.info("🎨 Creating detailed analysis plots...")
        
        # Create seasonal components plot for seasonal model
        if 'seasonal' in self.results and 'error' not in self.results['seasonal']:
            try:
                seasonal_result = self.results['seasonal']
                seasonal_model = seasonal_result['model']
                train_data = seasonal_result['train_data']
                
                fig, axes = seasonal_model.plot_components(train_data)
                plt.suptitle('Seasonal Model: Component Decomposition', fontsize=16, fontweight='bold')
                plt.savefig('/Users/sakaya/projects/pymc-marketing/pymc-supply-chain/seasonal_model_components.png',
                           dpi=300, bbox_inches='tight')
                logger.info("✅ Saved seasonal model components plot")
                plt.show()
            except Exception as e:
                logger.warning(f"⚠️ Could not create seasonal components plot: {e}")
        
        # Create hierarchical data visualization
        if 'hierarchical' in self.results and 'error' not in self.results['hierarchical']:
            try:
                hier_result = self.results['hierarchical']
                full_data = hier_result.get('full_data')
                
                if full_data is not None:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle('Hierarchical Demand Analysis', fontsize=16, fontweight='bold')
                    
                    # By region
                    region_data = full_data.groupby(['date', 'region'])['demand'].sum().unstack()
                    region_data.plot(ax=axes[0,0], title='Demand by Region', alpha=0.8)
                    
                    # By store
                    store_data = full_data.groupby(['date', 'store'])['demand'].sum().unstack()
                    store_data.plot(ax=axes[0,1], title='Demand by Store', alpha=0.8)
                    
                    # By product
                    product_data = full_data.groupby(['date', 'product'])['demand'].sum().unstack()
                    product_data.plot(ax=axes[1,0], title='Demand by Product', alpha=0.8)
                    
                    # Total demand
                    total_data = full_data.groupby('date')['demand'].sum()
                    total_data.plot(ax=axes[1,1], title='Total Demand', alpha=0.8, color='purple')
                    
                    for ax in axes.flat:
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                    
                    plt.tight_layout()
                    plt.savefig('/Users/sakaya/projects/pymc-marketing/pymc-supply-chain/hierarchical_analysis.png',
                               dpi=300, bbox_inches='tight')
                    logger.info("✅ Saved hierarchical analysis plot")
                    plt.show()
            except Exception as e:
                logger.warning(f"⚠️ Could not create hierarchical analysis plot: {e}")
        
        # Create intermittent demand pattern analysis
        if 'intermittent' in self.results and 'error' not in self.results['intermittent']:
            try:
                inter_result = self.results['intermittent']
                pattern_analysis = inter_result.get('pattern_analysis', {})
                train_data = inter_result['train_data']
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                fig.suptitle('Intermittent Demand Pattern Analysis', fontsize=16, fontweight='bold')
                
                # Demand time series
                axes[0,0].plot(train_data['date'], train_data['demand'], 'o-', alpha=0.7, markersize=2)
                axes[0,0].set_title('Intermittent Demand Time Series')
                axes[0,0].set_ylabel('Demand')
                axes[0,0].grid(True, alpha=0.3)
                
                # Demand distribution
                non_zero_demand = train_data[train_data['demand'] > 0]['demand']
                if len(non_zero_demand) > 0:
                    axes[0,1].hist(non_zero_demand, bins=20, alpha=0.7, color='orange')
                    axes[0,1].set_title('Non-Zero Demand Distribution')
                    axes[0,1].set_xlabel('Demand Size')
                    axes[0,1].set_ylabel('Frequency')
                    axes[0,1].grid(True, alpha=0.3)
                
                # Pattern characteristics
                axes[1,0].axis('off')
                pattern_text = f"""
Pattern Analysis:
• Pattern Type: {pattern_analysis.get('pattern_type', 'Unknown')}
• Zero Demand Periods: {pattern_analysis.get('zero_demand_periods', 0)}
• Zero Demand %: {pattern_analysis.get('zero_demand_percentage', 0):.1f}%
• Avg Demand Interval: {pattern_analysis.get('average_demand_interval', 0):.1f}
• Avg Demand Size: {pattern_analysis.get('average_demand_size', 0):.2f}
• CV²: {pattern_analysis.get('coefficient_of_variation_squared', 0):.3f}
                """
                axes[1,0].text(0.1, 0.9, pattern_text, transform=axes[1,0].transAxes, 
                              fontsize=12, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
                # Inter-arrival time analysis
                demand_dates = train_data[train_data['demand'] > 0]['date']
                if len(demand_dates) > 1:
                    inter_arrival = np.diff(demand_dates).astype('timedelta64[D]').astype(int)
                    axes[1,1].hist(inter_arrival, bins=15, alpha=0.7, color='green')
                    axes[1,1].set_title('Inter-Arrival Times Distribution')
                    axes[1,1].set_xlabel('Days Between Demand Events')
                    axes[1,1].set_ylabel('Frequency')
                    axes[1,1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('/Users/sakaya/projects/pymc-marketing/pymc-supply-chain/intermittent_analysis.png',
                           dpi=300, bbox_inches='tight')
                logger.info("✅ Saved intermittent analysis plot")
                plt.show()
            except Exception as e:
                logger.warning(f"⚠️ Could not create intermittent analysis plot: {e}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all models."""
        logger.info("🚀 Starting comprehensive model testing...")
        
        # Test all models
        self.results['base'] = self.test_base_demand_model()
        self.results['seasonal'] = self.test_seasonal_model()
        self.results['hierarchical'] = self.test_hierarchical_model()
        self.results['intermittent'] = self.test_intermittent_model()
        
        # Create visualizations
        logger.info("📊 Creating visualizations...")
        self.create_comprehensive_plots()
        
        # Create metrics table
        metrics_df = self.create_metrics_comparison_table()
        
        # Create detailed analysis plots
        self.create_detailed_analysis_plots()
        
        # Print summary
        self.print_summary()
        
        logger.info("✅ Comprehensive testing completed!")
        return self.results, metrics_df
    
    def print_summary(self):
        """Print a comprehensive summary of results."""
        print("\n" + "="*80)
        print("🎯 PYMC-SUPPLY-CHAIN DEMAND MODELS - COMPREHENSIVE TESTING RESULTS")
        print("="*80)
        
        success_count = 0
        
        for model_name, result in self.results.items():
            print(f"\n📊 {model_name.upper()} MODEL:")
            print("-" * 40)
            
            if result and 'error' not in result:
                success_count += 1
                metrics = result['metrics']
                print(f"✅ Status: SUCCESS")
                print(f"📈 MAE:   {metrics['MAE']:.3f}")
                print(f"📈 RMSE:  {metrics['RMSE']:.3f}")
                print(f"📈 MAPE:  {metrics.get('MAPE', np.nan):.2f}%" if not np.isnan(metrics.get('MAPE', np.nan)) else "📈 MAPE:  N/A")
                print(f"📈 WAPE:  {metrics.get('WAPE', np.nan):.2f}%" if not np.isnan(metrics.get('WAPE', np.nan)) else "📈 WAPE:  N/A")
                
                if 'pattern_analysis' in result:
                    pattern = result['pattern_analysis']
                    print(f"🔍 Pattern: {pattern.get('pattern_type', 'Unknown')}")
                    print(f"🔍 Zero Demand: {pattern.get('zero_demand_percentage', 0):.1f}%")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                print(f"❌ Status: FAILED")
                print(f"❗ Error: {error}")
        
        print(f"\n🏆 OVERALL RESULTS:")
        print(f"✅ Successful Models: {success_count}/4")
        print(f"📁 Plots Saved:")
        print(f"   • comprehensive_model_comparison.png")
        print(f"   • model_metrics_comparison.png")
        print(f"   • seasonal_model_components.png")
        print(f"   • hierarchical_analysis.png")
        print(f"   • intermittent_analysis.png")
        
        print("\n" + "="*80)


def main():
    """Main execution function."""
    print("🚀 PyMC-Supply-Chain Comprehensive Model Testing")
    print("=" * 60)
    
    try:
        # Initialize tester
        tester = ModelTester(random_seed=42)
        
        # Run comprehensive tests
        results, metrics_df = tester.run_comprehensive_test()
        
        print("\n🎉 Testing completed successfully!")
        print("Check the generated PNG files for detailed visualizations.")
        
        return results, metrics_df
        
    except Exception as e:
        logger.error(f"❌ Fatal error during testing: {e}")
        raise


if __name__ == "__main__":
    results, metrics = main()