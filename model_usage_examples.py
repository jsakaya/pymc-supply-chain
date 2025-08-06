#!/usr/bin/env python3
"""
PyMC-Supply-Chain Demand Models: Usage Examples

Quick examples showing how to use each of the 4 demand models individually.
This serves as a practical reference for implementing the models in real projects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pymc_supply_chain.demand import (
    DemandForecastModel,
    SeasonalDemandModel,
    HierarchicalDemandModel,
    IntermittentDemandModel
)

def example_base_model():
    """Example: Basic demand forecasting with trend and seasonality."""
    print("🔬 Example: Base Demand Forecast Model")
    print("=" * 50)
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    demand = 100 + 0.05 * np.arange(300) + 20 * np.sin(2 * np.pi * np.arange(300) / 365) + np.random.normal(0, 5, 300)
    promotion = np.random.binomial(1, 0.1, 300) * 20
    demand += promotion
    
    data = pd.DataFrame({
        'date': dates,
        'demand': np.maximum(demand, 0),
        'promotion': promotion
    })
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Average demand: {data['demand'].mean():.2f}")
    
    # Initialize model
    model = DemandForecastModel(
        date_column='date',
        target_column='demand',
        include_trend=True,
        include_seasonality=True,
        external_regressors=['promotion']
    )
    
    # Fit model
    print("\n📈 Fitting model...")
    model.fit(data, draws=300, tune=300, chains=2, progressbar=False)
    
    # Generate forecast
    print("🔮 Generating 30-day forecast...")
    forecast = model.forecast(steps=30, frequency='D')
    
    print(f"Forecast shape: {forecast.shape}")
    print(f"Mean forecast: {forecast['forecast'].mean():.2f}")
    print(f"Forecast range: {forecast['forecast'].min():.2f} - {forecast['forecast'].max():.2f}")
    
    return model, data, forecast


def example_seasonal_model():
    """Example: Advanced seasonal modeling with Fourier components."""
    print("\n🔬 Example: Seasonal Demand Model")
    print("=" * 50)
    
    # Generate complex seasonal data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    t = np.arange(500)
    
    demand = (200 + 0.1 * t +  # trend
              50 * np.sin(2 * np.pi * t / 365) +  # yearly
              20 * np.sin(2 * np.pi * t / 7) +   # weekly
              np.random.normal(0, 10, 500))      # noise
    
    data = pd.DataFrame({
        'date': dates,
        'demand': np.maximum(demand, 0)
    })
    
    print(f"Data shape: {data.shape}")
    print(f"Strong seasonality with yearly and weekly patterns")
    
    # Initialize seasonal model
    model = SeasonalDemandModel(
        date_column='date',
        target_column='demand',
        yearly_seasonality=8,  # Fourier terms for yearly
        weekly_seasonality=3,  # Fourier terms for weekly
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5.0
    )
    
    # Fit model
    print("\n📈 Fitting seasonal model...")
    model.fit(data, draws=300, tune=300, chains=2, progressbar=False)
    
    # Generate forecast
    print("🔮 Generating 60-day forecast...")
    forecast = model.forecast(steps=60, frequency='D')
    
    print(f"Forecast captures complex seasonality")
    print(f"Forecast uncertainty: {forecast['forecast_std'].mean():.2f}")
    
    return model, data, forecast


def example_hierarchical_model():
    """Example: Multi-location hierarchical forecasting."""
    print("\n🔬 Example: Hierarchical Demand Model")
    print("=" * 50)
    
    # Generate hierarchical data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    regions = ['East', 'West']
    stores = ['Store1', 'Store2']
    
    data = []
    for region in regions:
        for store in stores:
            region_effect = 1.2 if region == 'East' else 0.9
            store_effect = 1.1 if store == 'Store1' else 0.95
            
            base_demand = 60 * region_effect * store_effect
            trend = 0.02 * np.arange(200)
            seasonality = 10 * np.sin(2 * np.pi * np.arange(200) / 365)
            noise = np.random.normal(0, 3, 200)
            
            demand = base_demand + trend + seasonality + noise
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'region': region,
                    'store': store,
                    'demand': max(0, demand[i])
                })
    
    data = pd.DataFrame(data)
    
    # Test on single region-store combination
    subset = data[(data['region'] == 'East') & (data['store'] == 'Store1')].copy()
    
    print(f"Full hierarchical data: {len(data)} observations")
    print(f"Testing subset: {len(subset)} observations")
    print(f"Hierarchy: {len(regions)} regions × {len(stores)} stores")
    
    # Initialize hierarchical model
    model = HierarchicalDemandModel(
        hierarchy_cols=['region'],
        date_column='date',
        target_column='demand',
        pooling_strength=0.3  # Partial pooling
    )
    
    # Fit model
    print("\n📈 Fitting hierarchical model...")
    model.fit(subset, draws=300, tune=300, chains=2, progressbar=False)
    
    # Generate forecast
    print("🔮 Generating 30-day forecast...")
    forecast = model.forecast(steps=30, frequency='D')
    
    print(f"Benefits from hierarchical information sharing")
    print(f"More stable forecasts through partial pooling")
    
    return model, subset, forecast, data


def example_intermittent_model():
    """Example: Sparse/intermittent demand forecasting."""
    print("\n🔬 Example: Intermittent Demand Model")
    print("=" * 50)
    
    # Generate intermittent data
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    demand = np.zeros(300)
    
    # Only 15% of days have demand
    demand_days = np.random.choice(300, size=int(300 * 0.15), replace=False)
    demand[demand_days] = np.random.gamma(2, 10, len(demand_days))  # When demand occurs, it's significant
    
    data = pd.DataFrame({
        'date': dates,
        'demand': demand
    })
    
    print(f"Data shape: {data.shape}")
    print(f"Zero demand periods: {np.sum(demand == 0)} ({np.sum(demand == 0)/len(demand)*100:.1f}%)")
    print(f"Non-zero demand events: {np.sum(demand > 0)}")
    print(f"Average demand when non-zero: {demand[demand > 0].mean():.2f}")
    
    # Initialize intermittent model
    model = IntermittentDemandModel(
        date_column='date',
        target_column='demand',
        method='croston'  # Croston's method for intermittent demand
    )
    
    # Analyze demand pattern first
    pattern = model.analyze_demand_pattern(data['demand'])
    print(f"\n🔍 Demand Pattern Analysis:")
    print(f"  Pattern Type: {pattern['pattern_type']}")
    print(f"  Average Demand Interval: {pattern['average_demand_interval']:.1f} days")
    print(f"  Coefficient of Variation²: {pattern['coefficient_of_variation_squared']:.3f}")
    
    # Fit model
    print("\n📈 Fitting intermittent model...")
    model.fit(data, draws=300, tune=300, chains=2, progressbar=False)
    
    # Generate forecast
    print("🔮 Generating 30-day forecast with safety stock...")
    forecast = model.forecast(steps=30, frequency='D', service_level=0.95)
    
    print(f"Includes safety stock calculations")
    print(f"Appropriate for spare parts and slow-moving items")
    if 'demand_rate' in forecast.columns:
        print(f"Expected demand rate: {forecast['demand_rate'].iloc[0]:.3f}")
    
    return model, data, forecast


def plot_example_results(base_result, seasonal_result, hierarchical_result, intermittent_result):
    """Create a summary plot of all example results."""
    print("\n🎨 Creating summary visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PyMC-Supply-Chain Models: Usage Examples', fontsize=16, fontweight='bold')
    
    examples = [
        (base_result, 'Base Model', 0, 0),
        (seasonal_result, 'Seasonal Model', 0, 1),
        (hierarchical_result, 'Hierarchical Model', 1, 0),
        (intermittent_result, 'Intermittent Model', 1, 1)
    ]
    
    for (model, data, forecast), title, row, col in examples:
        ax = axes[row, col]
        
        # Plot historical data (last 100 points)
        if len(data) > 100:
            plot_data = data.tail(100)
        else:
            plot_data = data
            
        ax.plot(plot_data['date'], plot_data['demand'], 'o-', alpha=0.7, 
               markersize=2, label='Historical', color='gray')
        
        # Plot forecast
        if hasattr(forecast, 'index'):
            # Handle case where forecast might not have date column
            forecast_dates = pd.date_range(
                start=data['date'].iloc[-1] + pd.Timedelta(days=1),
                periods=len(forecast),
                freq='D'
            )
        else:
            forecast_dates = forecast.get('date', 
                pd.date_range(start=data['date'].iloc[-1] + pd.Timedelta(days=1),
                            periods=len(forecast), freq='D'))
        
        forecast_values = forecast.get('forecast', forecast)
        if hasattr(forecast_values, 'values'):
            forecast_values = forecast_values.values
        
        ax.plot(forecast_dates, forecast_values, 'o-', color='red', 
               linewidth=2, markersize=3, label='Forecast')
        
        # Add uncertainty bands if available
        if isinstance(forecast, pd.DataFrame):
            if 'forecast_lower' in forecast.columns:
                ax.fill_between(forecast_dates,
                              forecast['forecast_lower'],
                              forecast['forecast_upper'],
                              alpha=0.3, color='red', label='95% CI')
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Demand')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Rotate dates
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/sakaya/projects/pymc-marketing/pymc-supply-chain/usage_examples.png', 
               dpi=300, bbox_inches='tight')
    print("✅ Saved usage_examples.png")
    plt.show()


def main():
    """Run all usage examples."""
    print("🚀 PyMC-Supply-Chain Demand Models: Usage Examples")
    print("=" * 70)
    
    try:
        # Run examples
        base_result = example_base_model()
        seasonal_result = example_seasonal_model()
        hierarchical_result = example_hierarchical_model()
        intermittent_result = example_intermittent_model()
        
        # Unpack hierarchical result which returns 4 items
        hier_model, hier_data, hier_forecast, hier_full = hierarchical_result
        hierarchical_result = (hier_model, hier_data, hier_forecast)
        
        # Create summary plot
        plot_example_results(base_result, seasonal_result, 
                           hierarchical_result, intermittent_result)
        
        print("\n🎉 All usage examples completed successfully!")
        print("\n📚 Key Takeaways:")
        print("  • Base Model: Best for regular demand with simple patterns")
        print("  • Seasonal Model: Handles complex seasonality with Fourier components")
        print("  • Hierarchical Model: Leverages information across business units")
        print("  • Intermittent Model: Specialized for sparse/sporadic demand")
        print("\n💡 Choose the model that best matches your data characteristics!")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        raise


if __name__ == "__main__":
    main()