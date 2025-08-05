"""
PyMC-Supply-Chain Quick Start Example

This example demonstrates end-to-end supply chain optimization:
1. Demand forecasting with uncertainty
2. Safety stock optimization
3. Facility location planning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate example data
np.random.seed(42)

# 1. Create demand data with trend and seasonality
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
trend = np.linspace(100, 120, 365)
seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 5, 365)
demand = trend + seasonality + noise

demand_df = pd.DataFrame({
    'date': dates,
    'demand': np.maximum(0, demand)
})

print("=== DEMAND FORECASTING ===")
from pymc_supply_chain.demand import DemandForecastModel

# Fit demand model
demand_model = DemandForecastModel(
    date_column='date',
    target_column='demand',
    include_trend=True,
    include_seasonality=True,
    seasonality=7  # Weekly seasonality
)

print("Fitting demand model...")
demand_model.fit(demand_df, progressbar=False)

# Generate forecast
forecast = demand_model.forecast(steps=30)
print(f"\nNext 30 days forecast:")
print(f"Average daily demand: {forecast['forecast'].mean():.1f}")
print(f"95% CI: [{forecast['forecast_lower'].mean():.1f}, {forecast['forecast_upper'].mean():.1f}]")

# Plot forecast
fig, ax = plt.subplots(figsize=(12, 6))
demand_model.plot_forecast(forecast, demand_df, title="30-Day Demand Forecast")
plt.tight_layout()
plt.savefig('demand_forecast.png')
plt.close()

print("\n=== INVENTORY OPTIMIZATION ===")
from pymc_supply_chain.inventory import SafetyStockOptimizer

# Create demand and lead time data
inventory_data = pd.DataFrame({
    'demand': np.random.normal(100, 20, 100),
    'lead_time': np.random.gamma(2, 2, 100)  # Variable lead times
})

# Optimize safety stock
safety_stock_opt = SafetyStockOptimizer(
    holding_cost=2.0,
    stockout_cost=50.0,
    target_service_level=0.95,
    lead_time_distribution='gamma',
    demand_distribution='normal'
)

print("Calculating optimal safety stock...")
safety_stock_opt.fit(inventory_data, progressbar=False)
safety_stock_result = safety_stock_opt.calculate_safety_stock()

print(f"\nOptimal safety stock: {safety_stock_result['percentile_method']:.1f} units")
print(f"Lead time demand std: {safety_stock_result['lead_time_demand_std']:.1f}")
print(f"Service level achieved: {safety_stock_result['percentile_method_service_level']:.1%}")

print("\n=== FACILITY LOCATION OPTIMIZATION ===")
from pymc_supply_chain.network import FacilityLocationOptimizer

# Create sample locations
n_customers = 20
n_candidates = 8

# Customer locations (e.g., major cities)
customer_locations = pd.DataFrame({
    'location_id': [f'Customer_{i}' for i in range(n_customers)],
    'latitude': np.random.uniform(25, 48, n_customers),
    'longitude': np.random.uniform(-125, -65, n_customers),
    'demand': np.random.exponential(1000, n_customers)
})

# Candidate warehouse locations
candidate_locations = pd.DataFrame({
    'location_id': [f'Warehouse_{i}' for i in range(n_candidates)],
    'latitude': np.random.uniform(25, 48, n_candidates),
    'longitude': np.random.uniform(-125, -65, n_candidates)
})

# Fixed costs for each warehouse
fixed_costs = {f'Warehouse_{i}': np.random.uniform(50000, 150000) for i in range(n_candidates)}

# Optimize facility locations
location_opt = FacilityLocationOptimizer(
    demand_locations=customer_locations,
    candidate_locations=candidate_locations,
    fixed_costs=fixed_costs,
    transportation_cost_per_unit_distance=0.5
)

print("Optimizing facility locations...")
result = location_opt.optimize(max_facilities=3, service_distance=500)

print(f"\nOptimal solution:")
print(f"Selected facilities: {result.solution['selected_facilities']}")
print(f"Total cost: ${result.objective_value:,.2f}")
print(f"Fixed cost: ${result.metadata['fixed_cost']:,.2f}")
print(f"Transport cost: ${result.metadata['transport_cost']:,.2f}")

# Analyze solution
facility_analysis = location_opt.analyze_solution(result)
print("\nFacility utilization:")
for _, row in facility_analysis.iterrows():
    print(f"  {row['facility_id']}: {row['utilization']:.1%} utilized, "
          f"serving {row['n_customers']:.0f} customers")

print("\n=== INTEGRATED OPTIMIZATION ===")
# Combine insights from all models
print("\nIntegrated supply chain recommendations:")
print(f"1. Forecast average demand: {forecast['forecast'].mean():.1f} units/day")
print(f"2. Maintain safety stock: {safety_stock_result['percentile_method']:.1f} units")
print(f"3. Operate {len(result.solution['selected_facilities'])} distribution centers")
print(f"4. Total inventory investment: ${(forecast['forecast'].mean() + safety_stock_result['percentile_method']) * 10:.2f}")

# Sensitivity analysis
print("\n=== SENSITIVITY ANALYSIS ===")
service_levels = np.linspace(0.8, 0.99, 10)
sensitivity_results = []

for sl in service_levels:
    ss_result = safety_stock_opt.calculate_safety_stock(confidence_level=sl)
    sensitivity_results.append({
        'service_level': sl,
        'safety_stock': ss_result['percentile_method'],
        'holding_cost': ss_result['percentile_method'] * 2.0
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
print("\nService level vs. safety stock trade-off:")
print(sensitivity_df)

# Plot sensitivity
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sensitivity_df['service_level'] * 100, sensitivity_df['safety_stock'], 'b-', linewidth=2)
ax.set_xlabel('Service Level (%)')
ax.set_ylabel('Safety Stock (units)')
ax.set_title('Safety Stock Requirements vs Service Level')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sensitivity_analysis.png')
plt.close()

print("\n✅ Quick start example completed!")
print("Generated plots: demand_forecast.png, sensitivity_analysis.png")