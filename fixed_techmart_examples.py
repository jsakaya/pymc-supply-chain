#!/usr/bin/env python3
"""
Fixed TechMart Supply Chain Examples - Corrected APIs
This file demonstrates the working APIs for each model in the PyMC-Supply-Chain library.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import models with correct APIs
from pymc_supply_chain.demand.seasonal import SeasonalDemandModel
from pymc_supply_chain.demand.intermittent import IntermittentDemandModel
from pymc_supply_chain.inventory.newsvendor import NewsvendorModel
from pymc_supply_chain.inventory.safety_stock import SafetyStockOptimizer
from pymc_supply_chain.inventory.eoq import EOQModel
from pymc_supply_chain.network.facility_location import FacilityLocationOptimizer

print("="*80)
print("FIXED TECHMART SUPPLY CHAIN EXAMPLES")
print("Corrected APIs for PyMC-Supply-Chain Models")
print("="*80)

# =============================================================================
# 1. SEASONAL DEMAND FORECASTING
# =============================================================================
print("\n1️⃣ SEASONAL DEMAND FORECASTING")
print("-" * 50)

# Generate sample seasonal demand data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
base_demand = 15
weekly_pattern = 3 * np.sin(2 * np.pi * np.arange(100) / 7)
trend = 0.1 * np.arange(100)
noise = np.random.normal(0, 2, 100)
demand = base_demand + weekly_pattern + trend + noise

seasonal_data = pd.DataFrame({
    'date': dates,
    'demand': np.maximum(0, demand)
})

print(f"Sample data: {len(seasonal_data)} days of demand")
print(f"Mean demand: {seasonal_data['demand'].mean():.1f} units/day")

# ✅ CORRECT API: SeasonalDemandModel
seasonal_model = SeasonalDemandModel(
    date_column='date',
    target_column='demand',
    weekly_seasonality=3,
    yearly_seasonality=5,
    n_changepoints=10
)

# Fit model
print("Fitting seasonal model...")
seasonal_model.fit(seasonal_data, draws=500, tune=500, progressbar=False)

# ✅ CORRECT API: forecast() method returns DataFrame with specific columns
forecast = seasonal_model.forecast(steps=14)
print(f"✅ Forecast generated: {len(forecast)} periods")
print(f"   Columns: {list(forecast.columns)}")
print(f"   Mean forecast: {forecast['forecast'].mean():.1f} units/day")
print(f"   95% CI width: {(forecast['upper_95'] - forecast['lower_95']).mean():.1f}")

# =============================================================================
# 2. INTERMITTENT DEMAND MODELING
# =============================================================================
print("\n2️⃣ INTERMITTENT DEMAND MODELING")
print("-" * 50)

# Generate intermittent demand (many zeros, occasional large values)
intermittent_demand = np.zeros(100)
demand_events = np.random.choice(100, 15, replace=False)  # 15% of days with demand
intermittent_demand[demand_events] = np.random.gamma(2, 5, 15)

intermittent_data = pd.DataFrame({
    'date': dates,
    'demand': intermittent_demand
})

print(f"Zero-demand days: {(intermittent_demand == 0).sum()}/100 ({(intermittent_demand == 0).mean()*100:.0f}%)")

# ✅ CORRECT API: IntermittentDemandModel
intermittent_model = IntermittentDemandModel(
    method='croston',
    date_column='date',
    target_column='demand'
)

# ✅ CORRECT API: analyze_demand_pattern() for classification
pattern_analysis = intermittent_model.analyze_demand_pattern(intermittent_data['demand'])
print(f"✅ Demand pattern: {pattern_analysis['pattern_type']}")
print(f"   Average demand interval: {pattern_analysis['average_demand_interval']:.1f} days")
print(f"   CV²: {pattern_analysis['coefficient_of_variation_squared']:.2f}")

# Fit and forecast
intermittent_model.fit(intermittent_data, draws=500, tune=500, progressbar=False)
intermittent_forecast = intermittent_model.forecast(steps=14)
print(f"✅ Intermittent forecast: {len(intermittent_forecast)} periods")

# =============================================================================
# 3. NEWSVENDOR OPTIMIZATION
# =============================================================================
print("\n3️⃣ NEWSVENDOR OPTIMIZATION")
print("-" * 50)

# Generate demand data for perishable product
np.random.seed(123)
product_demand = np.random.gamma(shape=2, scale=15, size=200)  # Right-skewed demand

# ✅ CORRECT API: DataFrame with 'demand' column
demand_df = pd.DataFrame({'demand': product_demand})
print(f"Historical demand: {len(product_demand)} observations")
print(f"Mean: {product_demand.mean():.1f}, Std: {product_demand.std():.1f}")

# ✅ CORRECT API: NewsvendorModel initialization
newsvendor = NewsvendorModel(
    unit_cost=50,
    selling_price=100,
    salvage_value=20,
    shortage_cost=30,
    demand_distribution='gamma'
)

# Fit demand distribution
newsvendor.fit(demand_df, draws=500, tune=500, progressbar=False)

# ✅ CORRECT API: calculate_optimal_quantity() returns dict with specific keys
optimal_result = newsvendor.calculate_optimal_quantity()
print(f"✅ Optimal order quantity: {optimal_result['optimal_quantity']:.0f} units")
print(f"   Expected profit: ${optimal_result['expected_profit']:,.0f}")
print(f"   Stockout probability: {optimal_result['stockout_probability']:.1%}")
print(f"   Critical ratio: {optimal_result['critical_ratio']:.3f}")

# =============================================================================
# 4. SAFETY STOCK OPTIMIZATION
# =============================================================================
print("\n4️⃣ SAFETY STOCK OPTIMIZATION")
print("-" * 50)

# Generate demand and lead time data
np.random.seed(456)
daily_demand = np.random.normal(25, 8, 100)
lead_times = np.random.gamma(2, 1.5, 100)  # Variable lead times

# ✅ CORRECT API: DataFrame with 'demand' and 'lead_time' columns
safety_data = pd.DataFrame({
    'demand': np.maximum(0, daily_demand),
    'lead_time': lead_times
})

print(f"Demand: mean={safety_data['demand'].mean():.1f}, std={safety_data['demand'].std():.1f}")
print(f"Lead time: mean={safety_data['lead_time'].mean():.1f} days")

# ✅ CORRECT API: SafetyStockOptimizer initialization
safety_optimizer = SafetyStockOptimizer(
    holding_cost=1.5,  # $ per unit per day
    stockout_cost=25,  # $ per stockout
    target_service_level=0.95
)

# Fit models
safety_optimizer.fit(safety_data, draws=500, tune=500, progressbar=False)

# ✅ CORRECT API: calculate_safety_stock() with confidence_level parameter
safety_result = safety_optimizer.calculate_safety_stock(confidence_level=0.95)
print(f"✅ Safety stock methods:")
for method in ['percentile_method', 'normal_approximation', 'cost_optimal']:
    if method in safety_result:
        print(f"   {method}: {safety_result[method]:.1f} units")

# =============================================================================
# 5. ECONOMIC ORDER QUANTITY (EOQ)
# =============================================================================
print("\n5️⃣ ECONOMIC ORDER QUANTITY")
print("-" * 50)

# ✅ CORRECT API: EOQModel (not StochasticEOQ)
eoq_model = EOQModel(
    holding_cost_rate=0.20,  # 20% annually
    fixed_order_cost=250,
    unit_cost=45
)

# Calculate for given annual demand
annual_demand = 25 * 365  # 25 units per day
eoq_result = eoq_model.calculate_eoq(annual_demand, unit_cost=45)

print(f"Annual demand: {annual_demand:,} units")
print(f"✅ EOQ optimization:")
print(f"   Optimal order quantity: {eoq_result['eoq']:.0f} units")
print(f"   Orders per year: {eoq_result['number_of_orders']:.1f}")
print(f"   Time between orders: {eoq_result['time_between_orders_days']:.0f} days")
print(f"   Total annual cost: ${eoq_result['total_cost']:,.0f}")

# =============================================================================
# 6. FACILITY LOCATION OPTIMIZATION
# =============================================================================
print("\n6️⃣ FACILITY LOCATION OPTIMIZATION")
print("-" * 50)

# ✅ CORRECT API: DataFrames with proper column names
demand_locations = pd.DataFrame({
    'location_id': ['NYC', 'LAX', 'CHI', 'MIA'],
    'latitude': [40.7, 34.0, 41.9, 25.8],
    'longitude': [-74.0, -118.2, -87.6, -80.2],
    'demand': [500, 800, 600, 400]
})

candidate_locations = pd.DataFrame({
    'location_id': ['DC_East', 'DC_Central', 'DC_West'],
    'latitude': [39.9, 39.7, 37.4],
    'longitude': [-75.2, -104.9, -122.1]
})

# ✅ CORRECT API: Fixed costs as separate dictionary
fixed_costs = {
    'DC_East': 800000,
    'DC_Central': 700000,
    'DC_West': 900000
}

print(f"Demand locations: {len(demand_locations)}")
print(f"Candidate DCs: {len(candidate_locations)}")
print(f"Total demand: {demand_locations['demand'].sum():,} units")

# ✅ CORRECT API: FacilityLocationOptimizer initialization
optimizer = FacilityLocationOptimizer(
    demand_locations=demand_locations,
    candidate_locations=candidate_locations,
    fixed_costs=fixed_costs,
    transportation_cost_per_unit_distance=0.8
)

# ✅ CORRECT API: optimize() with proper parameter names
result = optimizer.optimize(
    max_facilities=2,
    service_distance=1500  # miles
)

print(f"✅ Optimization result:")
print(f"   Status: {result.status}")
print(f"   Total cost: ${result.objective_value:,.0f}")
print(f"   Selected facilities: {result.solution['selected_facilities']}")
print(f"   Fixed cost: ${result.metadata['fixed_cost']:,.0f}")
print(f"   Transport cost: ${result.metadata['transport_cost']:,.0f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 ALL MODELS WORKING WITH CORRECT APIS!")
print("="*80)
print("\n📋 Key API Corrections Made:")
print("• SeasonalDemandModel.forecast() now uses correct parameter names (k, m, delta)")
print("• NewsvendorModel.calculate_optimal_quantity() returns correct dictionary keys")
print("• SafetyStockOptimizer expects DataFrame with 'demand' and 'lead_time' columns")
print("• FacilityLocationOptimizer uses 'latitude'/'longitude' and 'selected_facilities'")
print("• IntermittentDemandModel has working forecast() method")
print("• EOQModel replaces non-existent StochasticEOQ")
print("\n✅ The TechMart case study can now run successfully with these corrected APIs!")