#!/usr/bin/env python3
"""
Debug script to test the actual PyMC-Supply-Chain model implementations.
This will help identify API mismatches and fix the case study.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Test SeasonalDemandModel
def test_seasonal_demand_model():
    print("Testing SeasonalDemandModel...")
    
    try:
        from pymc_supply_chain.demand.seasonal import SeasonalDemandModel
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        demand = 10 + 2 * np.sin(2 * np.pi * np.arange(100) / 7) + np.random.normal(0, 1, 100)
        data = pd.DataFrame({
            'date': dates,
            'demand': np.maximum(0, demand)
        })
        
        # Initialize model
        model = SeasonalDemandModel(
            date_column='date',
            target_column='demand',
            weekly_seasonality=2,
            yearly_seasonality=5
        )
        
        # Fit model
        print("  Fitting model...")
        model.fit(data, draws=100, tune=100, progressbar=False)
        print("  ✅ Model fitted successfully")
        
        # Test forecast
        print("  Testing forecast...")
        forecast = model.forecast(steps=10)
        print(f"  ✅ Forecast generated: {len(forecast)} periods")
        print(f"     Columns: {list(forecast.columns)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SeasonalDemandModel failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_newsvendor_model():
    print("\nTesting NewsvendorModel...")
    
    try:
        from pymc_supply_chain.inventory.newsvendor import NewsvendorModel
        
        # Create test data  
        demand_data = np.random.gamma(2, 5, 200)  # Gamma distributed demand
        data = pd.DataFrame({'demand': demand_data})
        
        # Initialize model
        model = NewsvendorModel(
            unit_cost=10,
            selling_price=20,
            salvage_value=5,
            demand_distribution='gamma'
        )
        
        # Fit model
        print("  Fitting model...")
        model.fit(data, draws=100, tune=100, progressbar=False)
        print("  ✅ Model fitted successfully")
        
        # Test optimization
        print("  Testing optimal quantity calculation...")
        optimal = model.calculate_optimal_quantity(n_samples=500)
        print(f"  ✅ Optimal quantity: {optimal['optimal_quantity']:.1f}")
        print(f"     Expected profit: ${optimal['expected_profit']:.0f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ NewsvendorModel failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safety_stock_optimizer():
    print("\nTesting SafetyStockOptimizer...")
    
    try:
        from pymc_supply_chain.inventory.safety_stock import SafetyStockOptimizer
        
        # Create test data
        demand_data = np.random.normal(20, 5, 100)
        lead_time_data = np.random.gamma(2, 1.5, 100)  # Variable lead times
        
        data = pd.DataFrame({
            'demand': np.maximum(0, demand_data),
            'lead_time': lead_time_data
        })
        
        # Initialize model
        model = SafetyStockOptimizer(
            holding_cost=2.0,
            stockout_cost=50.0,
            target_service_level=0.95
        )
        
        # Fit model
        print("  Fitting model...")
        model.fit(data, draws=100, tune=100, progressbar=False)
        print("  ✅ Model fitted successfully")
        
        # Test safety stock calculation
        print("  Testing safety stock calculation...")
        safety_stock = model.calculate_safety_stock()
        print(f"  ✅ Safety stock calculated")
        print(f"     Method: {list(safety_stock.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SafetyStockOptimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_facility_location_optimizer():
    print("\nTesting FacilityLocationOptimizer...")
    
    try:
        from pymc_supply_chain.network.facility_location import FacilityLocationOptimizer
        
        # Create test data
        demand_locations = pd.DataFrame({
            'location_id': ['Store1', 'Store2', 'Store3'],
            'latitude': [40.7, 34.0, 41.8],
            'longitude': [-74.0, -118.2, -87.6],
            'demand': [100, 150, 120]
        })
        
        candidate_locations = pd.DataFrame({
            'location_id': ['DC1', 'DC2', 'DC3'],
            'latitude': [39.0, 36.0, 42.0],
            'longitude': [-76.0, -115.0, -85.0]
        })
        
        fixed_costs = {'DC1': 100000, 'DC2': 120000, 'DC3': 90000}
        
        # Initialize optimizer
        optimizer = FacilityLocationOptimizer(
            demand_locations=demand_locations,
            candidate_locations=candidate_locations,
            fixed_costs=fixed_costs,
            transportation_cost_per_unit_distance=0.5
        )
        
        # Test optimization
        print("  Running optimization...")
        result = optimizer.optimize(max_facilities=2)
        print(f"  ✅ Optimization completed")
        print(f"     Status: {result.status}")
        print(f"     Objective value: ${result.objective_value:.0f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FacilityLocationOptimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("PyMC-Supply-Chain Model Debug Tests")
    print("=" * 60)
    
    results = {}
    results['SeasonalDemandModel'] = test_seasonal_demand_model()
    results['NewsvendorModel'] = test_newsvendor_model()
    results['SafetyStockOptimizer'] = test_safety_stock_optimizer()
    results['FacilityLocationOptimizer'] = test_facility_location_optimizer()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model:<25} {status}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_pass}/{total_tests} tests passed")
    
    if total_pass == total_tests:
        print("🎉 All models working correctly!")
    else:
        print("⚠️  Some models need fixing")


if __name__ == "__main__":
    main()