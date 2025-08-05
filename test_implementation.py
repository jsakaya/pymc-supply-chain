#!/usr/bin/env python3
"""
Comprehensive test script for PyMC-Supply-Chain implementation.

This script verifies that all major components work correctly by:
1. Testing imports
2. Testing basic functionality of each major component
3. Running a simple end-to-end pipeline
4. Providing clear error reporting

Run with: python test_implementation.py
"""

import sys
import traceback
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TestResults:
    """Track test results and generate summary."""
    
    def __init__(self):
        self.results = {}
        self.errors = {}
        
    def add_result(self, test_name: str, success: bool, error_msg: str = None):
        """Add a test result."""
        self.results[test_name] = success
        if error_msg:
            self.errors[test_name] = error_msg
            
    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*60)
        print("PYMC-SUPPLY-CHAIN TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for success in self.results.values() if success)
        total = len(self.results)
        
        print(f"Tests passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")
        print()
        
        # Group results by category
        categories = {
            'Import Tests': [],
            'Demand Forecasting': [],
            'Inventory Optimization': [],
            'Network Optimization': [],
            'End-to-End Pipeline': []
        }
        
        for test_name, success in self.results.items():
            if 'import' in test_name.lower():
                categories['Import Tests'].append((test_name, success))
            elif 'demand' in test_name.lower():
                categories['Demand Forecasting'].append((test_name, success))
            elif 'inventory' in test_name.lower():
                categories['Inventory Optimization'].append((test_name, success))
            elif 'network' in test_name.lower() or 'facility' in test_name.lower():
                categories['Network Optimization'].append((test_name, success))
            elif 'end-to-end' in test_name.lower() or 'pipeline' in test_name.lower():
                categories['End-to-End Pipeline'].append((test_name, success))
            else:
                categories['Import Tests'].append((test_name, success))
        
        for category, tests in categories.items():
            if tests:
                print(f"{category}:")
                for test_name, success in tests:
                    status = "✅ PASS" if success else "❌ FAIL"
                    print(f"  {status} {test_name}")
                print()
        
        # Print errors if any
        if self.errors:
            print("DETAILED ERROR MESSAGES:")
            print("-" * 40)
            for test_name, error_msg in self.errors.items():
                print(f"\n{test_name}:")
                print(f"  {error_msg}")
        
        print("\n" + "="*60)

def test_imports(results: TestResults):
    """Test all critical imports."""
    print("Testing imports...")
    
    # Core library imports
    import_tests = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("pymc", "import pymc as pm"),
        ("pytensor", "import pytensor.tensor as pt"),
        ("arviz", "import arviz as az"),
    ]
    
    for name, import_stmt in import_tests:
        try:
            exec(import_stmt)
            results.add_result(f"Import {name}", True)
            print(f"  ✅ {name}")
        except Exception as e:
            results.add_result(f"Import {name}", False, str(e))
            print(f"  ❌ {name}: {e}")
    
    # PyMC-Supply-Chain imports - test individual components
    supply_chain_imports = [
        ("base classes", "from pymc_supply_chain.base import SupplyChainModelBuilder"),
        ("demand base", "from pymc_supply_chain.demand.base import DemandForecastModel"),
        ("inventory safety stock", "from pymc_supply_chain.inventory.safety_stock import SafetyStockOptimizer"),
        ("inventory newsvendor", "from pymc_supply_chain.inventory.newsvendor import NewsvendorModel"),
        ("network facility location", "from pymc_supply_chain.network.facility_location import FacilityLocationOptimizer"),
    ]
    
    for name, import_stmt in supply_chain_imports:
        try:
            exec(import_stmt)
            results.add_result(f"Import {name}", True)
            print(f"  ✅ {name}")
        except Exception as e:
            results.add_result(f"Import {name}", False, str(e))
            print(f"  ❌ {name}: {e}")
    
    # Test full module imports (may fail due to incomplete implementation)
    full_module_imports = [
        ("main package", "import pymc_supply_chain"),
        ("demand module", "from pymc_supply_chain import demand"),
        ("inventory module", "from pymc_supply_chain import inventory"),
        ("network module", "from pymc_supply_chain import network"),
    ]
    
    print("\nTesting full module imports (some may fail)...")
    for name, import_stmt in full_module_imports:
        try:
            exec(import_stmt)
            results.add_result(f"Full import {name}", True)
            print(f"  ✅ {name}")
        except Exception as e:
            results.add_result(f"Full import {name}", False, f"Incomplete implementation: {str(e)}")
            print(f"  ⚠️  {name}: {e}")

def create_synthetic_demand_data() -> pd.DataFrame:
    """Create synthetic demand data for testing."""
    np.random.seed(42)
    
    # Create 100 days of data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate demand with trend and seasonality
    t = np.arange(100)
    trend = 100 + 0.5 * t  # Slight upward trend
    seasonality = 10 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    noise = np.random.normal(0, 5, 100)
    demand = trend + seasonality + noise
    demand = np.maximum(demand, 0)  # Ensure non-negative
    
    return pd.DataFrame({
        'date': dates,
        'demand': demand
    })

def test_demand_forecasting(results: TestResults):
    """Test demand forecasting functionality."""
    print("\nTesting demand forecasting...")
    
    try:
        from pymc_supply_chain.demand.base import DemandForecastModel
        results.add_result("Import DemandForecastModel", True)
        print("  ✅ Import successful")
    except Exception as e:
        results.add_result("Import DemandForecastModel", False, str(e))
        print(f"  ❌ Import failed: {e}")
        return
    
    # Test model initialization
    try:
        demand_model = DemandForecastModel(
            date_column='date',
            target_column='demand',
            include_trend=True,
            include_seasonality=True,
            seasonality=7
        )
        results.add_result("Initialize DemandForecastModel", True)
        print("  ✅ Model initialization")
    except Exception as e:
        results.add_result("Initialize DemandForecastModel", False, str(e))
        print(f"  ❌ Model initialization failed: {e}")
        return
    
    # Create test data
    try:
        demand_data = create_synthetic_demand_data()
        results.add_result("Create demand test data", True)
        print("  ✅ Test data creation")
    except Exception as e:
        results.add_result("Create demand test data", False, str(e))
        print(f"  ❌ Test data creation failed: {e}")
        return
    
    # Test model building
    try:
        model = demand_model.build_model(demand_data)
        results.add_result("Build demand model", True)
        print("  ✅ Model building")
    except Exception as e:
        results.add_result("Build demand model", False, str(e))
        print(f"  ❌ Model building failed: {e}")
        return
    
    # Test model fitting (with minimal sampling)
    try:
        demand_model.fit(
            demand_data,
            draws=100,  # Minimal for speed
            tune=50,
            chains=1,
            progressbar=False
        )
        results.add_result("Fit demand model", True)
        print("  ✅ Model fitting")
    except Exception as e:
        results.add_result("Fit demand model", False, str(e))
        print(f"  ❌ Model fitting failed: {e}")
        return
    
    # Test forecasting
    try:
        forecast = demand_model.forecast(steps=7)
        
        # Validate forecast structure
        expected_cols = ['date', 'forecast', 'forecast_lower', 'forecast_upper', 'forecast_std']
        if all(col in forecast.columns for col in expected_cols):
            results.add_result("Generate demand forecast", True)
            print("  ✅ Forecasting")
            print(f"    - Forecast mean: {forecast['forecast'].mean():.1f}")
            print(f"    - Forecast range: [{forecast['forecast'].min():.1f}, {forecast['forecast'].max():.1f}]")
        else:
            results.add_result("Generate demand forecast", False, f"Missing columns: {set(expected_cols) - set(forecast.columns)}")
            print(f"  ❌ Forecasting: Missing columns")
    except Exception as e:
        results.add_result("Generate demand forecast", False, str(e))
        print(f"  ❌ Forecasting failed: {e}")

def create_synthetic_inventory_data() -> pd.DataFrame:
    """Create synthetic inventory data for testing."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'demand': np.random.normal(100, 20, 50),
        'lead_time': np.random.gamma(2, 2, 50)
    })

def test_inventory_optimization(results: TestResults):
    """Test inventory optimization functionality."""
    print("\nTesting inventory optimization...")
    
    # Test SafetyStockOptimizer
    try:
        from pymc_supply_chain.inventory.safety_stock import SafetyStockOptimizer
        results.add_result("Import SafetyStockOptimizer", True)
        print("  ✅ Import SafetyStockOptimizer")
    except Exception as e:
        results.add_result("Import SafetyStockOptimizer", False, str(e))
        print(f"  ❌ Import SafetyStockOptimizer failed: {e}")
        return
    
    try:
        safety_stock_opt = SafetyStockOptimizer(
            holding_cost=2.0,
            stockout_cost=50.0,
            target_service_level=0.95
        )
        results.add_result("Initialize SafetyStockOptimizer", True)
        print("  ✅ SafetyStockOptimizer initialization")
    except Exception as e:
        results.add_result("Initialize SafetyStockOptimizer", False, str(e))
        print(f"  ❌ SafetyStockOptimizer initialization failed: {e}")
        return
    
    # Test with synthetic data
    try:
        inventory_data = create_synthetic_inventory_data()
        safety_stock_opt.fit(
            inventory_data,
            draws=100,
            tune=50,
            chains=1,
            progressbar=False
        )
        results.add_result("Fit SafetyStockOptimizer", True)
        print("  ✅ SafetyStockOptimizer fitting")
    except Exception as e:
        results.add_result("Fit SafetyStockOptimizer", False, str(e))
        print(f"  ❌ SafetyStockOptimizer fitting failed: {e}")
        return
    
    try:
        safety_stock_result = safety_stock_opt.calculate_safety_stock()
        if 'percentile_method' in safety_stock_result:
            results.add_result("Calculate safety stock", True)
            print("  ✅ Safety stock calculation")
            print(f"    - Optimal safety stock: {safety_stock_result['percentile_method']:.1f} units")
        else:
            results.add_result("Calculate safety stock", False, "Expected results not found")
            print("  ❌ Safety stock calculation: Expected results not found")
    except Exception as e:
        results.add_result("Calculate safety stock", False, str(e))
        print(f"  ❌ Safety stock calculation failed: {e}")
    
    # Test NewsvendorModel
    try:
        from pymc_supply_chain.inventory.newsvendor import NewsvendorModel
        newsvendor = NewsvendorModel(
            unit_cost=10.0,
            selling_price=25.0,
            salvage_value=5.0
        )
        results.add_result("Initialize NewsvendorModel", True)
        print("  ✅ NewsvendorModel initialization")
    except Exception as e:
        results.add_result("Initialize NewsvendorModel", False, str(e))
        print(f"  ❌ NewsvendorModel initialization failed: {e}")

def create_synthetic_location_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Create synthetic location data for testing."""
    np.random.seed(42)
    
    # Customer locations
    n_customers = 10
    customer_locations = pd.DataFrame({
        'location_id': [f'Customer_{i}' for i in range(n_customers)],
        'latitude': np.random.uniform(25, 48, n_customers),
        'longitude': np.random.uniform(-125, -65, n_customers),
        'demand': np.random.exponential(1000, n_customers)
    })
    
    # Candidate warehouse locations
    n_candidates = 5
    candidate_locations = pd.DataFrame({
        'location_id': [f'Warehouse_{i}' for i in range(n_candidates)],
        'latitude': np.random.uniform(25, 48, n_candidates),
        'longitude': np.random.uniform(-125, -65, n_candidates)
    })
    
    # Fixed costs
    fixed_costs = {f'Warehouse_{i}': np.random.uniform(50000, 150000) for i in range(n_candidates)}
    
    return customer_locations, candidate_locations, fixed_costs

def test_network_optimization(results: TestResults):
    """Test network optimization functionality."""
    print("\nTesting network optimization...")
    
    try:
        from pymc_supply_chain.network.facility_location import FacilityLocationOptimizer
        results.add_result("Import FacilityLocationOptimizer", True)
        print("  ✅ Import FacilityLocationOptimizer")
    except Exception as e:
        results.add_result("Import FacilityLocationOptimizer", False, str(e))
        print(f"  ❌ Import FacilityLocationOptimizer failed: {e}")
        return
    
    try:
        customer_locs, candidate_locs, fixed_costs = create_synthetic_location_data()
        
        location_opt = FacilityLocationOptimizer(
            demand_locations=customer_locs,
            candidate_locations=candidate_locs,
            fixed_costs=fixed_costs,
            transportation_cost_per_unit_distance=0.5
        )
        results.add_result("Initialize FacilityLocationOptimizer", True)
        print("  ✅ FacilityLocationOptimizer initialization")
    except Exception as e:
        results.add_result("Initialize FacilityLocationOptimizer", False, str(e))
        print(f"  ❌ FacilityLocationOptimizer initialization failed: {e}")
        return
    
    try:
        # Run optimization
        result = location_opt.optimize(max_facilities=2, service_distance=1000)
        
        if hasattr(result, 'solution') and 'selected_facilities' in result.solution:
            results.add_result("Optimize facility locations", True)
            print("  ✅ Facility location optimization")
            print(f"    - Selected facilities: {result.solution['selected_facilities']}")
            print(f"    - Total cost: ${result.objective_value:,.2f}")
        else:
            results.add_result("Optimize facility locations", False, "Expected solution structure not found")
            print("  ❌ Facility location optimization: Expected solution not found")
    except Exception as e:
        results.add_result("Optimize facility locations", False, str(e))
        print(f"  ❌ Facility location optimization failed: {e}")

def test_end_to_end_pipeline(results: TestResults):
    """Test a simple end-to-end supply chain optimization pipeline."""
    print("\nTesting end-to-end pipeline...")
    
    pipeline_success = True
    pipeline_errors = []
    
    # Step 1: Demand forecasting
    try:
        from pymc_supply_chain.demand.base import DemandForecastModel
        
        demand_data = create_synthetic_demand_data()
        demand_model = DemandForecastModel(
            date_column='date',
            target_column='demand',
            include_trend=True,
            seasonality=7
        )
        
        demand_model.fit(demand_data, draws=50, tune=25, chains=1, progressbar=False)
        forecast = demand_model.forecast(steps=14)
        
        avg_forecast = forecast['forecast'].mean()
        print(f"  ✅ Step 1 - Demand forecast: {avg_forecast:.1f} units/day")
        
    except Exception as e:
        pipeline_success = False
        pipeline_errors.append(f"Demand forecasting: {e}")
        print(f"  ❌ Step 1 - Demand forecasting failed: {e}")
    
    # Step 2: Safety stock optimization
    try:
        from pymc_supply_chain.inventory.safety_stock import SafetyStockOptimizer
        
        inventory_data = create_synthetic_inventory_data()
        safety_stock_opt = SafetyStockOptimizer(
            holding_cost=2.0,
            stockout_cost=50.0,
            target_service_level=0.95
        )
        
        safety_stock_opt.fit(inventory_data, draws=50, tune=25, chains=1, progressbar=False)
        safety_stock_result = safety_stock_opt.calculate_safety_stock()
        
        optimal_safety_stock = safety_stock_result['percentile_method']
        print(f"  ✅ Step 2 - Safety stock: {optimal_safety_stock:.1f} units")
        
    except Exception as e:
        pipeline_success = False
        pipeline_errors.append(f"Safety stock optimization: {e}")
        print(f"  ❌ Step 2 - Safety stock optimization failed: {e}")
    
    # Step 3: Network optimization
    try:
        from pymc_supply_chain.network.facility_location import FacilityLocationOptimizer
        
        customer_locs, candidate_locs, fixed_costs = create_synthetic_location_data()
        location_opt = FacilityLocationOptimizer(
            demand_locations=customer_locs,
            candidate_locations=candidate_locs,
            fixed_costs=fixed_costs,
            transportation_cost_per_unit_distance=0.5
        )
        
        result = location_opt.optimize(max_facilities=2)
        selected_facilities = result.solution['selected_facilities']
        total_cost = result.objective_value
        
        print(f"  ✅ Step 3 - Selected {len(selected_facilities)} facilities, cost: ${total_cost:,.0f}")
        
    except Exception as e:
        pipeline_success = False
        pipeline_errors.append(f"Network optimization: {e}")
        print(f"  ❌ Step 3 - Network optimization failed: {e}")
    
    # Step 4: Integration
    if pipeline_success:
        try:
            # Calculate total inventory investment
            total_inventory = avg_forecast + optimal_safety_stock
            inventory_cost = total_inventory * 10  # Assume $10 per unit
            total_system_cost = inventory_cost + total_cost
            
            print(f"  ✅ Step 4 - Integrated solution:")
            print(f"    - Total inventory: {total_inventory:.1f} units")
            print(f"    - Inventory investment: ${inventory_cost:,.0f}")
            print(f"    - Network costs: ${total_cost:,.0f}")
            print(f"    - Total system cost: ${total_system_cost:,.0f}")
            
            results.add_result("End-to-end pipeline", True)
            
        except Exception as e:
            pipeline_success = False
            pipeline_errors.append(f"Integration: {e}")
            print(f"  ❌ Step 4 - Integration failed: {e}")
    
    if not pipeline_success:
        error_msg = "; ".join(pipeline_errors)
        results.add_result("End-to-end pipeline", False, error_msg)

def test_optional_components(results: TestResults):
    """Test optional components that might not be fully implemented."""
    print("\nTesting optional components...")
    
    # Test additional demand models
    optional_imports = [
        ("HierarchicalDemandModel", "from pymc_supply_chain.demand import HierarchicalDemandModel"),
        ("IntermittentDemandModel", "from pymc_supply_chain.demand import IntermittentDemandModel"),
        ("SeasonalDemandModel", "from pymc_supply_chain.demand import SeasonalDemandModel"),
        ("EOQModel", "from pymc_supply_chain.inventory import EOQModel"),
        ("MultiEchelonInventory", "from pymc_supply_chain.inventory import MultiEchelonInventory"),
    ]
    
    for name, import_stmt in optional_imports:
        try:
            exec(import_stmt)
            results.add_result(f"Import {name}", True)
            print(f"  ✅ {name}")
        except Exception as e:
            results.add_result(f"Import {name}", False, f"Optional component: {str(e)}")
            print(f"  ⚠️  {name} (optional): {e}")

def main():
    """Run comprehensive test suite."""
    print("PyMC-Supply-Chain Implementation Test Suite")
    print("=" * 50)
    
    results = TestResults()
    
    try:
        # Core tests
        test_imports(results)
        test_demand_forecasting(results)
        test_inventory_optimization(results)
        test_network_optimization(results)
        test_end_to_end_pipeline(results)
        test_optional_components(results)
        
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user.")
        results.add_result("Test suite completion", False, "Interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error in test suite: {e}")
        traceback.print_exc()
        results.add_result("Test suite completion", False, f"Unexpected error: {e}")
    
    # Print comprehensive summary
    results.print_summary()
    
    # Return appropriate exit code
    all_passed = all(results.results.values())
    critical_passed = all(
        results.results.get(test, False) 
        for test in ["Import main package", "Import demand module", "Import inventory module"]
        if test in results.results
    )
    
    if all_passed:
        print("🎉 All tests passed! PyMC-Supply-Chain is working correctly.")
        return 0
    elif critical_passed:
        print("⚠️  Some tests failed, but core functionality is working.")
        return 1
    else:
        print("❌ Critical tests failed. Implementation needs attention.")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)