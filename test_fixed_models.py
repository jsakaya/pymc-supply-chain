"""
Test script with fixed PyMC models addressing API compatibility issues.
This demonstrates how to update the models to work with current PyMC version.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

print("Testing fixed PyMC models...")

# 1. Fixed Demand Forecast Model
print("\n1. Testing Fixed Demand Forecast Model")
try:
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    demand = 100 + 10 * np.sin(2 * np.pi * np.arange(100) / 7) + np.random.normal(0, 5, 100)
    df = pd.DataFrame({'date': dates, 'demand': demand})
    
    # Build a simple PyMC model with correct API
    with pm.Model() as demand_model:
        # Data - using correct PyMC API
        demand_data = pm.Data('demand_data', df['demand'].values)
        time_idx = pm.Data('time_idx', np.arange(len(df)))
        
        # Priors
        intercept = pm.Normal('intercept', mu=100, sigma=20)
        trend = pm.Normal('trend', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=10)
        
        # Model
        mu = intercept + trend * time_idx
        
        # Likelihood
        pm.Normal('demand', mu=mu, sigma=sigma, observed=demand_data)
        
        # Sample
        print("  Sampling from demand model...")
        trace = pm.sample(100, tune=50, progressbar=False, chains=2)
        
    print("  ✅ Fixed demand model works!")
    print(f"  Mean intercept: {trace.posterior['intercept'].mean():.2f}")
    
except Exception as e:
    print(f"  ❌ Fixed demand model failed: {e}")

# 2. Fixed Safety Stock Model
print("\n2. Testing Fixed Safety Stock Model")
try:
    # Create demand and lead time data
    demand_values = np.random.normal(100, 20, 50)
    lead_times = np.random.gamma(2, 1, 50)
    
    with pm.Model() as safety_stock_model:
        # Demand distribution
        demand_mu = pm.Normal('demand_mu', mu=100, sigma=20)
        demand_sigma = pm.HalfNormal('demand_sigma', sigma=10)
        
        # Lead time distribution  
        lead_time_alpha = pm.Exponential('lead_time_alpha', 1.0)
        lead_time_beta = pm.Exponential('lead_time_beta', 1.0)
        
        # Observations
        pm.Normal('demand_obs', mu=demand_mu, sigma=demand_sigma, observed=demand_values)
        pm.Gamma('lead_time_obs', alpha=lead_time_alpha, beta=lead_time_beta, observed=lead_times)
        
        # Sample
        print("  Sampling from safety stock model...")
        trace = pm.sample(100, tune=50, progressbar=False, chains=2)
        
    # Calculate lead time demand statistics
    with safety_stock_model:
        # Sample posterior predictive
        post_pred = pm.sample_posterior_predictive(trace, progressbar=False)
        
    print("  ✅ Fixed safety stock model works!")
    print(f"  Demand mean: {demand_mu.eval():.2f}")
    
except Exception as e:
    print(f"  ❌ Fixed safety stock model failed: {e}")

# 3. Fixed Newsvendor Model
print("\n3. Testing Fixed Newsvendor Model")
try:
    # Historical demand data
    historical_demand = np.random.gamma(5, 20, 100)
    
    with pm.Model() as newsvendor_model:
        # Gamma distribution for demand
        alpha = pm.Exponential('alpha', 1.0)
        beta = pm.Exponential('beta', 1.0/100)
        
        # Likelihood
        pm.Gamma('demand', alpha=alpha, beta=beta, observed=historical_demand)
        
        # Sample
        print("  Sampling from newsvendor model...")
        trace = pm.sample(100, tune=50, progressbar=False, chains=2)
        
        # Calculate optimal order quantity
        # For gamma distribution, critical fractile solution
        unit_cost = 10
        selling_price = 25
        critical_ratio = (selling_price - unit_cost) / selling_price
        
        # Sample from posterior predictive
        post_pred = pm.sample_posterior_predictive(trace, progressbar=False, predictions=True)
        demand_samples = post_pred.predictions['demand'].values.flatten()
        
        optimal_quantity = np.percentile(demand_samples, critical_ratio * 100)
        
    print("  ✅ Fixed newsvendor model works!")
    print(f"  Optimal order quantity: {optimal_quantity:.2f}")
    
except Exception as e:
    print(f"  ❌ Fixed newsvendor model failed: {e}")

# 4. Test Simple End-to-End Pipeline
print("\n4. Testing Simple End-to-End Pipeline")
try:
    # Step 1: Generate forecast
    future_demand_mean = trace.posterior['demand_mu'].mean().item()
    future_demand_std = trace.posterior['demand_sigma'].mean().item()
    
    # Step 2: Calculate safety stock (simple formula)
    service_level = 0.95
    z_score = 1.645  # 95% service level
    lead_time = 5  # days
    safety_stock = z_score * future_demand_std * np.sqrt(lead_time)
    
    # Step 3: Reorder point
    reorder_point = future_demand_mean * lead_time + safety_stock
    
    print("  ✅ End-to-end pipeline works!")
    print(f"  Forecast demand: {future_demand_mean:.2f} ± {future_demand_std:.2f}")
    print(f"  Safety stock: {safety_stock:.2f}")
    print(f"  Reorder point: {reorder_point:.2f}")
    
except Exception as e:
    print(f"  ❌ End-to-end pipeline failed: {e}")

print("\n" + "="*50)
print("SUMMARY: Fixed models demonstrate proper PyMC API usage")
print("Key changes needed:")
print("- Replace pm.ConstantData with pm.Data")
print("- Use pm.sample_posterior_predictive correctly")
print("- Ensure all PyMC models follow current API patterns")
print("="*50)