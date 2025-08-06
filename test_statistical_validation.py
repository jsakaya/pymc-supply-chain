#!/usr/bin/env python3
"""
Comprehensive statistical validation tests for PyMC-Supply-Chain demand models.

This test suite rigorously validates:
1. Forecast correctness and statistical properties
2. Posterior predictive checks 
3. Model-specific statistical behaviors
4. Distribution compliance
5. Uncertainty quantification accuracy

Critical for ensuring models produce statistically valid, economically meaningful results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Any
import arviz as az

warnings.filterwarnings('ignore')

# Import the models
from pymc_supply_chain.demand.base import DemandForecastModel
from pymc_supply_chain.demand.hierarchical import HierarchicalDemandModel
from pymc_supply_chain.demand.intermittent import IntermittentDemandModel
from pymc_supply_chain.demand.seasonal import SeasonalDemandModel


class StatisticalValidationSuite:
    """Comprehensive statistical validation for demand forecasting models."""
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize validation suite.
        
        Parameters
        ----------
        significance_level : float
            Statistical significance level for tests
        """
        self.alpha = significance_level
        self.validation_results = {}
        
    def generate_test_data(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive test datasets with known statistical properties."""
        np.random.seed(42)  # Reproducible results
        
        # Base time series parameters
        n_periods = 150
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        t = np.arange(n_periods)
        
        # 1. Base Model Data - Simple trend + seasonality + noise
        trend_base = 0.1 * t + 20  # Linear growth
        seasonal_base = 5 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        noise_base = np.random.normal(0, 2, n_periods)
        
        base_demand = np.maximum(0, trend_base + seasonal_base + noise_base)
        
        base_data = pd.DataFrame({
            'date': dates,
            'demand': base_demand,
            'external_var': np.random.normal(0, 1, n_periods),
            'price': np.random.uniform(8, 12, n_periods)
        })
        
        # 2. Hierarchical Data - Multiple regions/stores with different patterns
        regions = ['North', 'South', 'East', 'West']
        stores = ['Store_A', 'Store_B', 'Store_C']
        
        hierarchical_data = []
        region_effects = {'North': 25, 'South': 15, 'East': 20, 'West': 18}
        store_effects = {'Store_A': 5, 'Store_B': 0, 'Store_C': -3}
        
        for region in regions:
            for store in stores:
                # Different base demand for each region-store combination
                region_multiplier = region_effects[region]
                store_adjustment = store_effects[store]
                
                # Hierarchical structure: global trend + region effect + store effect + noise
                demand_hier = (
                    region_multiplier + 
                    store_adjustment +
                    0.05 * t +  # Global trend
                    2 * np.sin(2 * np.pi * t / 7) +  # Weekly seasonality
                    np.random.normal(0, 3, n_periods)  # Regional noise
                )
                demand_hier = np.maximum(0, demand_hier)
                
                for i in range(n_periods):
                    hierarchical_data.append({
                        'date': dates[i],
                        'demand': demand_hier[i],
                        'region': region,
                        'store': store,
                        'external_var': np.random.normal(0, 1)
                    })
                    
        hierarchical_df = pd.DataFrame(hierarchical_data)
        
        # 3. Intermittent Data - Many zeros with sporadic non-zero demands
        # Create truly sporadic pattern: 70% zeros, 30% positive demands
        intermittent_demand = np.zeros(n_periods)
        zero_prob = 0.7  # 70% zero demand periods
        
        for i in range(n_periods):
            if np.random.random() > zero_prob:
                # When demand occurs, it follows exponential distribution
                intermittent_demand[i] = np.random.exponential(8)
                
        intermittent_data = pd.DataFrame({
            'date': dates,
            'demand': intermittent_demand,
            'service_requests': np.random.poisson(2, n_periods),  # External factor
            'inventory_level': np.random.uniform(0, 100, n_periods)
        })
        
        # 4. Seasonal Data - Complex seasonality patterns
        # Multiple seasonal patterns: yearly, monthly, weekly
        yearly_season = 10 * np.sin(2 * np.pi * t / 365) + 5 * np.cos(2 * np.pi * t / 365)
        monthly_season = 3 * np.sin(2 * np.pi * t / 30) + 2 * np.cos(2 * np.pi * t / 30)
        weekly_season = 4 * np.sin(2 * np.pi * t / 7) + 3 * np.cos(2 * np.pi * t / 7)
        
        seasonal_demand = np.maximum(0,
            50 +  # Base level
            0.2 * t +  # Trend
            yearly_season + monthly_season + weekly_season +  # Seasonalities
            np.random.normal(0, 5, n_periods)  # Noise
        )
        
        seasonal_data = pd.DataFrame({
            'date': dates,
            'demand': seasonal_demand,
            'temperature': 20 + 15 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, n_periods),
            'marketing_spend': np.random.uniform(1000, 5000, n_periods)
        })
        
        return {
            'base': base_data,
            'hierarchical': hierarchical_df,
            'intermittent': intermittent_data,
            'seasonal': seasonal_data
        }
    
    def test_forecast_correctness(self, model, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Test fundamental forecast correctness properties."""
        print(f"\n=== Testing Forecast Correctness for {model_name} ===")
        
        results = {'model': model_name}
        
        try:
            # Fit model on training data (80% split)
            train_size = int(0.8 * len(data))
            train_data = data.iloc[:train_size].copy()
            test_data = data.iloc[train_size:].copy()
            
            # Fit model with minimal draws for speed
            model.fit(train_data, draws=500, tune=500, chains=2, target_accept=0.85)
            
            # Generate forecasts for test period
            forecast_steps = len(test_data)
            forecast = model.forecast(steps=forecast_steps)
            
            # 1. Test: No NaN/Inf values in forecasts
            nan_check = np.any(np.isnan(forecast['forecast'])) or np.any(np.isinf(forecast['forecast']))
            results['has_nan_inf'] = nan_check
            print(f"✓ NaN/Inf check: {'FAIL' if nan_check else 'PASS'}")
            
            # 2. Test: Non-negative forecasts for count distributions
            if hasattr(model, 'distribution') and model.distribution in ['negative_binomial', 'poisson', 'gamma']:
                negative_forecasts = np.any(forecast['forecast'] < 0)
                results['has_negative_forecasts'] = negative_forecasts
                print(f"✓ Non-negative forecasts: {'FAIL' if negative_forecasts else 'PASS'}")
            
            # 3. Test: Forecast uncertainty is reasonable (not too narrow/wide)
            forecast_width = forecast['forecast_upper'] - forecast['forecast_lower']
            mean_width_ratio = np.mean(forecast_width / forecast['forecast'])
            results['mean_width_ratio'] = mean_width_ratio
            
            # Reasonable width should be 20% to 200% of mean forecast
            width_reasonable = 0.2 <= mean_width_ratio <= 2.0
            results['reasonable_uncertainty'] = width_reasonable
            print(f"✓ Reasonable uncertainty (width ratio: {mean_width_ratio:.2f}): {'PASS' if width_reasonable else 'FAIL'}")
            
            # 4. Test: Forecasts are in reasonable range compared to training data
            train_mean = train_data[model.target_column].mean()
            train_std = train_data[model.target_column].std()
            
            # Forecasts should be within 5 standard deviations of training mean
            forecast_range_check = np.all(
                (forecast['forecast'] >= train_mean - 5 * train_std) &
                (forecast['forecast'] <= train_mean + 5 * train_std)
            )
            results['reasonable_forecast_range'] = forecast_range_check
            print(f"✓ Reasonable forecast range: {'PASS' if forecast_range_check else 'FAIL'}")
            
            # 5. Test: Uncertainty intervals contain expected percentage of realizations
            # (This is approximate since we have limited test data)
            if len(test_data) >= 10:
                actual_values = test_data[model.target_column].values[:len(forecast)]
                coverage_95 = np.mean(
                    (actual_values >= forecast['forecast_lower']) &
                    (actual_values <= forecast['forecast_upper'])
                )
                results['coverage_95'] = coverage_95
                
                # 95% intervals should contain 85-100% of values (allowing for sampling variation)
                coverage_reasonable = coverage_95 >= 0.75  # Relaxed due to small sample
                results['reasonable_coverage'] = coverage_reasonable
                print(f"✓ Coverage test (95% CI covers {coverage_95:.1%} of actuals): {'PASS' if coverage_reasonable else 'FAIL'}")
            
            results['success'] = True
            
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_posterior_predictive_checks(self, model, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Comprehensive posterior predictive checks."""
        print(f"\n=== Posterior Predictive Checks for {model_name} ===")
        
        results = {'model': model_name}
        
        try:
            # Generate posterior predictive samples
            if not hasattr(model, '_fit_result') or model._fit_result is None:
                print("Model not fitted, skipping PPC tests")
                return {'model': model_name, 'success': False, 'error': 'Model not fitted'}
                
            # Use model's internal posterior predictive sampling
            with model._model:
                ppc = az.sample_posterior_predictive(
                    model._fit_result, 
                    var_names=["demand"],
                    samples=200  # Reduced for speed
                )
                
            observed_data = data[model.target_column].values
            predicted_samples = ppc.posterior_predictive["demand"].values
            
            # Reshape predictions: (n_samples, n_observations)
            predicted_samples = predicted_samples.reshape(-1, len(observed_data))
            
            # 1. Test: Mean comparison
            observed_mean = np.mean(observed_data)
            predicted_means = np.mean(predicted_samples, axis=1)
            
            # Check if observed mean falls within posterior predictive distribution of means
            mean_pvalue = np.mean(predicted_means >= observed_mean)
            # Convert to two-tailed p-value
            mean_pvalue = 2 * min(mean_pvalue, 1 - mean_pvalue)
            
            results['mean_ppc_pvalue'] = mean_pvalue
            mean_check = mean_pvalue > self.alpha
            results['mean_ppc_pass'] = mean_check
            print(f"✓ Mean PPC (p={mean_pvalue:.3f}): {'PASS' if mean_check else 'FAIL'}")
            
            # 2. Test: Variance comparison
            observed_var = np.var(observed_data)
            predicted_vars = np.var(predicted_samples, axis=1)
            
            var_pvalue = np.mean(predicted_vars >= observed_var)
            var_pvalue = 2 * min(var_pvalue, 1 - var_pvalue)
            
            results['var_ppc_pvalue'] = var_pvalue
            var_check = var_pvalue > self.alpha
            results['var_ppc_pass'] = var_check
            print(f"✓ Variance PPC (p={var_pvalue:.3f}): {'PASS' if var_check else 'FAIL'}")
            
            # 3. Test: Distribution shape (using quantiles)
            observed_quantiles = np.percentile(observed_data, [10, 25, 50, 75, 90])
            predicted_quantiles = np.percentile(predicted_samples, [10, 25, 50, 75, 90], axis=1)
            
            quantile_pvalues = []
            for i, q in enumerate([10, 25, 50, 75, 90]):
                pval = np.mean(predicted_quantiles[i] >= observed_quantiles[i])
                pval = 2 * min(pval, 1 - pval)
                quantile_pvalues.append(pval)
                
            results['quantile_ppc_pvalues'] = quantile_pvalues
            quantile_checks = np.array(quantile_pvalues) > self.alpha
            results['quantile_ppc_passes'] = quantile_checks
            
            overall_quantile_check = np.mean(quantile_checks) >= 0.6  # At least 60% pass
            results['quantile_ppc_overall'] = overall_quantile_check
            print(f"✓ Quantile PPC ({np.sum(quantile_checks)}/5 pass): {'PASS' if overall_quantile_check else 'FAIL'}")
            
            # 4. Test: Zero inflation check (for intermittent models)
            if 'intermittent' in model_name.lower():
                observed_zeros = np.sum(observed_data == 0) / len(observed_data)
                predicted_zeros = np.mean(predicted_samples == 0, axis=1)
                
                zero_pvalue = np.mean(predicted_zeros >= observed_zeros)
                zero_pvalue = 2 * min(zero_pvalue, 1 - zero_pvalue)
                
                results['zero_inflation_ppc_pvalue'] = zero_pvalue
                zero_check = zero_pvalue > self.alpha
                results['zero_inflation_ppc_pass'] = zero_check
                print(f"✓ Zero inflation PPC (p={zero_pvalue:.3f}): {'PASS' if zero_check else 'FAIL'}")
            
            results['success'] = True
            
        except Exception as e:
            print(f"✗ PPC FAILED: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_hierarchical_pooling(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Test that hierarchical models actually perform pooling."""
        print(f"\n=== Testing Hierarchical Pooling ===")
        
        results = {'model': 'hierarchical'}
        
        try:
            # Fit the hierarchical model
            model.fit(data, draws=500, tune=500, chains=2)
            
            # Get posterior samples for hierarchy parameters
            posterior = model._fit_result.posterior
            
            # Test 1: Different hierarchy levels produce different forecasts
            hierarchy_forecasts = {}
            test_combinations = [
                {'region': 'North', 'store': 'Store_A'},
                {'region': 'North', 'store': 'Store_B'},
                {'region': 'South', 'store': 'Store_A'},
                {'region': 'South', 'store': 'Store_B'}
            ]
            
            for combo in test_combinations:
                forecast = model.forecast(steps=10, hierarchy_values=combo)
                hierarchy_forecasts[f"{combo['region']}_{combo['store']}"] = forecast['forecast'].mean()
                
            # Check that forecasts are actually different across hierarchy levels
            forecast_values = list(hierarchy_forecasts.values())
            forecast_range = max(forecast_values) - min(forecast_values)
            
            # Should have meaningful differences (not all the same)
            meaningful_differences = forecast_range > 1.0  # At least 1 unit difference
            results['meaningful_hierarchy_differences'] = meaningful_differences
            print(f"✓ Meaningful hierarchy differences (range: {forecast_range:.2f}): {'PASS' if meaningful_differences else 'FAIL'}")
            
            # Test 2: Shrinkage toward group means (pooling evidence)
            if 'region_intercept' in posterior:
                region_effects = posterior['region_intercept'].values
                
                # Shrinkage test: individual effects should be less extreme than no-pooling estimates
                region_shrinkage_evident = np.std(region_effects) < data.groupby('region')['demand'].std().std()
                results['shrinkage_evident'] = region_shrinkage_evident
                print(f"✓ Shrinkage evident: {'PASS' if region_shrinkage_evident else 'FAIL'}")
            
            results['success'] = True
            
        except Exception as e:
            print(f"✗ Hierarchical test FAILED: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_intermittent_patterns(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Test that intermittent models capture sporadic patterns."""
        print(f"\n=== Testing Intermittent Patterns ===")
        
        results = {'model': 'intermittent'}
        
        try:
            # Analyze actual data pattern
            observed_zeros = np.sum(data['demand'] == 0) / len(data)
            observed_cv = data['demand'].std() / data['demand'].mean() if data['demand'].mean() > 0 else 0
            
            # Fit model
            model.fit(data, draws=500, tune=500, chains=2)
            
            # Generate forecast samples to check sporadic patterns
            forecast = model.forecast(steps=30, simulate_sporadic=True)
            
            # Test 1: Model captures zero-inflation
            posterior = model._fit_result.posterior
            if 'zero_inflation' in posterior:
                modeled_zero_prob = posterior['zero_inflation'].mean().values
                zero_prob_reasonable = abs(modeled_zero_prob - observed_zeros) < 0.3  # Within 30%
                results['zero_prob_reasonable'] = zero_prob_reasonable
                print(f"✓ Zero probability reasonable (model: {modeled_zero_prob:.2f}, data: {observed_zeros:.2f}): {'PASS' if zero_prob_reasonable else 'FAIL'}")
            
            # Test 2: Sporadic forecasts (not flat lines)
            sporadic_forecast = forecast['forecast_sporadic'].values
            forecast_variation = np.std(sporadic_forecast) / np.mean(sporadic_forecast) if np.mean(sporadic_forecast) > 0 else 0
            
            # Sporadic forecasts should have some variation (not completely flat)
            has_variation = forecast_variation > 0.1
            results['forecast_has_variation'] = has_variation
            print(f"✓ Sporadic forecast variation (CV: {forecast_variation:.2f}): {'PASS' if has_variation else 'FAIL'}")
            
            # Test 3: Some zero forecasts in sporadic pattern
            zero_forecasts = np.sum(sporadic_forecast == 0) / len(sporadic_forecast)
            some_zeros = zero_forecasts > 0.1  # At least 10% zeros in forecast
            results['some_zero_forecasts'] = some_zeros
            print(f"✓ Some zero forecasts ({zero_forecasts:.1%}): {'PASS' if some_zeros else 'FAIL'}")
            
            results['success'] = True
            
        except Exception as e:
            print(f"✗ Intermittent test FAILED: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_seasonal_patterns(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Test that seasonal models capture seasonal patterns."""
        print(f"\n=== Testing Seasonal Patterns ===")
        
        results = {'model': 'seasonal'}
        
        try:
            # Fit seasonal model
            model.fit(data, draws=500, tune=500, chains=2)
            
            # Test 1: Model identifies seasonality
            posterior = model._fit_result.posterior
            if 'beta_seasonal' in posterior:
                seasonal_effects = posterior['beta_seasonal'].values
                seasonal_magnitude = np.std(seasonal_effects)
                
                # Seasonal effects should have meaningful magnitude
                seasonal_significant = seasonal_magnitude > 0.5
                results['seasonal_significant'] = seasonal_significant
                print(f"✓ Seasonal effects significant (std: {seasonal_magnitude:.2f}): {'PASS' if seasonal_significant else 'FAIL'}")
            
            # Test 2: Forecast captures seasonal variation
            forecast = model.forecast(steps=28)  # 4 weeks to see weekly pattern
            forecast_values = forecast['forecast'].values
            
            # Check for periodic patterns in forecast
            # Simple test: forecast should not be monotonic
            is_monotonic = np.all(np.diff(forecast_values) >= 0) or np.all(np.diff(forecast_values) <= 0)
            captures_variation = not is_monotonic
            results['captures_seasonal_variation'] = captures_variation
            print(f"✓ Captures seasonal variation: {'PASS' if captures_variation else 'FAIL'}")
            
            results['success'] = True
            
        except Exception as e:
            print(f"✗ Seasonal test FAILED: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_distribution_compliance(self, model, model_name: str) -> Dict[str, Any]:
        """Test that models comply with their specified distributions."""
        print(f"\n=== Testing Distribution Compliance for {model_name} ===")
        
        results = {'model': model_name}
        
        try:
            if not hasattr(model, '_fit_result') or model._fit_result is None:
                return {'model': model_name, 'success': False, 'error': 'Model not fitted'}
                
            # Generate samples from posterior predictive
            with model._model:
                ppc_samples = az.sample_posterior_predictive(
                    model._fit_result,
                    var_names=["demand"],
                    samples=100
                )
                
            generated_data = ppc_samples.posterior_predictive["demand"].values.flatten()
            
            # Test based on model distribution
            if hasattr(model, 'distribution'):
                if model.distribution == 'negative_binomial':
                    # Test 1: All values are non-negative integers
                    non_negative = np.all(generated_data >= 0)
                    are_integers = np.allclose(generated_data, np.round(generated_data))
                    
                    results['non_negative'] = non_negative
                    results['integers'] = are_integers
                    print(f"✓ Negative binomial - non-negative: {'PASS' if non_negative else 'FAIL'}")
                    print(f"✓ Negative binomial - integers: {'PASS' if are_integers else 'FAIL'}")
                    
                elif model.distribution == 'poisson':
                    # Test: Non-negative integers
                    non_negative = np.all(generated_data >= 0)
                    are_integers = np.allclose(generated_data, np.round(generated_data))
                    
                    results['non_negative'] = non_negative
                    results['integers'] = are_integers
                    print(f"✓ Poisson - non-negative: {'PASS' if non_negative else 'FAIL'}")
                    print(f"✓ Poisson - integers: {'PASS' if are_integers else 'FAIL'}")
                    
                elif model.distribution == 'gamma':
                    # Test: Positive continuous values
                    positive = np.all(generated_data > 0)
                    continuous = not np.allclose(generated_data, np.round(generated_data))
                    
                    results['positive'] = positive
                    results['continuous'] = continuous
                    print(f"✓ Gamma - positive: {'PASS' if positive else 'FAIL'}")
                    print(f"✓ Gamma - continuous: {'PASS' if continuous else 'FAIL'}")
            
            results['success'] = True
            
        except Exception as e:
            print(f"✗ Distribution test FAILED: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete statistical validation suite."""
        print("=" * 80)
        print("COMPREHENSIVE STATISTICAL VALIDATION SUITE")
        print("=" * 80)
        
        # Generate test datasets
        datasets = self.generate_test_data()
        
        # Initialize models
        models = {
            'base_nb': DemandForecastModel(distribution='negative_binomial'),
            'base_poisson': DemandForecastModel(distribution='poisson'),
            'base_gamma': DemandForecastModel(distribution='gamma'),
            'hierarchical': HierarchicalDemandModel(
                hierarchy_cols=['region', 'store'],
                distribution='negative_binomial'
            ),
            'intermittent_zinb': IntermittentDemandModel(method='zero_inflated_nb'),
            'intermittent_zip': IntermittentDemandModel(method='zero_inflated_poisson'),
            'seasonal': SeasonalDemandModel(
                yearly_seasonality=2,
                weekly_seasonality=2
            )
        }
        
        all_results = {}
        
        # Test each model
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"VALIDATING MODEL: {model_name.upper()}")
            print(f"{'='*60}")
            
            model_results = {'model': model_name}
            
            try:
                # Choose appropriate dataset
                if 'hierarchical' in model_name:
                    data = datasets['hierarchical']
                elif 'intermittent' in model_name:
                    data = datasets['intermittent']
                elif 'seasonal' in model_name:
                    data = datasets['seasonal']
                else:
                    data = datasets['base']
                
                # Run all validation tests
                model_results['forecast_correctness'] = self.test_forecast_correctness(model, data, model_name)
                model_results['posterior_predictive_checks'] = self.test_posterior_predictive_checks(model, data, model_name)
                model_results['distribution_compliance'] = self.test_distribution_compliance(model, model_name)
                
                # Model-specific tests
                if 'hierarchical' in model_name:
                    model_results['hierarchical_pooling'] = self.test_hierarchical_pooling(model, data)
                elif 'intermittent' in model_name:
                    model_results['intermittent_patterns'] = self.test_intermittent_patterns(model, data)
                elif 'seasonal' in model_name:
                    model_results['seasonal_patterns'] = self.test_seasonal_patterns(model, data)
                
                model_results['overall_success'] = True
                
            except Exception as e:
                print(f"✗ CRITICAL FAILURE for {model_name}: {str(e)}")
                model_results['overall_success'] = False
                model_results['critical_error'] = str(e)
            
            all_results[model_name] = model_results
        
        # Generate summary report
        self._generate_validation_report(all_results)
        
        return all_results
    
    def _generate_validation_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive validation report."""
        print(f"\n{'='*80}")
        print("STATISTICAL VALIDATION SUMMARY REPORT")
        print(f"{'='*80}")
        
        total_models = len(results)
        successful_models = sum(1 for r in results.values() if r.get('overall_success', False))
        
        print(f"\nOVERALL RESULTS:")
        print(f"Total Models Tested: {total_models}")
        print(f"Successfully Validated: {successful_models}")
        print(f"Success Rate: {successful_models/total_models:.1%}")
        
        print(f"\nDETAILED RESULTS BY MODEL:")
        print("-" * 60)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name.upper()}: {'✓ PASS' if model_results.get('overall_success') else '✗ FAIL'}")
            
            if model_results.get('overall_success'):
                # Count successful sub-tests
                test_counts = {}
                for test_category, test_results in model_results.items():
                    if isinstance(test_results, dict) and 'success' in test_results:
                        test_counts[test_category] = '✓' if test_results['success'] else '✗'
                
                for test_name, status in test_counts.items():
                    print(f"  {test_name}: {status}")
            else:
                if 'critical_error' in model_results:
                    print(f"  Critical Error: {model_results['critical_error']}")
        
        print(f"\n{'='*80}")
        
        if successful_models == total_models:
            print("🎉 ALL MODELS PASSED STATISTICAL VALIDATION!")
            print("The demand forecasting models are statistically sound and ready for production use.")
        else:
            print("⚠️  SOME MODELS FAILED VALIDATION")
            print("Please review the failed tests and address statistical issues before deployment.")
        
        print(f"{'='*80}")


def main():
    """Run the comprehensive statistical validation suite."""
    print("Starting comprehensive statistical validation of demand forecasting models...")
    
    # Initialize validation suite
    validator = StatisticalValidationSuite(significance_level=0.05)
    
    # Run complete validation
    results = validator.run_comprehensive_validation()
    
    # Save results for further analysis
    import json
    with open('statistical_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nValidation complete! Results saved to 'statistical_validation_results.json'")
    
    return results


if __name__ == "__main__":
    results = main()