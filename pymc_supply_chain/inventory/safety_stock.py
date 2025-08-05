"""Bayesian safety stock optimization under uncertainty."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

from pymc_supply_chain.base import SupplyChainModelBuilder


class SafetyStockOptimizer(SupplyChainModelBuilder):
    """Bayesian safety stock optimization with demand and lead time uncertainty.
    
    Features:
    - Joint demand and lead time uncertainty
    - Multiple service level definitions
    - Cost-service trade-off optimization
    - Pooling effects for multiple locations
    """
    
    def __init__(
        self,
        holding_cost: float,
        stockout_cost: float,
        service_type: str = "cycle",
        target_service_level: Optional[float] = None,
        lead_time_distribution: str = "fixed",
        demand_distribution: str = "normal",
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize safety stock optimizer.
        
        Parameters
        ----------
        holding_cost : float
            Cost per unit held in safety stock
        stockout_cost : float  
            Cost per unit of stockout
        service_type : str
            'cycle' (Type 1) or 'fill' (Type 2) service
        target_service_level : float, optional
            Target service level (0-1)
        lead_time_distribution : str
            'fixed', 'normal', 'gamma'
        demand_distribution : str
            'normal', 'lognormal', 'gamma', 'negative_binomial'
        """
        super().__init__(model_config, sampler_config)
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.service_type = service_type
        self.target_service_level = target_service_level
        self.lead_time_distribution = lead_time_distribution
        self.demand_distribution = demand_distribution
        
    def build_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pm.Model:
        """Build Bayesian model for demand and lead time.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features including demand and lead time data
        y : pd.Series, optional
            Not used, demand should be in X
            
        Returns
        -------
        pm.Model
            PyMC model
        """
        # Extract demand and lead time data
        demand_data = X["demand"].values
        lead_time_data = X.get("lead_time", pd.Series([1] * len(X))).values
        
        coords = {
            "obs_id": np.arange(len(demand_data)),
        }
        
        with pm.Model(coords=coords) as model:
            # Demand distribution
            if self.demand_distribution == "normal":
                demand_mu = pm.Normal("demand_mu", mu=demand_data.mean(), sigma=demand_data.std())
                demand_sigma = pm.HalfNormal("demand_sigma", sigma=demand_data.std())
                pm.Normal("demand", mu=demand_mu, sigma=demand_sigma, observed=demand_data, dims="obs_id")
                
            elif self.demand_distribution == "lognormal":
                log_demand = np.log(demand_data[demand_data > 0])
                demand_mu = pm.Normal("demand_mu", mu=log_demand.mean(), sigma=log_demand.std())
                demand_sigma = pm.HalfNormal("demand_sigma", sigma=log_demand.std())
                pm.LogNormal("demand", mu=demand_mu, sigma=demand_sigma, observed=demand_data, dims="obs_id")
                
            elif self.demand_distribution == "gamma":
                alpha = pm.Exponential("demand_alpha", 1.0)
                beta = pm.Exponential("demand_beta", 1.0 / demand_data.mean())
                pm.Gamma("demand", alpha=alpha, beta=beta, observed=demand_data, dims="obs_id")
                
            elif self.demand_distribution == "negative_binomial":
                mu = pm.Exponential("demand_mu", 1.0 / demand_data.mean())
                alpha = pm.Exponential("demand_alpha", 1.0)
                pm.NegativeBinomial("demand", mu=mu, alpha=alpha, observed=demand_data, dims="obs_id")
                
            # Lead time distribution
            if self.lead_time_distribution == "fixed":
                lead_time_value = pm.Data("lead_time", lead_time_data.mean())
                
            elif self.lead_time_distribution == "normal":
                lt_mu = pm.Normal("lead_time_mu", mu=lead_time_data.mean(), sigma=lead_time_data.std())
                lt_sigma = pm.HalfNormal("lead_time_sigma", sigma=lead_time_data.std())
                pm.TruncatedNormal("lead_time", mu=lt_mu, sigma=lt_sigma, lower=0, observed=lead_time_data, dims="obs_id")
                
            elif self.lead_time_distribution == "gamma":
                lt_alpha = pm.Exponential("lead_time_alpha", 1.0)
                lt_beta = pm.Exponential("lead_time_beta", 1.0 / lead_time_data.mean())
                pm.Gamma("lead_time", alpha=lt_alpha, beta=lt_beta, observed=lead_time_data, dims="obs_id")
                
        return model
    
    def calculate_safety_stock(
        self,
        confidence_level: Optional[float] = None,
        n_samples: int = 10000,
    ) -> Dict[str, float]:
        """Calculate optimal safety stock using posterior distributions.
        
        Parameters
        ----------
        confidence_level : float, optional
            Override target service level
        n_samples : int
            Number of samples for Monte Carlo
            
        Returns
        -------
        dict
            Safety stock calculations
        """
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted first")
            
        confidence_level = confidence_level or self.target_service_level or 0.95
        
        # Sample from posterior predictive
        with self._model:
            post_pred = pm.sample_posterior_predictive(
                self._fit_result,
                var_names=["demand", "lead_time"] if self.lead_time_distribution != "fixed" else ["demand"],
                predictions=True,
                progressbar=False,
            )
            
        # Extract samples
        demand_samples = post_pred.predictions["demand"].values.flatten()
        
        if self.lead_time_distribution != "fixed":
            lead_time_samples = post_pred.predictions["lead_time"].values.flatten()
        else:
            lead_time_samples = np.ones(len(demand_samples)) * self._model["lead_time"].eval()
            
        # Sample lead time demand
        n_sim = min(n_samples, len(demand_samples))
        ltd_samples = []
        
        for _ in range(n_sim):
            # Random demand and lead time
            lt = np.random.choice(lead_time_samples)
            daily_demands = np.random.choice(demand_samples, size=int(np.ceil(lt)), replace=True)
            ltd = np.sum(daily_demands)
            ltd_samples.append(ltd)
            
        ltd_samples = np.array(ltd_samples)
        
        # Calculate safety stock for different methods
        results = {}
        
        # Method 1: Direct percentile
        safety_stock_percentile = np.percentile(ltd_samples, confidence_level * 100) - np.mean(ltd_samples)
        results["percentile_method"] = safety_stock_percentile
        
        # Method 2: Normal approximation
        ltd_mean = np.mean(ltd_samples)
        ltd_std = np.std(ltd_samples)
        z_score = stats.norm.ppf(confidence_level)
        safety_stock_normal = z_score * ltd_std
        results["normal_approximation"] = safety_stock_normal
        
        # Method 3: Cost optimization
        if self.target_service_level is None:
            safety_stock_optimal = self._optimize_safety_stock_cost(ltd_samples)
            results["cost_optimal"] = safety_stock_optimal
        else:
            results["cost_optimal"] = safety_stock_percentile
            
        # Calculate metrics
        results["lead_time_demand_mean"] = ltd_mean
        results["lead_time_demand_std"] = ltd_std
        results["lead_time_demand_cv"] = ltd_std / ltd_mean if ltd_mean > 0 else 0
        
        # Service levels achieved - create copy of items to avoid iteration over changing dict
        items_to_check = [(method, ss) for method, ss in results.items() 
                         if method.endswith("_method") or method.endswith("_approximation") or method == "cost_optimal"]
        
        for method, ss in items_to_check:
            achieved_sl = np.mean(ltd_samples <= ltd_mean + ss)
            results[f"{method}_service_level"] = achieved_sl
                
        return results
    
    def _optimize_safety_stock_cost(self, ltd_samples: np.ndarray) -> float:
        """Find cost-optimal safety stock level."""
        ltd_mean = np.mean(ltd_samples)
        ltd_std = np.std(ltd_samples)
        
        # Search range
        ss_range = np.linspace(-ltd_std, 4 * ltd_std, 100)
        costs = []
        
        for ss in ss_range:
            # Holding cost
            holding = ss * self.holding_cost
            
            # Expected stockout
            stockouts = np.maximum(ltd_samples - (ltd_mean + ss), 0)
            expected_stockout = np.mean(stockouts) * self.stockout_cost
            
            total_cost = holding + expected_stockout
            costs.append(total_cost)
            
        # Find minimum
        optimal_idx = np.argmin(costs)
        optimal_ss = ss_range[optimal_idx]
        
        return optimal_ss
    
    def pooling_analysis(
        self,
        location_demands: Dict[str, pd.Series],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Analyze safety stock pooling benefits.
        
        Parameters
        ----------
        location_demands : dict
            Demand data by location
        correlation_matrix : array, optional
            Correlation between locations
            
        Returns
        -------
        dict
            Pooling analysis results
        """
        n_locations = len(location_demands)
        locations = list(location_demands.keys())
        
        # Fit individual models
        individual_results = {}
        total_individual_ss = 0
        
        for loc, demand in location_demands.items():
            # Create data
            X_loc = pd.DataFrame({"demand": demand})
            
            # Fit model
            self.fit(X_loc, progressbar=False)
            
            # Calculate safety stock
            ss_result = self.calculate_safety_stock()
            individual_results[loc] = ss_result
            total_individual_ss += ss_result["percentile_method"]
            
        # Pooled demand analysis
        pooled_demand = pd.concat(location_demands.values()).reset_index(drop=True)
        X_pooled = pd.DataFrame({"demand": pooled_demand})
        
        # Fit pooled model
        self.fit(X_pooled, progressbar=False)
        pooled_ss = self.calculate_safety_stock()
        
        # Calculate pooling factor
        if correlation_matrix is None:
            # Estimate from data
            demand_matrix = pd.DataFrame(location_demands).fillna(0)
            correlation_matrix = demand_matrix.corr().values
            
        # Risk pooling factor (square root law with correlation)
        avg_correlation = (np.sum(correlation_matrix) - n_locations) / (n_locations * (n_locations - 1))
        pooling_factor = np.sqrt(n_locations + n_locations * (n_locations - 1) * avg_correlation) / n_locations
        
        theoretical_pooled_ss = total_individual_ss * pooling_factor
        
        results = {
            "individual_safety_stocks": individual_results,
            "total_individual": total_individual_ss,
            "pooled_safety_stock": pooled_ss["percentile_method"],
            "pooling_benefit": total_individual_ss - pooled_ss["percentile_method"],
            "pooling_benefit_pct": (total_individual_ss - pooled_ss["percentile_method"]) / total_individual_ss * 100,
            "theoretical_pooled": theoretical_pooled_ss,
            "pooling_factor": pooling_factor,
            "average_correlation": avg_correlation,
        }
        
        return results
    
    def sensitivity_analysis(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Analyze sensitivity to parameters.
        
        Parameters
        ----------
        param_ranges : dict
            Parameter ranges to test
        n_points : int
            Points per parameter
            
        Returns
        -------
        pd.DataFrame
            Sensitivity results
        """
        results = []
        base_params = {
            "holding_cost": self.holding_cost,
            "stockout_cost": self.stockout_cost,
            "target_service_level": self.target_service_level,
        }
        
        for param, (low, high) in param_ranges.items():
            if param not in base_params:
                continue
                
            values = np.linspace(low, high, n_points)
            
            for value in values:
                # Update parameter
                original = getattr(self, param)
                setattr(self, param, value)
                
                # Calculate safety stock
                ss_result = self.calculate_safety_stock()
                
                results.append({
                    "parameter": param,
                    "value": value,
                    "safety_stock": ss_result["percentile_method"],
                    "total_cost": ss_result["percentile_method"] * self.holding_cost,
                })
                
                # Restore
                setattr(self, param, original)
                
        return pd.DataFrame(results)
    
    def plot_service_cost_tradeoff(self, service_levels: Optional[np.ndarray] = None):
        """Plot service level vs cost trade-off curve."""
        import matplotlib.pyplot as plt
        
        if service_levels is None:
            service_levels = np.linspace(0.8, 0.99, 20)
            
        results = []
        
        for sl in service_levels:
            ss_result = self.calculate_safety_stock(confidence_level=sl)
            ss = ss_result["percentile_method"]
            
            # Calculate expected costs
            holding_cost = ss * self.holding_cost
            
            # Approximate expected stockout
            with self._model:
                post_pred = pm.sample_posterior_predictive(
                    self._fit_result,
                    var_names=["demand"],
                    predictions=True,
                    progressbar=False,
                )
            demand_samples = post_pred.predictions["demand"].values.flatten()
            
            stockout_prob = 1 - sl
            expected_stockout_cost = stockout_prob * np.mean(demand_samples) * self.stockout_cost
            
            total_cost = holding_cost + expected_stockout_cost
            
            results.append({
                "service_level": sl,
                "safety_stock": ss,
                "holding_cost": holding_cost,
                "stockout_cost": expected_stockout_cost,
                "total_cost": total_cost,
            })
            
        df = pd.DataFrame(results)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Safety stock vs service level
        ax1.plot(df["service_level"] * 100, df["safety_stock"], 'b-', linewidth=2)
        ax1.set_xlabel("Service Level (%)")
        ax1.set_ylabel("Safety Stock")
        ax1.set_title("Safety Stock Requirements")
        ax1.grid(True, alpha=0.3)
        
        # Cost components
        ax2.plot(df["service_level"] * 100, df["total_cost"], 'k-', linewidth=2, label="Total")
        ax2.plot(df["service_level"] * 100, df["holding_cost"], 'b--', label="Holding")
        ax2.plot(df["service_level"] * 100, df["stockout_cost"], 'r--', label="Stockout")
        ax2.set_xlabel("Service Level (%)")
        ax2.set_ylabel("Cost")
        ax2.set_title("Cost Trade-offs")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_idx = df["total_cost"].idxmin()
        ax2.axvline(df.loc[optimal_idx, "service_level"] * 100, color='g', linestyle=':', label="Optimal")
        
        plt.tight_layout()
        return fig, df