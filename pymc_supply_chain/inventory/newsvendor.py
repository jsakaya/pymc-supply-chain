"""Bayesian Newsvendor model for single-period inventory optimization."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

from pymc_supply_chain.base import OptimizationResult, SupplyChainModelBuilder


class NewsvendorModel(SupplyChainModelBuilder):
    """Bayesian Newsvendor model for single-period inventory decisions.
    
    The newsvendor problem optimizes order quantity for perishable goods
    or single-period items under demand uncertainty.
    
    Features:
    - Demand distribution learning
    - Optimal order quantity with uncertainty
    - Service level constraints
    - Profit/cost optimization
    """
    
    def __init__(
        self,
        unit_cost: float,
        selling_price: float,
        salvage_value: float = 0.0,
        shortage_cost: Optional[float] = None,
        demand_distribution: str = "normal",
        service_level: Optional[float] = None,
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Newsvendor model.
        
        Parameters
        ----------
        unit_cost : float
            Cost per unit ordered
        selling_price : float
            Selling price per unit
        salvage_value : float
            Value of unsold units
        shortage_cost : float, optional
            Penalty cost per unit of unmet demand
        demand_distribution : str
            Distribution type: 'normal', 'lognormal', 'gamma', 'negative_binomial'
        service_level : float, optional
            Target service level (0-1)
        """
        super().__init__(model_config, sampler_config)
        self.unit_cost = unit_cost
        self.selling_price = selling_price
        self.salvage_value = salvage_value
        self.shortage_cost = shortage_cost or (selling_price - unit_cost)
        self.demand_distribution = demand_distribution
        self.service_level = service_level
        
        # Calculate critical ratio
        self.overage_cost = unit_cost - salvage_value
        self.underage_cost = selling_price - unit_cost + self.shortage_cost
        self.critical_ratio = self.underage_cost / (self.underage_cost + self.overage_cost)
        
    def build_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pm.Model:
        """Build Bayesian demand model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (can include external factors)
        y : pd.Series, optional
            Historical demand values
            
        Returns
        -------
        pm.Model
            PyMC model for demand distribution
        """
        if y is None:
            y = X["demand"]
            
        coords = {
            "obs_id": np.arange(len(y)),
        }
        
        with pm.Model(coords=coords) as model:
            # Demand observations
            demand_obs = pm.Data("demand_obs", y.values)
            
            if self.demand_distribution == "normal":
                # Normal distribution
                mu = pm.Normal("mu", mu=y.mean(), sigma=y.std())
                sigma = pm.HalfNormal("sigma", sigma=y.std())
                
                pm.Normal(
                    "demand",
                    mu=mu,
                    sigma=sigma,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            elif self.demand_distribution == "lognormal":
                # Log-normal distribution
                mu_log = pm.Normal("mu_log", mu=np.log(y.mean()), sigma=1)
                sigma_log = pm.HalfNormal("sigma_log", sigma=0.5)
                
                pm.LogNormal(
                    "demand",
                    mu=mu_log,
                    sigma=sigma_log,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            elif self.demand_distribution == "gamma":
                # Gamma distribution
                alpha = pm.Exponential("alpha", 1.0)
                beta = pm.Exponential("beta", 1.0 / y.mean())
                
                pm.Gamma(
                    "demand",
                    alpha=alpha,
                    beta=beta,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            elif self.demand_distribution == "negative_binomial":
                # Negative binomial for count data
                mu = pm.Exponential("mu", 1.0 / y.mean())
                alpha = pm.Exponential("alpha", 1.0)
                
                pm.NegativeBinomial(
                    "demand",
                    mu=mu,
                    alpha=alpha,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            else:
                raise ValueError(f"Unknown distribution: {self.demand_distribution}")
                
        return model
    
    def calculate_optimal_quantity(
        self,
        n_samples: int = 1000,
        include_uncertainty: bool = True,
    ) -> Dict[str, float]:
        """Calculate optimal order quantity using Bayesian posterior.
        
        Parameters
        ----------
        n_samples : int
            Number of samples for Monte Carlo integration
        include_uncertainty : bool
            Whether to include parameter uncertainty
            
        Returns
        -------
        dict
            Optimal quantities and expected profits
        """
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted before optimization")
            
        posterior = self._fit_result.posterior
        
        if include_uncertainty:
            # Sample from posterior predictive
            with self._model:
                post_pred = pm.sample_posterior_predictive(
                    self._fit_result,
                    var_names=["demand"],
                    predictions=True,
                    progressbar=False,
                )
            demand_samples = post_pred.predictions["demand"].values.flatten()
        else:
            # Use posterior mean parameters
            if self.demand_distribution == "normal":
                mu = posterior["mu"].mean().values
                sigma = posterior["sigma"].mean().values
                demand_samples = np.random.normal(mu, sigma, n_samples)
            elif self.demand_distribution == "lognormal":
                mu_log = posterior["mu_log"].mean().values
                sigma_log = posterior["sigma_log"].mean().values
                demand_samples = np.random.lognormal(mu_log, sigma_log, n_samples)
            # Add other distributions as needed
            
        # Calculate optimal quantity
        if self.service_level is not None:
            # Service level constraint
            optimal_q = np.percentile(demand_samples, self.service_level * 100)
        else:
            # Critical ratio optimization
            optimal_q = np.percentile(demand_samples, self.critical_ratio * 100)
            
        # Calculate expected profit for different quantities
        quantities = np.linspace(0, np.max(demand_samples) * 1.5, 100)
        expected_profits = []
        
        for q in quantities:
            profits = self._calculate_profit(q, demand_samples)
            expected_profits.append(np.mean(profits))
            
        best_q_idx = np.argmax(expected_profits)
        best_q = quantities[best_q_idx]
        best_profit = expected_profits[best_q_idx]
        
        # Calculate metrics for optimal quantity
        profits_at_optimal = self._calculate_profit(optimal_q, demand_samples)
        
        results = {
            "optimal_quantity": optimal_q,
            "expected_profit": np.mean(profits_at_optimal),
            "profit_std": np.std(profits_at_optimal),
            "profit_5th_percentile": np.percentile(profits_at_optimal, 5),
            "profit_95th_percentile": np.percentile(profits_at_optimal, 95),
            "stockout_probability": np.mean(demand_samples > optimal_q),
            "expected_leftover": np.mean(np.maximum(optimal_q - demand_samples, 0)),
            "critical_ratio": self.critical_ratio,
            "best_profit_q": best_q,
            "best_expected_profit": best_profit,
        }
        
        return results
    
    def _calculate_profit(self, quantity: float, demand: np.ndarray) -> np.ndarray:
        """Calculate profit for given quantity and demand realizations."""
        sales = np.minimum(quantity, demand)
        leftover = np.maximum(quantity - demand, 0)
        shortage = np.maximum(demand - quantity, 0)
        
        revenue = sales * self.selling_price
        salvage_revenue = leftover * self.salvage_value
        purchase_cost = quantity * self.unit_cost
        shortage_penalty = shortage * self.shortage_cost
        
        profit = revenue + salvage_revenue - purchase_cost - shortage_penalty
        
        return profit
    
    def sensitivity_analysis(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on model parameters.
        
        Parameters
        ----------
        param_ranges : dict
            Parameter ranges to test
        n_points : int
            Number of points per parameter
            
        Returns
        -------
        pd.DataFrame
            Sensitivity analysis results
        """
        results = []
        
        # Save original parameters
        original_params = {
            "unit_cost": self.unit_cost,
            "selling_price": self.selling_price,
            "salvage_value": self.salvage_value,
            "shortage_cost": self.shortage_cost,
        }
        
        for param, (low, high) in param_ranges.items():
            if param not in original_params:
                continue
                
            values = np.linspace(low, high, n_points)
            
            for value in values:
                # Update parameter
                setattr(self, param, value)
                
                # Recalculate critical ratio
                self.overage_cost = self.unit_cost - self.salvage_value
                self.underage_cost = self.selling_price - self.unit_cost + self.shortage_cost
                self.critical_ratio = self.underage_cost / (self.underage_cost + self.overage_cost)
                
                # Calculate optimal quantity
                opt_results = self.calculate_optimal_quantity(n_samples=500)
                
                results.append({
                    "parameter": param,
                    "value": value,
                    "optimal_quantity": opt_results["optimal_quantity"],
                    "expected_profit": opt_results["expected_profit"],
                    "critical_ratio": self.critical_ratio,
                })
                
        # Restore original parameters
        for param, value in original_params.items():
            setattr(self, param, value)
            
        return pd.DataFrame(results)
    
    def plot_profit_function(self, ax=None):
        """Plot expected profit as a function of order quantity."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Generate demand samples
        with self._model:
            post_pred = pm.sample_posterior_predictive(
                self._fit_result,
                var_names=["demand"],
                predictions=True,
                progressbar=False,
            )
        demand_samples = post_pred.predictions["demand"].values.flatten()
        
        # Calculate profits for range of quantities
        quantities = np.linspace(0, np.percentile(demand_samples, 99) * 1.2, 200)
        expected_profits = []
        profit_stds = []
        
        for q in quantities:
            profits = self._calculate_profit(q, demand_samples)
            expected_profits.append(np.mean(profits))
            profit_stds.append(np.std(profits))
            
        expected_profits = np.array(expected_profits)
        profit_stds = np.array(profit_stds)
        
        # Plot
        ax.plot(quantities, expected_profits, 'b-', label='Expected Profit', linewidth=2)
        ax.fill_between(
            quantities,
            expected_profits - 2 * profit_stds,
            expected_profits + 2 * profit_stds,
            alpha=0.3,
            label='95% CI'
        )
        
        # Mark optimal quantity
        opt_results = self.calculate_optimal_quantity()
        ax.axvline(
            opt_results["optimal_quantity"],
            color='r',
            linestyle='--',
            label=f'Optimal Q = {opt_results["optimal_quantity"]:.0f}'
        )
        
        ax.set_xlabel('Order Quantity')
        ax.set_ylabel('Profit')
        ax.set_title('Expected Profit vs Order Quantity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def simulate_periods(
        self,
        n_periods: int,
        order_quantity: Optional[float] = None,
    ) -> pd.DataFrame:
        """Simulate multiple periods with given order quantity.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        order_quantity : float, optional
            Order quantity (uses optimal if None)
            
        Returns
        -------
        pd.DataFrame
            Simulation results
        """
        if order_quantity is None:
            opt_results = self.calculate_optimal_quantity()
            order_quantity = opt_results["optimal_quantity"]
            
        # Generate demand samples
        with self._model:
            post_pred = pm.sample_posterior_predictive(
                self._fit_result,
                var_names=["demand"],
                predictions=True,
                progressbar=False,
                random_seed=42,
            )
            
        # Sample n_periods demands
        all_demands = post_pred.predictions["demand"].values.flatten()
        period_demands = np.random.choice(all_demands, n_periods, replace=True)
        
        # Calculate period results
        results = []
        for i, demand in enumerate(period_demands):
            sales = min(order_quantity, demand)
            leftover = max(order_quantity - demand, 0)
            shortage = max(demand - order_quantity, 0)
            
            profit = self._calculate_profit(order_quantity, np.array([demand]))[0]
            
            results.append({
                "period": i + 1,
                "demand": demand,
                "order_quantity": order_quantity,
                "sales": sales,
                "leftover": leftover,
                "shortage": shortage,
                "profit": profit,
                "service_level": 1 if shortage == 0 else 0,
            })
            
        return pd.DataFrame(results)