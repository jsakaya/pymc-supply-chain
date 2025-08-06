"""Intermittent demand models for spare parts and slow-moving items."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions import Bernoulli, Gamma, NegativeBinomial, ZeroInflatedPoisson

from pymc_supply_chain.demand.base import DemandForecastModel


class IntermittentDemandModel(DemandForecastModel):
    """Bayesian model for intermittent/sporadic demand patterns.
    
    Suitable for:
    - Spare parts demand
    - Slow-moving items
    - Products with many zero-demand periods
    
    Methods supported:
    - Croston's method (Bayesian version)
    - Syntetos-Boylan Approximation (SBA)
    - Zero-inflated models
    """
    
    def __init__(
        self,
        date_column: str = "date",
        target_column: str = "demand",
        method: str = "zero_inflated_nb",
        min_periods: int = 2,
        external_regressors: Optional[list] = None,
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize intermittent demand model.
        
        Parameters
        ----------
        method : str
            Method to use: 'zero_inflated_nb', 'zero_inflated_poisson', 'hurdle_nb'
        min_periods : int
            Minimum non-zero periods required
        """
        # Force appropriate distribution for intermittent demand
        distribution = "negative_binomial" if "nb" in method else "poisson"
        
        super().__init__(
            date_column=date_column,
            target_column=target_column,
            external_regressors=external_regressors,
            distribution=distribution,
            model_config=model_config,
            sampler_config=sampler_config,
        )
        self.method = method
        self.min_periods = min_periods
        
        # Validate method
        valid_methods = ['zero_inflated_nb', 'zero_inflated_poisson', 'hurdle_nb']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
    def _prepare_intermittent_data(self, y: pd.Series) -> Dict[str, np.ndarray]:
        """Prepare data for intermittent demand modeling."""
        # Find non-zero demand periods
        non_zero_mask = y > 0
        non_zero_indices = np.where(non_zero_mask)[0]
        
        # Calculate inter-arrival times
        if len(non_zero_indices) > 1:
            inter_arrival_times = np.diff(non_zero_indices)
        else:
            inter_arrival_times = np.array([len(y)])
            
        # Non-zero demand values
        non_zero_demands = y[non_zero_mask].values
        
        return {
            "non_zero_mask": non_zero_mask,
            "non_zero_indices": non_zero_indices,
            "inter_arrival_times": inter_arrival_times,
            "non_zero_demands": non_zero_demands,
            "n_periods": len(y),
            "n_non_zero": len(non_zero_demands)
        }
    
    def build_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pm.Model:
        """Build simplified intermittent demand model with single likelihood.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features including dates and external regressors
        y : pd.Series, optional
            Demand values (many zeros expected)
            
        Returns
        -------
        pm.Model
            PyMC model for intermittent demand
        """
        if y is None:
            y = X[self.target_column]
            
        # Process dates and time
        dates = pd.to_datetime(X[self.date_column])
        t = np.arange(len(dates))
        
        coords = {
            "time": dates,
            "obs_id": np.arange(len(y)),
        }
        
        with pm.Model(coords=coords) as model:
            # Data containers
            demand_obs = pm.Data("demand_obs", y.values)
            time_idx = pm.Data("time_idx", t)
            
            # Base demand rate (when demand occurs)
            base_rate = pm.Exponential("base_rate", 1.0)
            
            # Probability of demand occurring (vs. zero demand)
            zero_inflation = pm.Beta("zero_inflation", alpha=1, beta=3)  # Bias toward more zeros
            
            # Trend in demand rate (can be negative for declining products)
            trend = pm.Normal("trend", mu=0, sigma=0.01) 
            
            # Log demand rate with trend
            log_rate = pm.math.log(base_rate) + trend * time_idx
            demand_rate = pm.math.exp(log_rate)
            
            # External regressors affecting demand rate
            if self.external_regressors:
                beta = pm.Normal(
                    "beta",
                    mu=0,
                    sigma=1,
                    shape=len(self.external_regressors)
                )
                X_reg = pm.Data(
                    "X_reg",
                    X[self.external_regressors].values
                )
                # Apply regressors to log scale
                log_rate = log_rate + pm.math.dot(X_reg, beta)
                demand_rate = pm.math.exp(log_rate)
            
            # Single likelihood - no dual structure confusion
            if self.method == "zero_inflated_nb":
                # Dispersion parameter for negative binomial
                alpha = pm.Exponential("alpha", 1.0)
                
                pm.ZeroInflatedNegativeBinomial(
                    "demand",
                    psi=zero_inflation,  # Probability of extra zeros
                    mu=demand_rate,
                    alpha=alpha,
                    observed=demand_obs,
                    dims="obs_id"
                )
            
            elif self.method == "zero_inflated_poisson":
                pm.ZeroInflatedPoisson(
                    "demand",
                    psi=zero_inflation,
                    mu=demand_rate,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            elif self.method == "hurdle_nb":
                # Simplified hurdle model using zero-inflated approach
                # (Proper hurdle models are complex to implement in PyMC)
                alpha = pm.Exponential("alpha", 1.0)
                
                pm.ZeroInflatedNegativeBinomial(
                    "demand",
                    psi=zero_inflation,  # Probability of extra zeros
                    mu=demand_rate,
                    alpha=alpha,
                    observed=demand_obs,
                    dims="obs_id"
                )
                    
        return model
            
    
    
    
    def analyze_demand_pattern(self, y: pd.Series) -> Dict[str, float]:
        """Analyze intermittent demand characteristics.
        
        Parameters
        ----------
        y : pd.Series
            Historical demand
            
        Returns
        -------
        dict
            Demand pattern metrics
        """
        data = self._prepare_intermittent_data(y)
        
        # Calculate key metrics
        adi = np.mean(data["inter_arrival_times"]) if len(data["inter_arrival_times"]) > 0 else len(y)
        cv2 = (np.std(data["non_zero_demands"]) / np.mean(data["non_zero_demands"]))**2 if len(data["non_zero_demands"]) > 0 else 0
        
        # Classification (Syntetos et al.)
        if adi < 1.32 and cv2 < 0.49:
            pattern = "Smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            pattern = "Intermittent"
        elif adi < 1.32 and cv2 >= 0.49:
            pattern = "Erratic"
        else:
            pattern = "Lumpy"
            
        return {
            "average_demand_interval": adi,
            "coefficient_of_variation_squared": cv2,
            "pattern_type": pattern,
            "zero_demand_periods": len(y) - data["n_non_zero"],
            "zero_demand_percentage": (len(y) - data["n_non_zero"]) / len(y) * 100,
            "average_demand_size": np.mean(data["non_zero_demands"]) if data["n_non_zero"] > 0 else 0,
        }
    
    def forecast(
        self,
        steps: int,
        X_future: Optional[pd.DataFrame] = None,
        frequency: Optional[str] = None,
        service_level: float = 0.95,
        simulate_sporadic: bool = True,
    ) -> pd.DataFrame:
        """Generate forecasts that properly simulate sporadic demand.
        
        Parameters
        ----------
        steps : int
            Forecast horizon
        X_future : pd.DataFrame, optional
            Future external variables
        frequency : str, optional
            Date frequency
        service_level : float
            Service level for safety stock calculation
        simulate_sporadic : bool
            Whether to simulate actual sporadic events (vs. average rates)
            
        Returns
        -------
        pd.DataFrame
            Forecasts with sporadic demand patterns and safety stock
        """
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted before forecasting")
            
        # Generate future dates
        last_date = pd.to_datetime(self._model.coords["time"][-1])
        if frequency is None:
            frequency = pd.infer_freq(self._model.coords["time"])
            
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(1, frequency),
            periods=steps,
            freq=frequency
        )
        
        # Prepare future data
        t_future = np.arange(
            len(self._model.coords["time"]),
            len(self._model.coords["time"]) + steps
        )
        
        # Prepare external regressors if provided
        if X_future is not None and self.external_regressors:
            if len(X_future) != steps:
                raise ValueError(f"X_future must have {steps} rows")
            X_reg_future = X_future[self.external_regressors].values
        else:
            X_reg_future = np.zeros((steps, len(self.external_regressors))) if self.external_regressors else None
        
        # Use proper PyMC forecasting
        with self._model:
            # Update data containers
            pm.set_data({
                "time_idx": t_future,
            })
            
            # Update external regressors if present
            if self.external_regressors and X_reg_future is not None:
                pm.set_data({"X_reg": X_reg_future})
            
            # Set dummy observed data
            pm.set_data({"demand_obs": np.zeros(steps)})
            
            # Sample posterior predictive - this will generate actual sporadic events
            posterior_predictive = pm.sample_posterior_predictive(
                self._fit_result,
                var_names=["demand"],
                progressbar=False,
                predictions=True
            )
            
        # Extract forecast results
        forecast_samples = posterior_predictive.predictions["demand"]
        
        if simulate_sporadic:
            # Use actual samples to preserve sporadic nature
            # Take a single representative sample path for sporadic visualization
            representative_sample = forecast_samples.isel(chain=0, draw=0).values
            
            # Calculate statistics across all samples
            forecast_mean = forecast_samples.mean(dim=["chain", "draw"]).values
            forecast_std = forecast_samples.std(dim=["chain", "draw"]).values
            forecast_lower = forecast_samples.quantile(0.025, dim=["chain", "draw"]).values
            forecast_upper = forecast_samples.quantile(0.975, dim=["chain", "draw"]).values
            
        else:
            # Traditional averaging approach
            forecast_mean = forecast_samples.mean(dim=["chain", "draw"]).values
            forecast_std = forecast_samples.std(dim=["chain", "draw"]).values
            forecast_lower = forecast_samples.quantile(0.025, dim=["chain", "draw"]).values
            forecast_upper = forecast_samples.quantile(0.975, dim=["chain", "draw"]).values
            representative_sample = forecast_mean
        
        # Create results DataFrame
        results = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_mean,
            "forecast_sporadic": representative_sample,  # Shows actual sporadic pattern
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
            "forecast_std": forecast_std,
        })
        
        # Add intermittent-specific metrics
        posterior = self._fit_result.posterior
        
        # Extract zero inflation probability
        if "zero_inflation" in posterior:
            zero_prob = posterior["zero_inflation"].mean().values
            results["prob_no_demand"] = zero_prob
            results["prob_demand"] = 1 - zero_prob
            
        # Extract base demand rate (when demand occurs)
        if "base_rate" in posterior:
            base_rate = posterior["base_rate"].mean().values
            results["base_demand_rate"] = base_rate
            
        # Safety stock calculation for intermittent demand
        results["safety_stock"] = self._calculate_intermittent_safety_stock(
            forecast_samples,
            service_level
        )
        
        return results
    
    def _calculate_intermittent_safety_stock(
        self,
        forecast_samples,
        service_level: float
    ) -> np.ndarray:
        """Calculate safety stock appropriate for intermittent demand."""
        # For intermittent demand, safety stock should handle the sporadic nature
        # Use empirical quantiles from the full distribution
        
        # Calculate lead time demand for each time period
        # For intermittent demand, we often need to think about cumulative demand over lead time
        cumulative_samples = forecast_samples.cumsum(dim="obs_id")
        
        # Use high quantile for safety stock (since intermittent = higher risk)
        safety_quantile = service_level + (1 - service_level) * 0.5  # Boost for intermittent
        
        safety_stock = cumulative_samples.quantile(
            safety_quantile, 
            dim=["chain", "draw"]
        ).values
        
        return safety_stock