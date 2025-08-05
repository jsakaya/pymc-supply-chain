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
        method: str = "croston",
        min_periods: int = 2,
        smoothing_param: Optional[float] = None,
        external_regressors: Optional[list] = None,
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize intermittent demand model.
        
        Parameters
        ----------
        method : str
            Method to use: 'croston', 'sba', 'zero_inflated'
        min_periods : int
            Minimum non-zero periods required
        smoothing_param : float, optional
            Smoothing parameter (estimated if None)
        """
        super().__init__(
            date_column=date_column,
            target_column=target_column,
            external_regressors=external_regressors,
            model_config=model_config,
            sampler_config=sampler_config,
        )
        self.method = method
        self.min_periods = min_periods
        self.smoothing_param = smoothing_param
        
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
        """Build Bayesian intermittent demand model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series, optional
            Demand values
            
        Returns
        -------
        pm.Model
            PyMC model for intermittent demand
        """
        if y is None:
            y = X[self.target_column]
            
        intermittent_data = self._prepare_intermittent_data(y)
        
        if self.method == "croston":
            return self._build_croston_model(X, y, intermittent_data)
        elif self.method == "sba":
            return self._build_sba_model(X, y, intermittent_data)
        elif self.method == "zero_inflated":
            return self._build_zero_inflated_model(X, y, intermittent_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
    def _build_croston_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data: Dict[str, np.ndarray]
    ) -> pm.Model:
        """Build Bayesian Croston's method model."""
        coords = {
            "obs_id": np.arange(data["n_periods"]),
            "non_zero_id": np.arange(data["n_non_zero"]),
        }
        
        with pm.Model(coords=coords) as model:
            # Smoothing parameter
            if self.smoothing_param is None:
                alpha = pm.Beta("alpha", alpha=2, beta=2)
            else:
                alpha = pm.Data("alpha", self.smoothing_param)
                
            # Model for demand size when non-zero
            demand_size_mu = pm.Exponential("demand_size_mu", 1.0)
            demand_size_sigma = pm.HalfNormal("demand_size_sigma", sigma=10)
            
            # Observed non-zero demands
            pm.Gamma(
                "demand_size",
                mu=demand_size_mu,
                sigma=demand_size_sigma,
                observed=data["non_zero_demands"],
                dims="non_zero_id"
            )
            
            # Model for inter-arrival times
            if data["n_non_zero"] > 1:
                interval_rate = pm.Exponential("interval_rate", 1.0)
                pm.Exponential(
                    "intervals",
                    interval_rate,
                    observed=data["inter_arrival_times"]
                )
                
                # Expected interval
                expected_interval = pm.Deterministic(
                    "expected_interval",
                    1.0 / interval_rate
                )
            else:
                expected_interval = pm.Data(
                    "expected_interval",
                    data["n_periods"]
                )
                
            # Croston's forecast
            demand_rate = pm.Deterministic(
                "demand_rate",
                demand_size_mu / expected_interval
            )
            
            # For posterior predictive
            # Model full time series as zero-inflated
            p_demand = pm.Deterministic(
                "p_demand",
                1.0 / expected_interval
            )
            
            # Likelihood for full series
            demand_obs = pm.Data("demand_obs", y.values)
            
            # Zero-inflated Gamma
            pm.ZeroInflatedNegativeBinomial(
                "demand",
                psi=1 - p_demand,
                mu=demand_size_mu,
                alpha=demand_size_sigma,
                observed=demand_obs,
                dims="obs_id"
            )
            
        return model
    
    def _build_sba_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data: Dict[str, np.ndarray]
    ) -> pm.Model:
        """Build Syntetos-Boylan Approximation model."""
        # Similar to Croston but with bias correction
        coords = {
            "obs_id": np.arange(data["n_periods"]),
            "non_zero_id": np.arange(data["n_non_zero"]),
        }
        
        with pm.Model(coords=coords) as model:
            # Smoothing parameter
            if self.smoothing_param is None:
                alpha = pm.Beta("alpha", alpha=2, beta=2)
            else:
                alpha = pm.Data("alpha", self.smoothing_param)
                
            # Demand size model
            demand_size_mu = pm.Exponential("demand_size_mu", 1.0)
            demand_size_sigma = pm.HalfNormal("demand_size_sigma", sigma=10)
            
            pm.Gamma(
                "demand_size",
                mu=demand_size_mu,
                sigma=demand_size_sigma,
                observed=data["non_zero_demands"],
                dims="non_zero_id"
            )
            
            # Interval model
            if data["n_non_zero"] > 1:
                interval_mu = pm.Exponential("interval_mu", 1.0)
                pm.Exponential(
                    "intervals",
                    1.0 / interval_mu,
                    observed=data["inter_arrival_times"]
                )
            else:
                interval_mu = pm.Data("interval_mu", data["n_periods"])
                
            # SBA bias correction factor
            bias_factor = pm.Deterministic(
                "bias_factor",
                1 - alpha / 2
            )
            
            # SBA forecast
            demand_rate = pm.Deterministic(
                "demand_rate",
                bias_factor * demand_size_mu / interval_mu
            )
            
            # Full series likelihood
            p_demand = 1.0 / interval_mu
            demand_obs = pm.Data("demand_obs", y.values)
            
            pm.ZeroInflatedNegativeBinomial(
                "demand",
                psi=1 - p_demand,
                mu=demand_size_mu,
                alpha=demand_size_sigma,
                observed=demand_obs,
                dims="obs_id"
            )
            
        return model
    
    def _build_zero_inflated_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data: Dict[str, np.ndarray]
    ) -> pm.Model:
        """Build zero-inflated Poisson/NegBin model."""
        dates = pd.to_datetime(X[self.date_column])
        t = np.arange(len(dates))
        
        coords = {
            "time": dates,
            "obs_id": np.arange(len(y)),
        }
        
        with pm.Model(coords=coords) as model:
            # Probability of zero (no demand)
            if self.external_regressors:
                # Logistic regression for zero probability
                X_reg = pm.Data(
                    "X_reg",
                    X[self.external_regressors].values
                )
                beta_zero = pm.Normal(
                    "beta_zero",
                    0,
                    1,
                    shape=len(self.external_regressors)
                )
                logit_p = pm.math.dot(X_reg, beta_zero)
                psi = pm.Deterministic("psi", pm.math.sigmoid(logit_p))
            else:
                # Simple probability
                psi = pm.Beta("psi", alpha=1, beta=1)
                
            # Demand intensity when non-zero
            if self.external_regressors:
                beta_intensity = pm.Normal(
                    "beta_intensity",
                    0,
                    1,
                    shape=len(self.external_regressors)
                )
                log_mu = pm.math.dot(X_reg, beta_intensity)
                mu = pm.math.exp(log_mu)
            else:
                mu = pm.Exponential("mu", 1.0)
                
            # Dispersion parameter
            alpha = pm.Exponential("alpha", 1.0)
            
            # Likelihood
            demand_obs = pm.Data("demand_obs", y.values)
            
            pm.ZeroInflatedNegativeBinomial(
                "demand",
                psi=psi,
                mu=mu,
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
    ) -> pd.DataFrame:
        """Generate forecasts for intermittent demand.
        
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
            
        Returns
        -------
        pd.DataFrame
            Forecasts with safety stock recommendations
        """
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted before forecasting")
            
        # Generate future dates manually since this model doesn't use "time" coord
        if frequency is None:
            frequency = 'D'  # Default to daily
            
        # Create a simple forecast DataFrame
        future_dates = pd.date_range(
            start=pd.Timestamp.now(),
            periods=steps,
            freq=frequency
        )
        
        # Extract demand rate for forecast
        posterior = self._fit_result.posterior
        if "demand_rate" in posterior:
            demand_rate = posterior["demand_rate"].mean().values
        else:
            demand_rate = 1.0  # Fallback
            
        # Simple forecast with uncertainty
        forecast_mean = np.full(steps, demand_rate)
        forecast_std = np.full(steps, demand_rate * 0.5)  # Simple estimate
        
        forecast_df = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_mean,
            "forecast_lower": forecast_mean - 1.96 * forecast_std,
            "forecast_upper": forecast_mean + 1.96 * forecast_std,
            "forecast_std": forecast_std,
        })
        
        # Add intermittent-specific metrics
        posterior = self._fit_result.posterior
        
        if self.method in ["croston", "sba"]:
            # Extract demand rate
            demand_rate = posterior["demand_rate"].mean().values
            forecast_df["demand_rate"] = demand_rate
            
            # Lead time demand distribution
            if "expected_interval" in posterior:
                expected_interval = posterior["expected_interval"].mean().values
                forecast_df["probability_of_demand"] = 1 / expected_interval
                
        # Safety stock calculation
        forecast_df["safety_stock"] = self._calculate_safety_stock(
            forecast_df,
            service_level
        )
        
        return forecast_df
    
    def _calculate_safety_stock(
        self,
        forecast_df: pd.DataFrame,
        service_level: float
    ) -> np.ndarray:
        """Calculate safety stock for intermittent demand."""
        from scipy import stats
        
        # Simple method: use forecast upper bound
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * forecast_df["forecast_std"].values
        
        return safety_stock