"""Base demand forecasting model."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.transforms import Interval
from pymc.distributions import NegativeBinomial, Poisson, Gamma

from pymc_supply_chain.base import SupplyChainModelBuilder


class DemandForecastModel(SupplyChainModelBuilder):
    """Base Bayesian demand forecasting model with uncertainty quantification.
    
    This model provides a foundation for demand forecasting with:
    - Trend components
    - Seasonality
    - External regressors
    - Uncertainty quantification
    
    Attributes
    ----------
    date_column : str
        Name of the date column
    target_column : str
        Name of the demand/sales column
    seasonality : int
        Seasonal period (e.g., 12 for monthly, 52 for weekly)
    include_trend : bool
        Whether to include trend component
    """
    
    def __init__(
        self,
        date_column: str = "date",
        target_column: str = "demand",
        seasonality: Optional[int] = None,
        include_trend: bool = True,
        include_seasonality: bool = True,
        external_regressors: Optional[List[str]] = None,
        distribution: str = "negative_binomial",
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize demand forecast model.
        
        Parameters
        ----------
        date_column : str
            Column name for dates
        target_column : str  
            Column name for demand values
        seasonality : int, optional
            Seasonal period (auto-detected if None)
        include_trend : bool
            Whether to model trend
        include_seasonality : bool
            Whether to model seasonality
        external_regressors : list of str, optional
            Names of external regressor columns
        distribution : str
            Distribution for demand: 'negative_binomial', 'poisson', 'gamma', 'normal'
        model_config : dict, optional
            Model configuration
        sampler_config : dict, optional
            Sampler configuration
        """
        super().__init__(model_config, sampler_config)
        self.date_column = date_column
        self.target_column = target_column
        self.seasonality = seasonality
        self.include_trend = include_trend
        self.include_seasonality = include_seasonality
        self.external_regressors = external_regressors or []
        self.distribution = distribution
        
        # Validate distribution
        valid_distributions = ['negative_binomial', 'poisson', 'gamma', 'normal']
        if distribution not in valid_distributions:
            raise ValueError(f"Distribution must be one of {valid_distributions}")
        
    def _detect_seasonality(self, dates: pd.Series) -> int:
        """Auto-detect seasonality from date frequency."""
        freq = pd.infer_freq(dates)
        if freq:
            if freq.startswith('D'):
                return 7  # Weekly seasonality for daily data
            elif freq.startswith('W'):
                return 52  # Yearly seasonality for weekly data
            elif freq.startswith('M'):
                return 12  # Yearly seasonality for monthly data
        return 12  # Default to monthly
        
    def build_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pm.Model:
        """Build the PyMC demand forecasting model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features including date and external regressors
        y : pd.Series, optional
            Demand values (can also be in X)
            
        Returns
        -------
        pm.Model
            The PyMC model
        """
        # Extract demand if not provided separately
        if y is None:
            y = X[self.target_column]
            
        # Process dates
        dates = pd.to_datetime(X[self.date_column])
        t = np.arange(len(dates))
        
        # Auto-detect seasonality if needed
        if self.seasonality is None and self.include_seasonality:
            self.seasonality = self._detect_seasonality(dates)
            
        coords = {
            "time": dates,
            "obs_id": np.arange(len(y)),
        }
        
        if self.include_seasonality and self.seasonality:
            coords["season"] = np.arange(self.seasonality)
            
        with pm.Model(coords=coords) as model:
            # Data containers
            demand_obs = pm.Data("demand_obs", y.values)
            time_idx = pm.Data("time_idx", t)
            
            # Intercept
            intercept = pm.Normal("intercept", mu=y.mean(), sigma=y.std())
            
            # Trend component
            if self.include_trend:
                trend_coef = pm.Normal("trend_coef", mu=0, sigma=0.1)
                trend = trend_coef * time_idx
            else:
                trend = 0
                
            # Seasonal component
            if self.include_seasonality and self.seasonality:
                season_idx = pm.Data(
                    "season_idx", 
                    t % self.seasonality
                )
                seasonal_effects = pm.Normal(
                    "seasonal_effects",
                    mu=0,
                    sigma=1,
                    dims="season"
                )
                seasonality = seasonal_effects[season_idx]
            else:
                seasonality = 0
                
            # External regressors
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
                external_effect = pm.math.dot(X_reg, beta)
            else:
                external_effect = 0
                
            # Combine components
            mu = intercept + trend + seasonality + external_effect
            
            # Likelihood based on chosen distribution
            if self.distribution == "negative_binomial":
                # Use log link to ensure mu > 0
                mu_pos = pm.math.exp(mu)
                
                alpha = pm.Exponential("alpha", 1.0)  # Dispersion parameter
                pm.NegativeBinomial(
                    "demand",
                    mu=mu_pos,
                    alpha=alpha,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            elif self.distribution == "poisson":
                # Use log link to ensure mu > 0  
                mu_pos = pm.math.exp(mu)
                
                pm.Poisson(
                    "demand",
                    mu=mu_pos,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            elif self.distribution == "gamma":
                # Use log link to ensure mu > 0
                mu_pos = pm.math.exp(mu)
                
                sigma = pm.HalfNormal("sigma", sigma=y.std())
                pm.Gamma(
                    "demand",
                    mu=mu_pos,
                    sigma=sigma,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            else:  # normal (kept for backwards compatibility)
                sigma = pm.HalfNormal("sigma", sigma=y.std())
                pm.Normal(
                    "demand",
                    mu=mu,
                    sigma=sigma,
                    observed=demand_obs,
                    dims="obs_id"
                )
            
        return model
    
    def forecast(
        self,
        steps: int,
        X_future: Optional[pd.DataFrame] = None,
        frequency: Optional[str] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        """Generate future demand forecasts using proper PyMC patterns.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        X_future : pd.DataFrame, optional
            Future values of external regressors
        frequency : str, optional
            Pandas frequency string for future dates
        include_history : bool
            Whether to include historical fitted values
            
        Returns
        -------
        pd.DataFrame
            Forecast results with credible intervals
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
        
        # Use proper PyMC forecasting with pm.set_data and sample_posterior_predictive
        with self._model:
            # Update data containers for forecasting
            pm.set_data({
                "time_idx": t_future,
            })
            
            # Update seasonality if used
            if self.include_seasonality and self.seasonality:
                pm.set_data({"season_idx": t_future % self.seasonality})
            
            # Update external regressors if present
            if self.external_regressors and X_reg_future is not None:
                pm.set_data({"X_reg": X_reg_future})
            
            # Set dummy observed data for prediction (required by PyMC)
            pm.set_data({"demand_obs": np.zeros(steps)})  # Will be ignored in prediction
            
            # Sample posterior predictive
            posterior_predictive = pm.sample_posterior_predictive(
                self._fit_result,
                var_names=["demand"],
                progressbar=False,
                predictions=True  # This tells PyMC we're making predictions
            )
            
        # Extract forecast results
        forecast_samples = posterior_predictive.predictions["demand"]
        
        # Calculate summary statistics
        forecast_mean = forecast_samples.mean(dim=["chain", "draw"]).values
        forecast_std = forecast_samples.std(dim=["chain", "draw"]).values
        
        # Calculate prediction intervals
        forecast_lower = forecast_samples.quantile(0.025, dim=["chain", "draw"]).values
        forecast_upper = forecast_samples.quantile(0.975, dim=["chain", "draw"]).values
        
        # Ensure non-negative forecasts for count/demand data
        if self.distribution in ["negative_binomial", "poisson", "gamma"]:
            forecast_lower = np.maximum(forecast_lower, 0)
            forecast_mean = np.maximum(forecast_mean, 0)
        
        # Create results DataFrame
        results = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_mean,
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
            "forecast_std": forecast_std,
        })
        
        return results
    
    def plot_forecast(
        self,
        forecast_df: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None,
        title: str = "Demand Forecast",
    ):
        """Plot forecast with uncertainty bands.
        
        Parameters
        ----------
        forecast_df : pd.DataFrame
            Forecast results from forecast()
        historical_data : pd.DataFrame, optional
            Historical demand data
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data if provided
        if historical_data is not None:
            ax.plot(
                historical_data[self.date_column],
                historical_data[self.target_column],
                'k.',
                label='Historical',
                alpha=0.5
            )
            
        # Plot forecast
        ax.plot(
            forecast_df['date'],
            forecast_df['forecast'],
            'b-',
            label='Forecast',
            linewidth=2
        )
        
        # Plot uncertainty bands
        ax.fill_between(
            forecast_df['date'],
            forecast_df['forecast_lower'],
            forecast_df['forecast_upper'],
            alpha=0.3,
            color='blue',
            label='95% Credible Interval'
        )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand')
        ax.set_title(title)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig, ax