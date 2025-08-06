"""Hierarchical demand forecasting for multi-level supply chains."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc_supply_chain.demand.base import DemandForecastModel


class HierarchicalDemandModel(DemandForecastModel):
    """Hierarchical Bayesian demand model for multi-location/product forecasting.
    
    This model handles:
    - Multiple products across multiple locations
    - Hierarchical structure (e.g., SKU -> Category -> Total)
    - Adaptive partial pooling learned from data
    - Cross-location and cross-product learning
    
    Attributes
    ----------
    hierarchy_cols : list
        Column names defining hierarchy (e.g., ['region', 'store', 'product'])
    distribution : str
        Distribution for demand modeling (prevents negative forecasts)
    """
    
    def __init__(
        self,
        hierarchy_cols: List[str],
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
        """Initialize hierarchical demand model.
        
        Parameters
        ----------
        hierarchy_cols : list of str
            Columns defining the hierarchy (e.g., ['region', 'store', 'product'])
        distribution : str
            Distribution for demand: 'negative_binomial', 'poisson', 'gamma', 'normal'
        Other parameters as in DemandForecastModel
        """
        super().__init__(
            date_column=date_column,
            target_column=target_column,
            seasonality=seasonality,
            include_trend=include_trend,
            include_seasonality=include_seasonality,
            external_regressors=external_regressors,
            distribution=distribution,
            model_config=model_config,
            sampler_config=sampler_config,
        )
        self.hierarchy_cols = hierarchy_cols
        self._hierarchy_mapping = {}
        
    def _create_hierarchy_mapping(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create mappings for hierarchical structure."""
        mappings = {}
        
        for i, col in enumerate(self.hierarchy_cols):
            # Get unique values and create encoding
            unique_vals = X[col].unique()
            mappings[f"{col}_vals"] = unique_vals
            mappings[f"{col}_idx"] = pd.Categorical(X[col]).codes
            
            # Store for later use
            self._hierarchy_mapping[col] = {
                val: idx for idx, val in enumerate(unique_vals)
            }
            
        return mappings
    
    def build_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pm.Model:
        """Build hierarchical Bayesian demand model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features including hierarchy columns
        y : pd.Series, optional
            Demand values
            
        Returns
        -------
        pm.Model
            The hierarchical PyMC model
        """
        if y is None:
            y = X[self.target_column]
            
        # Process dates and hierarchy
        dates = pd.to_datetime(X[self.date_column])
        t = np.arange(len(dates))
        hierarchy_mapping = self._create_hierarchy_mapping(X)
        
        # Auto-detect seasonality
        if self.seasonality is None and self.include_seasonality:
            self.seasonality = self._detect_seasonality(dates)
            
        # Set up coordinates
        coords = {
            "time": dates,
            "obs_id": np.arange(len(y)),
        }
        
        # Add hierarchy coordinates
        for col in self.hierarchy_cols:
            coords[col] = hierarchy_mapping[f"{col}_vals"]
            
        if self.include_seasonality and self.seasonality:
            coords["season"] = np.arange(self.seasonality)
            
        with pm.Model(coords=coords) as model:
            # Data containers
            demand_obs = pm.Data("demand_obs", y.values)
            time_idx = pm.Data("time_idx", t)
            
            # Hierarchy indices
            hierarchy_idx = {}
            for col in self.hierarchy_cols:
                hierarchy_idx[col] = pm.Data(
                    f"{col}_idx",
                    hierarchy_mapping[f"{col}_idx"]
                )
            
            # Global parameters
            global_intercept = pm.Normal("global_intercept", mu=y.mean(), sigma=y.std())
            global_sigma = pm.HalfNormal("global_sigma", sigma=1)
            
            # Hierarchical intercepts
            intercepts = {}
            for i, col in enumerate(self.hierarchy_cols):
                # Hyperpriors for this level
                mu_hyper = pm.Normal(f"{col}_mu_hyper", mu=0, sigma=1)
                sigma_hyper = pm.HalfNormal(f"{col}_sigma_hyper", sigma=1)
                
                # Level-specific effects (let model learn pooling strength)
                intercepts[col] = pm.Normal(
                    f"{col}_intercept",
                    mu=mu_hyper,
                    sigma=sigma_hyper,
                    dims=col
                )
            
            # Trend component (can vary by hierarchy)
            if self.include_trend:
                # Global trend
                global_trend = pm.Normal("global_trend", mu=0, sigma=0.1)
                
                # Hierarchical trend adjustments
                trend_effects = {}
                for col in self.hierarchy_cols[:1]:  # Only top level for simplicity
                    trend_sigma = pm.HalfNormal(f"{col}_trend_sigma", sigma=0.05)
                    trend_effects[col] = pm.Normal(
                        f"{col}_trend",
                        mu=0,
                        sigma=trend_sigma,
                        dims=col
                    )
                    
                # Combine trends
                trend = global_trend * time_idx
                for col, effect in trend_effects.items():
                    trend = trend + effect[hierarchy_idx[col]] * time_idx
            else:
                trend = 0
                
            # Seasonal component (shared across hierarchy)
            if self.include_seasonality and self.seasonality:
                season_idx = pm.Data("season_idx", t % self.seasonality)
                
                # Global seasonality
                seasonal_effects = pm.Normal(
                    "seasonal_effects",
                    mu=0,
                    sigma=1,
                    dims="season"
                )
                seasonality = seasonal_effects[season_idx]
            else:
                seasonality = 0
                
            # External regressors with hierarchical coefficients
            if self.external_regressors:
                external_effect = 0
                for reg in self.external_regressors:
                    # Global coefficient
                    global_beta = pm.Normal(f"beta_{reg}_global", mu=0, sigma=1)
                    
                    # Hierarchical adjustments (for top level only)
                    col = self.hierarchy_cols[0]
                    beta_sigma = pm.HalfNormal(f"beta_{reg}_{col}_sigma", sigma=0.5)
                    beta_offset = pm.Normal(
                        f"beta_{reg}_{col}",
                        mu=0,
                        sigma=beta_sigma,
                        dims=col
                    )
                    
                    X_reg = pm.Data(f"X_{reg}", X[reg].values)
                    external_effect += (global_beta + beta_offset[hierarchy_idx[col]]) * X_reg
            else:
                external_effect = 0
                
            # Combine all components
            mu = global_intercept + trend + seasonality + external_effect
            
            # Add hierarchical intercepts
            for col in self.hierarchy_cols:
                mu = mu + intercepts[col][hierarchy_idx[col]]
                
            # Likelihood based on chosen distribution
            if self.distribution == "negative_binomial":
                # Use log link to ensure mu > 0
                log_mu = pm.Deterministic("log_mu", mu)
                mu_pos = pm.math.exp(log_mu)
                
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
                log_mu = pm.Deterministic("log_mu", mu)
                mu_pos = pm.math.exp(log_mu)
                
                pm.Poisson(
                    "demand",
                    mu=mu_pos,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            elif self.distribution == "gamma":
                # Use log link to ensure mu > 0
                log_mu = pm.Deterministic("log_mu", mu)
                mu_pos = pm.math.exp(log_mu)
                
                sigma = pm.HalfNormal("sigma", sigma=y.std())
                pm.Gamma(
                    "demand",
                    mu=mu_pos,
                    sigma=sigma,
                    observed=demand_obs,
                    dims="obs_id"
                )
                
            else:  # normal (kept for backwards compatibility)
                noise_sigma = pm.HalfNormal("noise_sigma", sigma=y.std())
                pm.Normal(
                    "demand",
                    mu=mu,
                    sigma=noise_sigma,
                    observed=demand_obs,
                    dims="obs_id"
                )
            
        return model
    
    def forecast(
        self,
        steps: int,
        hierarchy_values: Dict[str, Any],
        X_future: Optional[pd.DataFrame] = None,
        frequency: Optional[str] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        """Generate hierarchical demand forecasts preserving hierarchy structure.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        hierarchy_values : dict
            Specific hierarchy level values to forecast for (e.g., {'region': 'North', 'store': 'A'})
        X_future : pd.DataFrame, optional
            Future values of external regressors
        frequency : str, optional
            Pandas frequency string for future dates
        include_history : bool
            Whether to include historical fitted values
            
        Returns
        -------
        pd.DataFrame
            Forecast results with credible intervals for the specific hierarchy level
        """
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted before forecasting")
            
        # Validate hierarchy values
        missing_cols = set(self.hierarchy_cols) - set(hierarchy_values.keys())
        if missing_cols:
            raise ValueError(f"Missing hierarchy values for: {missing_cols}")
            
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
        
        # Get hierarchy indices for the specified values
        hierarchy_indices = {}
        for col, value in hierarchy_values.items():
            if value not in self._hierarchy_mapping[col]:
                raise ValueError(f"Unknown {col} value: {value}")
            hierarchy_indices[col] = self._hierarchy_mapping[col][value]
        
        # Prepare external regressors if provided
        if X_future is not None and self.external_regressors:
            if len(X_future) != steps:
                raise ValueError(f"X_future must have {steps} rows")
            X_reg_future = X_future[self.external_regressors].values
        else:
            X_reg_future = np.zeros((steps, len(self.external_regressors))) if self.external_regressors else None
        
        # Create dummy hierarchy indices for future periods
        future_hierarchy_data = {}
        for col in self.hierarchy_cols:
            future_hierarchy_data[f"{col}_idx"] = np.full(steps, hierarchy_indices[col])
        
        # Use proper PyMC forecasting
        with self._model:
            # Update data containers
            pm.set_data({
                "time_idx": t_future,
                "season_idx": t_future % self.seasonality if (self.include_seasonality and self.seasonality) else t_future,
                **future_hierarchy_data
            })
            
            # Update external regressors if present
            if self.external_regressors and X_reg_future is not None:
                for i, reg in enumerate(self.external_regressors):
                    pm.set_data({f"X_{reg}": X_reg_future[:, i]})
            
            # Set dummy observed data
            pm.set_data({"demand_obs": np.zeros(steps)})
            
            # Sample posterior predictive
            posterior_predictive = pm.sample_posterior_predictive(
                self._fit_result,
                var_names=["demand"],
                progressbar=False,
                predictions=True
            )
            
        # Extract forecast results
        forecast_samples = posterior_predictive.predictions["demand"]
        
        # Calculate summary statistics
        forecast_mean = forecast_samples.mean(dim=["chain", "draw"]).values
        forecast_std = forecast_samples.std(dim=["chain", "draw"]).values
        forecast_lower = forecast_samples.quantile(0.025, dim=["chain", "draw"]).values
        forecast_upper = forecast_samples.quantile(0.975, dim=["chain", "draw"]).values
        
        # Ensure non-negative forecasts for count/demand data
        if self.distribution in ["negative_binomial", "poisson", "gamma"]:
            forecast_lower = np.maximum(forecast_lower, 0)
            forecast_mean = np.maximum(forecast_mean, 0)
        
        # Create results DataFrame with hierarchy information
        results = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_mean,
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
            "forecast_std": forecast_std,
        })
        
        # Add hierarchy level information
        for col, value in hierarchy_values.items():
            results[col] = value
        
        return results

    def forecast_hierarchical(
        self,
        steps: int,
        hierarchy_values: Dict[str, List[Any]],
        X_future: Optional[pd.DataFrame] = None,
        frequency: Optional[str] = None,
        aggregate_levels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate hierarchical forecasts with reconciliation.
        
        Parameters
        ----------
        steps : int
            Forecast horizon
        hierarchy_values : dict
            Values for each hierarchy level to forecast
        X_future : pd.DataFrame, optional
            Future external regressors
        frequency : str, optional
            Date frequency
        aggregate_levels : list, optional
            Levels to aggregate forecasts
            
        Returns
        -------
        pd.DataFrame
            Hierarchical forecasts with reconciliation
        """
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted before forecasting")
            
        results = []
        
        # Generate forecasts for each combination
        for combo in self._generate_hierarchy_combinations(hierarchy_values):
            # Create future data for this combination
            future_data = self._create_future_data(steps, combo, X_future, frequency)
            
            # Get base forecast
            forecast = self.forecast(
                steps=steps,
                X_future=future_data,
                frequency=frequency
            )
            
            # Add hierarchy information
            for col, val in combo.items():
                forecast[col] = val
                
            results.append(forecast)
            
        # Combine all forecasts
        all_forecasts = pd.concat(results, ignore_index=True)
        
        # Reconcile if requested
        if aggregate_levels:
            all_forecasts = self._reconcile_forecasts(
                all_forecasts,
                aggregate_levels
            )
            
        return all_forecasts
    
    def _generate_hierarchy_combinations(
        self,
        hierarchy_values: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of hierarchy values."""
        import itertools
        
        keys = list(hierarchy_values.keys())
        values = [hierarchy_values[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
            
        return combinations
    
    def _reconcile_forecasts(
        self,
        forecasts: pd.DataFrame,
        aggregate_levels: List[str]
    ) -> pd.DataFrame:
        """Reconcile forecasts to ensure coherence across hierarchy."""
        # Simple bottom-up reconciliation
        # More sophisticated methods (MinT, etc.) can be added
        
        reconciled = forecasts.copy()
        
        for level in aggregate_levels:
            # Aggregate to this level
            group_cols = [col for col in self.hierarchy_cols if col != level]
            if group_cols:
                aggregated = forecasts.groupby(group_cols + ['date']).agg({
                    'forecast': 'sum',
                    'forecast_lower': 'sum',
                    'forecast_upper': 'sum',
                    'forecast_std': lambda x: np.sqrt(np.sum(x**2))
                }).reset_index()
                
                aggregated[level] = 'Total'
                reconciled = pd.concat([reconciled, aggregated], ignore_index=True)
                
        return reconciled