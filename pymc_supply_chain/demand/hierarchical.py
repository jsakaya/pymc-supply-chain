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
    - Partial pooling for improved estimates
    - Cross-location and cross-product learning
    
    Attributes
    ----------
    hierarchy_cols : list
        Column names defining hierarchy (e.g., ['region', 'store', 'product'])
    pooling_strength : float
        Strength of hierarchical pooling (0=no pooling, 1=complete pooling)
    """
    
    def __init__(
        self,
        hierarchy_cols: List[str],
        date_column: str = "date",
        target_column: str = "demand",
        pooling_strength: float = 0.5,
        seasonality: Optional[int] = None,
        include_trend: bool = True,
        include_seasonality: bool = True,
        external_regressors: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize hierarchical demand model.
        
        Parameters
        ----------
        hierarchy_cols : list of str
            Columns defining the hierarchy
        pooling_strength : float
            Hierarchical pooling strength (0-1)
        Other parameters as in DemandForecastModel
        """
        super().__init__(
            date_column=date_column,
            target_column=target_column,
            seasonality=seasonality,
            include_trend=include_trend,
            include_seasonality=include_seasonality,
            external_regressors=external_regressors,
            model_config=model_config,
            sampler_config=sampler_config,
        )
        self.hierarchy_cols = hierarchy_cols
        self.pooling_strength = pooling_strength
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
                
                # Level-specific effects
                intercepts[col] = pm.Normal(
                    f"{col}_intercept",
                    mu=mu_hyper,
                    sigma=sigma_hyper * (1 - self.pooling_strength),
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
                
            # Observation noise (can vary by group)
            noise_sigma = pm.HalfNormal("noise_sigma", sigma=y.std())
            
            # Likelihood
            pm.Normal(
                "demand",
                mu=mu,
                sigma=noise_sigma,
                observed=demand_obs,
                dims="obs_id"
            )
            
        return model
    
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