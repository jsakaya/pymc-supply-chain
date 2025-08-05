"""Advanced seasonal demand forecasting models."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc_supply_chain.demand.base import DemandForecastModel


class SeasonalDemandModel(DemandForecastModel):
    """Advanced seasonal demand model with multiple seasonality patterns.
    
    Features:
    - Multiple seasonal patterns (daily, weekly, yearly)
    - Holiday effects
    - Fourier series for smooth seasonality
    - Changepoint detection for trend changes
    """
    
    def __init__(
        self,
        date_column: str = "date",
        target_column: str = "demand",
        yearly_seasonality: int = 10,
        weekly_seasonality: int = 3,
        daily_seasonality: Optional[int] = None,
        holidays: Optional[pd.DataFrame] = None,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        n_changepoints: int = 25,
        external_regressors: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize seasonal demand model.
        
        Parameters
        ----------
        yearly_seasonality : int
            Number of Fourier terms for yearly seasonality
        weekly_seasonality : int
            Number of Fourier terms for weekly seasonality
        daily_seasonality : int, optional
            Number of Fourier terms for daily seasonality
        holidays : pd.DataFrame, optional
            DataFrame with 'holiday' and 'ds' columns
        changepoint_prior_scale : float
            Flexibility of trend changepoints
        seasonality_prior_scale : float
            Strength of seasonality
        n_changepoints : int
            Number of potential changepoints
        """
        super().__init__(
            date_column=date_column,
            target_column=target_column,
            external_regressors=external_regressors,
            model_config=model_config,
            sampler_config=sampler_config,
        )
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.n_changepoints = n_changepoints
        
    def _make_seasonality_features(
        self,
        dates: pd.Series,
        period: float,
        fourier_order: int,
        name: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Create Fourier features for seasonality."""
        t = np.array((dates - dates.min()).dt.total_seconds()) / (3600 * 24)  # Days
        features = []
        feature_names = []
        
        for i in range(1, fourier_order + 1):
            features.append(np.sin(2 * np.pi * i * t / period))
            features.append(np.cos(2 * np.pi * i * t / period))
            feature_names.extend([f"{name}_sin_{i}", f"{name}_cos_{i}"])
            
        return np.column_stack(features), feature_names
    
    def _make_holiday_features(
        self,
        dates: pd.Series,
        holidays: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Create holiday indicator features."""
        holiday_dates = pd.to_datetime(holidays['ds']).values
        holiday_names = holidays['holiday'].values
        
        unique_holidays = np.unique(holiday_names)
        features = np.zeros((len(dates), len(unique_holidays)))
        
        for i, holiday in enumerate(unique_holidays):
            holiday_subset = holiday_dates[holiday_names == holiday]
            features[:, i] = np.isin(dates.values, holiday_subset).astype(float)
            
        return features, list(unique_holidays)
    
    def _get_changepoint_matrix(
        self,
        t: np.ndarray,
        n_changepoints: int,
        t_change: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Create changepoint matrix for piecewise linear trend."""
        if t_change is None:
            t_change = np.linspace(0, np.max(t) * 0.8, n_changepoints)
            
        A = (t[:, None] > t_change).astype(float)
        return A
    
    def build_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pm.Model:
        """Build advanced seasonal demand model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features including dates
        y : pd.Series, optional
            Demand values
            
        Returns
        -------
        pm.Model
            PyMC model with advanced seasonality
        """
        if y is None:
            y = X[self.target_column]
            
        dates = pd.to_datetime(X[self.date_column])
        t = np.arange(len(dates))
        t_scaled = t / t.max()
        
        # Create seasonality features
        features_list = []
        feature_names = []
        
        # Yearly seasonality
        if self.yearly_seasonality > 0:
            yearly_features, yearly_names = self._make_seasonality_features(
                dates, 365.25, self.yearly_seasonality, "yearly"
            )
            features_list.append(yearly_features)
            feature_names.extend(yearly_names)
            
        # Weekly seasonality
        if self.weekly_seasonality > 0:
            weekly_features, weekly_names = self._make_seasonality_features(
                dates, 7, self.weekly_seasonality, "weekly"
            )
            features_list.append(weekly_features)
            feature_names.extend(weekly_names)
            
        # Daily seasonality (if applicable)
        if self.daily_seasonality and self.daily_seasonality > 0:
            daily_features, daily_names = self._make_seasonality_features(
                dates, 1, self.daily_seasonality, "daily"
            )
            features_list.append(daily_features)
            feature_names.extend(daily_names)
            
        # Holiday features
        if self.holidays is not None:
            holiday_features, holiday_names = self._make_holiday_features(
                dates, self.holidays
            )
            features_list.append(holiday_features)
            feature_names.extend([f"holiday_{h}" for h in holiday_names])
            
        # Combine all features
        if features_list:
            seasonality_features = np.hstack(features_list)
        else:
            seasonality_features = None
            
        # Changepoint matrix
        changepoint_matrix = self._get_changepoint_matrix(t_scaled, self.n_changepoints)
        
        # Build model
        coords = {
            "time": dates,
            "obs_id": np.arange(len(y)),
            "changepoints": np.arange(self.n_changepoints),
        }
        
        if seasonality_features is not None:
            coords["seasonality_features"] = feature_names
            
        with pm.Model(coords=coords) as model:
            # Data
            demand_obs = pm.Data("demand_obs", y.values)
            t_data = pm.Data("t", t_scaled)
            
            # Trend parameters
            k = pm.Normal("k", 0, 5)  # Base growth rate
            m = pm.Normal("m", y.mean(), y.std())  # Offset
            
            # Changepoints
            delta = pm.Laplace(
                "delta",
                0,
                self.changepoint_prior_scale,
                dims="changepoints"
            )
            
            # Trend with changepoints
            growth = k + pm.math.dot(changepoint_matrix, delta)
            trend = growth * t_data + m
            
            # Seasonality
            if seasonality_features is not None:
                seasonality_data = pm.Data(
                    "seasonality_features",
                    seasonality_features
                )
                beta_seasonal = pm.Normal(
                    "beta_seasonal",
                    0,
                    self.seasonality_prior_scale,
                    dims="seasonality_features"
                )
                seasonality = pm.math.dot(seasonality_data, beta_seasonal)
            else:
                seasonality = 0
                
            # External regressors
            if self.external_regressors:
                beta_external = pm.Normal(
                    "beta_external",
                    0,
                    1,
                    shape=len(self.external_regressors)
                )
                X_external = pm.Data(
                    "X_external",
                    X[self.external_regressors].values
                )
                external_effect = pm.math.dot(X_external, beta_external)
            else:
                external_effect = 0
                
            # Combine components
            mu = trend + seasonality + external_effect
            
            # Observation noise
            sigma = pm.HalfNormal("sigma", sigma=y.std())
            
            # Likelihood
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
        """Generate future demand forecasts.
        
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
        
        # Prepare future time indices
        n_historical = len(self._model.coords["time"])
        t_future = np.arange(n_historical, n_historical + steps)
        t_future_scaled = t_future / n_historical  # Scale similar to training
        
        # Extract posterior samples - use the actual parameter names from our model
        posterior = self._fit_result.posterior
        k_samples = posterior["k"].values.flatten()
        m_samples = posterior["m"].values.flatten()
        sigma_samples = posterior["sigma"].values.flatten()
        
        if "delta" in posterior:
            delta_samples = posterior["delta"].values
        else:
            delta_samples = np.zeros((len(k_samples), self.n_changepoints))
            
        if "beta_seasonal" in posterior:
            beta_seasonal_samples = posterior["beta_seasonal"].values
            beta_seasonal_samples = beta_seasonal_samples.reshape(-1, beta_seasonal_samples.shape[-1])
        else:
            beta_seasonal_samples = None
            
        # Sample predictions
        n_samples = len(k_samples)
        forecasts = []
        
        # Create seasonality features for future dates
        if beta_seasonal_samples is not None:
            future_seasonality_features = []
            feature_names = []
            
            # Yearly seasonality
            if self.yearly_seasonality > 0:
                yearly_features, yearly_names = self._make_seasonality_features(
                    pd.Series(future_dates), 365.25, self.yearly_seasonality, "yearly"
                )
                future_seasonality_features.append(yearly_features)
                feature_names.extend(yearly_names)
                
            # Weekly seasonality
            if self.weekly_seasonality > 0:
                weekly_features, weekly_names = self._make_seasonality_features(
                    pd.Series(future_dates), 7, self.weekly_seasonality, "weekly"
                )
                future_seasonality_features.append(weekly_features)
                feature_names.extend(weekly_names)
                
            # Daily seasonality (if applicable)
            if self.daily_seasonality and self.daily_seasonality > 0:
                daily_features, daily_names = self._make_seasonality_features(
                    pd.Series(future_dates), 1, self.daily_seasonality, "daily"
                )
                future_seasonality_features.append(daily_features)
                feature_names.extend(daily_names)
                
            if future_seasonality_features:
                seasonality_matrix = np.hstack(future_seasonality_features)
            else:
                seasonality_matrix = None
        else:
            seasonality_matrix = None
            
        # Generate changepoint matrix for future
        changepoint_matrix = self._get_changepoint_matrix(t_future_scaled, self.n_changepoints)
        
        for i in range(n_samples):
            # Trend component
            if delta_samples.ndim == 3:
                # Shape is (chain, draw, changepoints)
                delta_sample = delta_samples.reshape(-1, delta_samples.shape[-1])[i]
            else:
                delta_sample = delta_samples[i]
                
            growth = k_samples[i] + np.dot(changepoint_matrix, delta_sample)
            trend = growth * t_future_scaled + m_samples[i]
            
            # Seasonal component
            if seasonality_matrix is not None and beta_seasonal_samples is not None:
                seasonal_effect = np.dot(seasonality_matrix, beta_seasonal_samples[i])
            else:
                seasonal_effect = 0
                
            # Combined mean
            mu_forecast = trend + seasonal_effect
            
            # Sample from normal distribution
            forecast_sample = np.random.normal(mu_forecast, sigma_samples[i])
            forecasts.append(forecast_sample)
            
        forecasts = np.array(forecasts)
        
        # Calculate summary statistics
        forecast_mean = np.mean(forecasts, axis=0)
        forecast_std = np.std(forecasts, axis=0)
        forecast_lower = np.percentile(forecasts, 2.5, axis=0)
        forecast_upper = np.percentile(forecasts, 97.5, axis=0)
        
        # Create results DataFrame
        results = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_mean,
            "lower_95": forecast_lower,
            "upper_95": forecast_upper,
            "forecast_std": forecast_std,
        })
        
        return results

    def plot_components(self, X: pd.DataFrame, forecast_df: Optional[pd.DataFrame] = None):
        """Plot trend and seasonal components.
        
        Parameters
        ----------
        X : pd.DataFrame
            Historical data
        forecast_df : pd.DataFrame, optional
            Forecast results
        """
        import matplotlib.pyplot as plt
        
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted before plotting components")
            
        # Extract posterior samples
        posterior = self._fit_result.posterior
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot trend
        dates = pd.to_datetime(X[self.date_column])
        t = np.arange(len(dates))
        t_scaled = t / t.max()
        
        # Reconstruct trend
        k_samples = posterior["k"].values.flatten()
        m_samples = posterior["m"].values.flatten()
        delta_samples = posterior["delta"].values
        
        changepoint_matrix = self._get_changepoint_matrix(t_scaled, self.n_changepoints)
        
        trend_samples = []
        for i in range(len(k_samples)):
            growth = k_samples[i] + np.dot(changepoint_matrix, delta_samples[:, :, i].T)
            trend = growth * t_scaled + m_samples[i]
            trend_samples.append(trend)
            
        trend_samples = np.array(trend_samples)
        trend_mean = np.mean(trend_samples, axis=0)
        trend_std = np.std(trend_samples, axis=0)
        
        axes[0].plot(dates, trend_mean, 'b-', label='Trend')
        axes[0].fill_between(
            dates,
            trend_mean - 2 * trend_std,
            trend_mean + 2 * trend_std,
            alpha=0.3
        )
        axes[0].set_title('Trend Component')
        axes[0].set_ylabel('Demand')
        axes[0].legend()
        
        # Plot seasonality (simplified - showing yearly)
        if self.yearly_seasonality > 0:
            # Create one year of dates
            year_dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
            yearly_features, _ = self._make_seasonality_features(
                year_dates, 365.25, self.yearly_seasonality, "yearly"
            )
            
            # Get seasonal coefficients
            beta_seasonal = posterior["beta_seasonal"].values
            yearly_effect = np.dot(yearly_features, beta_seasonal[:, :self.yearly_seasonality*2].T)
            
            axes[1].plot(year_dates, np.mean(yearly_effect, axis=1), 'g-', label='Yearly Seasonality')
            axes[1].fill_between(
                year_dates,
                np.percentile(yearly_effect, 2.5, axis=1),
                np.percentile(yearly_effect, 97.5, axis=1),
                alpha=0.3,
                color='green'
            )
            axes[1].set_title('Yearly Seasonality')
            axes[1].set_ylabel('Seasonal Effect')
            axes[1].legend()
            
        # Plot actual vs fitted
        fitted_mean = posterior["demand"].mean(dim=["chain", "draw"]).values
        
        axes[2].plot(dates, X[self.target_column], 'k.', alpha=0.5, label='Actual')
        axes[2].plot(dates, fitted_mean, 'r-', label='Fitted')
        axes[2].set_title('Actual vs Fitted')
        axes[2].set_ylabel('Demand')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        
        plt.tight_layout()
        return fig, axes