"""Base classes for supply chain optimization models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pydantic import BaseModel, Field, field_validator


class SupplyChainModelBuilder(ABC):
    """Abstract base class for supply chain optimization models.
    
    This class provides the foundation for all supply chain models,
    following the pattern established by PyMC-Marketing.
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the supply chain model builder.
        
        Parameters
        ----------
        model_config : dict, optional
            Configuration for the model
        sampler_config : dict, optional
            Configuration for the sampler
        """
        self.model_config = model_config or {}
        self.sampler_config = sampler_config or {}
        self._model = None
        self._fit_result = None
        self._posterior_predictive = None
        
    @abstractmethod
    def build_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pm.Model:
        """Build the PyMC model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable (if applicable)
            
        Returns
        -------
        pm.Model
            The built PyMC model
        """
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        progressbar: bool = True,
        **kwargs
    ) -> az.InferenceData:
        """Fit the model using MCMC sampling.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable
        progressbar : bool
            Whether to show progress bar
        **kwargs
            Additional arguments passed to pm.sample
            
        Returns
        -------
        az.InferenceData
            The inference data object
        """
        self._model = self.build_model(X, y)
        
        with self._model:
            sampler_kwargs = {
                "progressbar": progressbar,
                **self.sampler_config,
                **kwargs
            }
            self._fit_result = pm.sample(**sampler_kwargs)
            
        return self._fit_result
    
    def predict(
        self,
        X_pred: pd.DataFrame,
        include_last: bool = True,
        kind: str = "mean",
    ) -> np.ndarray:
        """Generate predictions for new data.
        
        Parameters
        ----------
        X_pred : pd.DataFrame
            Data for prediction
        include_last : bool
            Whether to include the last observation
        kind : str
            Type of prediction ('mean', 'median', or samples)
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if self._fit_result is None:
            raise RuntimeError("Model must be fitted before making predictions")
            
        with self._model:
            pm.set_data({"X_pred": X_pred})
            self._posterior_predictive = pm.sample_posterior_predictive(
                self._fit_result,
                progressbar=False,
            )
            
        if kind == "mean":
            return self._posterior_predictive.posterior_predictive.mean(dim=["chain", "draw"])
        elif kind == "median":
            return self._posterior_predictive.posterior_predictive.median(dim=["chain", "draw"])
        else:
            return self._posterior_predictive.posterior_predictive
            
    @property
    def fit_result(self) -> Optional[az.InferenceData]:
        """Return the fitted result."""
        return self._fit_result
    
    @property
    def model(self) -> Optional[pm.Model]:
        """Return the PyMC model."""
        return self._model


class OptimizationResult(BaseModel):
    """Container for optimization results."""
    
    objective_value: Union[float, int] = Field(..., description="Optimal objective function value")
    solution: Union[Dict[str, Any], List[Any], Any] = Field(..., description="Optimal variable values")
    status: str = Field(..., description="Optimization status")
    solver_time: Union[float, int] = Field(..., description="Time taken by solver in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    
class SupplyChainOptimizer(ABC):
    """Abstract base class for deterministic supply chain optimizers."""
    
    @abstractmethod
    def optimize(self, **kwargs) -> OptimizationResult:
        """Run the optimization."""
        pass
    
    @abstractmethod
    def get_constraints(self) -> List[Any]:
        """Get the optimization constraints."""
        pass
    
    @abstractmethod
    def get_objective(self) -> Any:
        """Get the optimization objective function."""
        pass