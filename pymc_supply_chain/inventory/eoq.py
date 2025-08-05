"""Economic Order Quantity (EOQ) models with uncertainty."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.optimize import minimize_scalar

from pymc_supply_chain.base import OptimizationResult, SupplyChainOptimizer


class EOQModel(SupplyChainOptimizer):
    """Classic Economic Order Quantity model with extensions.
    
    Features:
    - Basic EOQ calculation
    - Quantity discounts
    - Backordering allowed
    - Multi-item EOQ
    """
    
    def __init__(
        self,
        holding_cost_rate: float = 0.2,
        fixed_order_cost: float = 100,
        unit_cost: Optional[float] = None,
        backorder_cost: Optional[float] = None,
        quantity_discounts: Optional[Dict[float, float]] = None,
    ):
        """Initialize EOQ model.
        
        Parameters
        ----------
        holding_cost_rate : float
            Annual holding cost as fraction of unit cost
        fixed_order_cost : float
            Fixed cost per order
        unit_cost : float, optional
            Unit purchase cost
        backorder_cost : float, optional
            Cost per unit backordered per time
        quantity_discounts : dict, optional
            Quantity thresholds and discount rates
        """
        self.holding_cost_rate = holding_cost_rate
        self.fixed_order_cost = fixed_order_cost
        self.unit_cost = unit_cost
        self.backorder_cost = backorder_cost
        self.quantity_discounts = quantity_discounts or {}
        
    def calculate_eoq(
        self,
        annual_demand: float,
        unit_cost: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate basic EOQ.
        
        Parameters
        ----------
        annual_demand : float
            Annual demand quantity
        unit_cost : float, optional
            Override unit cost
            
        Returns
        -------
        dict
            EOQ results
        """
        unit_cost = unit_cost or self.unit_cost
        if unit_cost is None:
            raise ValueError("Unit cost must be provided")
            
        holding_cost = self.holding_cost_rate * unit_cost
        
        # Basic EOQ formula
        eoq = np.sqrt(2 * annual_demand * self.fixed_order_cost / holding_cost)
        
        # Calculate associated metrics
        number_of_orders = annual_demand / eoq
        time_between_orders = 1 / number_of_orders  # in years
        
        # Total costs
        ordering_cost = number_of_orders * self.fixed_order_cost
        holding_cost_total = (eoq / 2) * holding_cost
        total_cost = ordering_cost + holding_cost_total
        
        return {
            "eoq": eoq,
            "number_of_orders": number_of_orders,
            "time_between_orders_days": time_between_orders * 365,
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost_total,
            "total_cost": total_cost,
        }
    
    def calculate_eoq_with_discounts(
        self,
        annual_demand: float,
        base_unit_cost: float,
    ) -> Dict[str, Any]:
        """Calculate EOQ with quantity discounts.
        
        Parameters
        ----------
        annual_demand : float
            Annual demand
        base_unit_cost : float
            Base unit cost before discounts
            
        Returns
        -------
        dict
            Optimal order quantity and costs
        """
        results = []
        
        # Sort discount thresholds
        thresholds = sorted(self.quantity_discounts.keys())
        
        # Add base case (no discount)
        thresholds = [0] + thresholds
        discounts = [0] + [self.quantity_discounts[t] for t in thresholds[1:]]
        
        for i, (threshold, discount) in enumerate(zip(thresholds, discounts)):
            # Calculate discounted unit cost
            unit_cost = base_unit_cost * (1 - discount)
            
            # Calculate EOQ for this price
            eoq_result = self.calculate_eoq(annual_demand, unit_cost)
            eoq = eoq_result["eoq"]
            
            # Check if EOQ is feasible for this discount tier
            if i < len(thresholds) - 1:
                # Not the last tier
                if eoq < threshold:
                    # EOQ too small, use threshold
                    order_quantity = threshold
                elif i < len(thresholds) - 1 and eoq >= thresholds[i + 1]:
                    # EOQ too large for this tier, skip
                    continue
                else:
                    order_quantity = eoq
            else:
                # Last tier
                if eoq < threshold:
                    order_quantity = threshold
                else:
                    order_quantity = eoq
                    
            # Calculate total cost including purchase cost
            holding_cost = self.holding_cost_rate * unit_cost * order_quantity / 2
            ordering_cost = annual_demand / order_quantity * self.fixed_order_cost
            purchase_cost = annual_demand * unit_cost
            total_cost = holding_cost + ordering_cost + purchase_cost
            
            results.append({
                "quantity": order_quantity,
                "unit_cost": unit_cost,
                "discount": discount,
                "total_cost": total_cost,
                "holding_cost": holding_cost,
                "ordering_cost": ordering_cost,
                "purchase_cost": purchase_cost,
            })
            
        # Find optimal
        best_result = min(results, key=lambda x: x["total_cost"])
        
        return {
            "optimal": best_result,
            "all_options": results,
        }
    
    def calculate_eoq_with_backorders(
        self,
        annual_demand: float,
        unit_cost: float,
        backorder_cost: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate EOQ when backorders are allowed.
        
        Parameters
        ----------
        annual_demand : float
            Annual demand
        unit_cost : float
            Unit cost
        backorder_cost : float, optional
            Cost per unit backordered
            
        Returns
        -------
        dict
            EOQ with backorder results
        """
        backorder_cost = backorder_cost or self.backorder_cost
        if backorder_cost is None:
            raise ValueError("Backorder cost must be provided")
            
        holding_cost = self.holding_cost_rate * unit_cost
        
        # EOQ with backorders formula
        eoq = np.sqrt(
            2 * annual_demand * self.fixed_order_cost * 
            (holding_cost + backorder_cost) / (holding_cost * backorder_cost)
        )
        
        # Maximum backorder level
        max_backorder = eoq * holding_cost / (holding_cost + backorder_cost)
        
        # Maximum inventory level
        max_inventory = eoq - max_backorder
        
        # Costs
        number_of_orders = annual_demand / eoq
        ordering_cost = number_of_orders * self.fixed_order_cost
        holding_cost_total = (max_inventory ** 2) / (2 * eoq) * holding_cost
        backorder_cost_total = (max_backorder ** 2) / (2 * eoq) * backorder_cost
        total_cost = ordering_cost + holding_cost_total + backorder_cost_total
        
        return {
            "eoq": eoq,
            "max_inventory": max_inventory,
            "max_backorder": max_backorder,
            "number_of_orders": number_of_orders,
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost_total,
            "backorder_cost": backorder_cost_total,
            "total_cost": total_cost,
        }
    
    def optimize(self, **kwargs) -> OptimizationResult:
        """Run EOQ optimization."""
        annual_demand = kwargs.get("annual_demand")
        unit_cost = kwargs.get("unit_cost", self.unit_cost)
        
        if self.quantity_discounts:
            result = self.calculate_eoq_with_discounts(annual_demand, unit_cost)
            optimal = result["optimal"]
        else:
            result = self.calculate_eoq(annual_demand, unit_cost)
            optimal = result
            
        return OptimizationResult(
            objective_value=optimal.get("total_cost", 0),
            solution={"order_quantity": optimal.get("eoq", optimal.get("quantity"))},
            status="optimal",
            solver_time=0.001,
            metadata=optimal,
        )
    
    def get_constraints(self):
        """EOQ has no explicit constraints."""
        return []
    
    def get_objective(self):
        """Return objective function description."""
        return "Minimize total inventory cost (ordering + holding)"


class StochasticEOQ(SupplyChainOptimizer):
    """Stochastic EOQ model with demand uncertainty.
    
    Features:
    - Demand uncertainty modeling
    - Service level constraints
    - Reorder point calculation
    - Safety stock optimization
    """
    
    def __init__(
        self,
        holding_cost_rate: float = 0.2,
        fixed_order_cost: float = 100,
        unit_cost: float = 10,
        stockout_cost: float = 50,
        lead_time_days: float = 7,
        service_level: float = 0.95,
    ):
        """Initialize stochastic EOQ model.
        
        Parameters
        ----------
        holding_cost_rate : float
            Annual holding cost rate
        fixed_order_cost : float
            Fixed ordering cost
        unit_cost : float
            Unit purchase cost
        stockout_cost : float
            Cost per unit of stockout
        lead_time_days : float
            Lead time in days
        service_level : float
            Target service level
        """
        self.holding_cost_rate = holding_cost_rate
        self.fixed_order_cost = fixed_order_cost
        self.unit_cost = unit_cost
        self.stockout_cost = stockout_cost
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        
    def fit_demand_distribution(self, demand_data: pd.Series) -> Dict[str, float]:
        """Fit demand distribution from historical data.
        
        Parameters
        ----------
        demand_data : pd.Series
            Historical daily demand
            
        Returns
        -------
        dict
            Distribution parameters
        """
        # Calculate daily statistics
        daily_mean = demand_data.mean()
        daily_std = demand_data.std()
        
        # Annual projections
        annual_mean = daily_mean * 365
        annual_std = daily_std * np.sqrt(365)
        
        # Lead time demand statistics
        lead_time_mean = daily_mean * self.lead_time_days
        lead_time_std = daily_std * np.sqrt(self.lead_time_days)
        
        return {
            "daily_mean": daily_mean,
            "daily_std": daily_std,
            "annual_mean": annual_mean,
            "annual_std": annual_std,
            "lead_time_mean": lead_time_mean,
            "lead_time_std": lead_time_std,
        }
    
    def calculate_stochastic_eoq(
        self,
        demand_params: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate EOQ with stochastic demand.
        
        Parameters
        ----------
        demand_params : dict
            Demand distribution parameters
            
        Returns
        -------
        dict
            Stochastic EOQ results
        """
        from scipy.stats import norm
        
        annual_demand = demand_params["annual_mean"]
        lead_time_demand_mean = demand_params["lead_time_mean"]
        lead_time_demand_std = demand_params["lead_time_std"]
        
        # Basic EOQ (using mean demand)
        holding_cost = self.holding_cost_rate * self.unit_cost
        eoq = np.sqrt(2 * annual_demand * self.fixed_order_cost / holding_cost)
        
        # Safety stock calculation
        z_score = norm.ppf(self.service_level)
        safety_stock = z_score * lead_time_demand_std
        
        # Reorder point
        reorder_point = lead_time_demand_mean + safety_stock
        
        # Expected costs
        number_of_orders = annual_demand / eoq
        ordering_cost = number_of_orders * self.fixed_order_cost
        
        # Average inventory = EOQ/2 + safety stock
        avg_inventory = eoq / 2 + safety_stock
        holding_cost_total = avg_inventory * holding_cost
        
        # Expected stockout cost (approximate)
        expected_stockout = lead_time_demand_std * norm.pdf(z_score) - \
                           lead_time_demand_std * z_score * (1 - norm.cdf(z_score))
        stockout_cost_total = expected_stockout * number_of_orders * self.stockout_cost
        
        total_cost = ordering_cost + holding_cost_total + stockout_cost_total
        
        return {
            "eoq": eoq,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "avg_inventory": avg_inventory,
            "number_of_orders": number_of_orders,
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost_total,
            "expected_stockout_cost": stockout_cost_total,
            "total_cost": total_cost,
            "service_level": self.service_level,
            "fill_rate": 1 - expected_stockout / eoq,
        }
    
    def optimize_service_level(
        self,
        demand_params: Dict[str, float],
        service_levels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Find optimal service level balancing costs.
        
        Parameters
        ----------
        demand_params : dict
            Demand parameters
        service_levels : array, optional
            Service levels to evaluate
            
        Returns
        -------
        pd.DataFrame
            Results for different service levels
        """
        if service_levels is None:
            service_levels = np.linspace(0.8, 0.99, 20)
            
        results = []
        
        for sl in service_levels:
            self.service_level = sl
            result = self.calculate_stochastic_eoq(demand_params)
            result["service_level"] = sl
            results.append(result)
            
        df = pd.DataFrame(results)
        
        # Find optimal
        optimal_idx = df["total_cost"].idxmin()
        df["is_optimal"] = False
        df.loc[optimal_idx, "is_optimal"] = True
        
        return df
    
    def simulate_inventory_policy(
        self,
        demand_data: pd.Series,
        n_periods: int = 365,
        eoq: Optional[float] = None,
        reorder_point: Optional[float] = None,
    ) -> pd.DataFrame:
        """Simulate inventory policy performance.
        
        Parameters
        ----------
        demand_data : pd.Series
            Historical demand for sampling
        n_periods : int
            Number of periods to simulate
        eoq : float, optional
            Order quantity (calculated if None)
        reorder_point : float, optional
            Reorder point (calculated if None)
            
        Returns
        -------
        pd.DataFrame
            Simulation results
        """
        # Fit demand if needed
        demand_params = self.fit_demand_distribution(demand_data)
        
        if eoq is None or reorder_point is None:
            policy = self.calculate_stochastic_eoq(demand_params)
            eoq = eoq or policy["eoq"]
            reorder_point = reorder_point or policy["reorder_point"]
            
        # Initialize simulation
        inventory = eoq  # Start with full order
        position = inventory  # Inventory position (on-hand + on-order)
        on_order = 0
        orders_in_transit = []  # (arrival_time, quantity)
        
        results = []
        
        for t in range(n_periods):
            # Generate demand
            demand = np.random.choice(demand_data.values)
            
            # Receive orders
            arrived = [q for arrival, q in orders_in_transit if arrival <= t]
            orders_in_transit = [(arrival, q) for arrival, q in orders_in_transit if arrival > t]
            inventory += sum(arrived)
            on_order -= sum(arrived)
            
            # Satisfy demand
            satisfied = min(inventory, demand)
            stockout = max(demand - inventory, 0)
            inventory = max(inventory - demand, 0)
            
            # Update position
            position = inventory + on_order
            
            # Check reorder point
            if position <= reorder_point and on_order == 0:
                # Place order
                on_order = eoq
                arrival_time = t + self.lead_time_days
                orders_in_transit.append((arrival_time, eoq))
                order_placed = True
            else:
                order_placed = False
                
            results.append({
                "period": t,
                "demand": demand,
                "inventory": inventory,
                "position": position,
                "on_order": on_order,
                "satisfied": satisfied,
                "stockout": stockout,
                "order_placed": order_placed,
            })
            
        return pd.DataFrame(results)
    
    def optimize(self, **kwargs) -> OptimizationResult:
        """Run stochastic EOQ optimization."""
        demand_data = kwargs.get("demand_data")
        if demand_data is None:
            raise ValueError("demand_data required for stochastic EOQ")
            
        demand_params = self.fit_demand_distribution(demand_data)
        result = self.calculate_stochastic_eoq(demand_params)
        
        return OptimizationResult(
            objective_value=result["total_cost"],
            solution={
                "eoq": result["eoq"],
                "safety_stock": result["safety_stock"],
                "reorder_point": result["reorder_point"],
            },
            status="optimal",
            solver_time=0.01,
            metadata=result,
        )
    
    def get_constraints(self):
        """Return constraints."""
        return [f"Service level >= {self.service_level}"]
    
    def get_objective(self):
        """Return objective."""
        return "Minimize total cost (ordering + holding + stockout)"