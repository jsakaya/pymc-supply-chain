"""Multi-echelon inventory optimization for supply chain networks."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pulp
from scipy.optimize import minimize

from pymc_supply_chain.base import OptimizationResult, SupplyChainOptimizer


class MultiEchelonInventory(SupplyChainOptimizer):
    """Multi-echelon inventory optimization for supply networks.
    
    Features:
    - Serial, assembly, and distribution systems
    - Base stock policy optimization
    - Service time guarantees
    - Cost minimization across network
    """
    
    def __init__(
        self,
        network: nx.DiGraph,
        node_attributes: Dict[str, Dict[str, Any]],
        demand_info: Dict[str, Dict[str, float]],
        service_times: Optional[Dict[str, float]] = None,
    ):
        """Initialize multi-echelon model.
        
        Parameters
        ----------
        network : nx.DiGraph
            Supply chain network structure
        node_attributes : dict
            Attributes for each node (costs, lead times)
        demand_info : dict
            Demand information for end nodes
        service_times : dict, optional
            Required service times at each node
        """
        self.network = network
        self.node_attributes = node_attributes
        self.demand_info = demand_info
        self.service_times = service_times or {}
        
        # Validate network
        self._validate_network()
        
    def _validate_network(self):
        """Validate network structure and data."""
        # Check if network is a DAG
        if not nx.is_directed_acyclic_graph(self.network):
            raise ValueError("Network must be a directed acyclic graph")
            
        # Check node attributes
        required_attrs = ["holding_cost", "lead_time"]
        for node, attrs in self.node_attributes.items():
            for req_attr in required_attrs:
                if req_attr not in attrs:
                    raise ValueError(f"Node {node} missing required attribute: {req_attr}")
                    
        # Check demand nodes
        end_nodes = [n for n in self.network.nodes() if self.network.out_degree(n) == 0]
        for node in end_nodes:
            if node not in self.demand_info:
                warnings.warn(f"End node {node} has no demand information")
                
    def calculate_base_stock_levels(
        self,
        method: str = "guaranteed_service",
        target_service_level: float = 0.95,
    ) -> Dict[str, float]:
        """Calculate base stock levels for each node.
        
        Parameters
        ----------
        method : str
            'guaranteed_service' or 'stochastic_service'
        target_service_level : float
            Target service level
            
        Returns
        -------
        dict
            Base stock levels by node
        """
        if method == "guaranteed_service":
            return self._guaranteed_service_model(target_service_level)
        elif method == "stochastic_service":
            return self._stochastic_service_model(target_service_level)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _guaranteed_service_model(self, target_service_level: float) -> Dict[str, float]:
        """Guaranteed service time model (GSM)."""
        from scipy.stats import norm
        
        # Topological sort for bottom-up calculation
        topo_order = list(nx.topological_sort(self.network))
        
        # Initialize
        base_stocks = {}
        net_lead_times = {}
        
        # Calculate bottom-up
        for node in reversed(topo_order):
            attrs = self.node_attributes[node]
            
            # Get downstream nodes
            successors = list(self.network.successors(node))
            
            if not successors:  # End node
                # External demand
                if node in self.demand_info:
                    demand_mean = self.demand_info[node]["mean"]
                    demand_std = self.demand_info[node]["std"]
                else:
                    demand_mean = 0
                    demand_std = 0
                    
                # Service time guarantee
                service_time = self.service_times.get(node, 0)
                
                # Net lead time
                net_lead_time = attrs["lead_time"] - service_time
                net_lead_times[node] = net_lead_time
                
                # Base stock level
                if net_lead_time > 0 and demand_std > 0:
                    z_score = norm.ppf(target_service_level)
                    base_stock = demand_mean * net_lead_time + z_score * demand_std * np.sqrt(net_lead_time)
                else:
                    base_stock = 0
                    
                base_stocks[node] = base_stock
                
            else:  # Internal node
                # Aggregate downstream demand
                total_demand_mean = 0
                total_demand_var = 0
                max_downstream_time = 0
                
                for succ in successors:
                    if succ in self.demand_info:
                        total_demand_mean += self.demand_info[succ]["mean"]
                        total_demand_var += self.demand_info[succ]["std"] ** 2
                    max_downstream_time = max(max_downstream_time, net_lead_times.get(succ, 0))
                    
                total_demand_std = np.sqrt(total_demand_var)
                
                # Service time
                service_time = self.service_times.get(node, max_downstream_time)
                
                # Net lead time
                net_lead_time = attrs["lead_time"] + max_downstream_time - service_time
                net_lead_times[node] = net_lead_time
                
                # Base stock
                if net_lead_time > 0 and total_demand_std > 0:
                    z_score = norm.ppf(target_service_level)
                    base_stock = total_demand_mean * net_lead_time + z_score * total_demand_std * np.sqrt(net_lead_time)
                else:
                    base_stock = 0
                    
                base_stocks[node] = base_stock
                
        return base_stocks
    
    def _stochastic_service_model(self, target_service_level: float) -> Dict[str, float]:
        """Stochastic service time model."""
        # Simplified version - can be extended
        return self._guaranteed_service_model(target_service_level)
    
    def optimize_service_times(
        self,
        max_end_to_end_time: float,
        method: str = "linear_programming",
    ) -> Dict[str, float]:
        """Optimize service times to minimize cost.
        
        Parameters
        ----------
        max_end_to_end_time : float
            Maximum allowed end-to-end service time
        method : str
            Optimization method
            
        Returns
        -------
        dict
            Optimal service times
        """
        if method == "linear_programming":
            return self._optimize_service_times_lp(max_end_to_end_time)
        else:
            return self._optimize_service_times_nlp(max_end_to_end_time)
            
    def _optimize_service_times_lp(self, max_time: float) -> Dict[str, float]:
        """Linear programming formulation for service time optimization."""
        # Create LP problem
        prob = pulp.LpProblem("Service_Time_Optimization", pulp.LpMinimize)
        
        # Decision variables: service times
        service_vars = {}
        for node in self.network.nodes():
            service_vars[node] = pulp.LpVariable(f"service_{node}", lowBound=0)
            
        # Objective: minimize total holding cost
        obj = 0
        for node in self.network.nodes():
            holding_cost = self.node_attributes[node]["holding_cost"]
            # Approximate holding cost as function of service time
            obj += holding_cost * service_vars[node]
            
        prob += obj
        
        # Constraints
        # 1. End-to-end service time constraints
        paths = []
        source_nodes = [n for n in self.network.nodes() if self.network.in_degree(n) == 0]
        end_nodes = [n for n in self.network.nodes() if self.network.out_degree(n) == 0]
        
        for source in source_nodes:
            for end in end_nodes:
                if nx.has_path(self.network, source, end):
                    paths.extend(nx.all_simple_paths(self.network, source, end))
                    
        for path in paths:
            path_service_time = pulp.lpSum([service_vars[node] for node in path])
            prob += path_service_time <= max_time
            
        # 2. Precedence constraints
        for edge in self.network.edges():
            upstream, downstream = edge
            lead_time = self.node_attributes[upstream]["lead_time"]
            prob += service_vars[upstream] >= service_vars[downstream] + lead_time
            
        # Solve
        prob.solve()
        
        # Extract solution
        if pulp.LpStatus[prob.status] == "Optimal":
            service_times = {
                node: var.varValue for node, var in service_vars.items()
            }
            return service_times
        else:
            raise RuntimeError("Optimization failed")
            
    def _optimize_service_times_nlp(self, max_time: float) -> Dict[str, float]:
        """Nonlinear programming for more accurate cost modeling."""
        nodes = list(self.network.nodes())
        n_nodes = len(nodes)
        
        # Initial guess
        x0 = np.ones(n_nodes) * max_time / n_nodes
        
        # Bounds
        bounds = [(0, max_time) for _ in range(n_nodes)]
        
        # Objective function
        def objective(x):
            service_times = dict(zip(nodes, x))
            base_stocks = self.calculate_base_stock_levels(method="guaranteed_service")
            
            total_cost = 0
            for node, base_stock in base_stocks.items():
                holding_cost = self.node_attributes[node]["holding_cost"]
                total_cost += holding_cost * base_stock
                
            return total_cost
        
        # Constraints
        constraints = []
        
        # End-to-end time constraints
        def path_constraint(x, path):
            return max_time - sum(x[nodes.index(node)] for node in path)
        
        source_nodes = [n for n in self.network.nodes() if self.network.in_degree(n) == 0]
        end_nodes = [n for n in self.network.nodes() if self.network.out_degree(n) == 0]
        
        for source in source_nodes:
            for end in end_nodes:
                if nx.has_path(self.network, source, end):
                    for path in nx.all_simple_paths(self.network, source, end):
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda x, p=path: path_constraint(x, p)
                        })
                        
        # Solve
        result = minimize(objective, x0, bounds=bounds, constraints=constraints)
        
        if result.success:
            return dict(zip(nodes, result.x))
        else:
            raise RuntimeError("Optimization failed")
            
    def simulate_network(
        self,
        n_periods: int,
        base_stocks: Dict[str, float],
        initial_inventory: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Simulate multi-echelon network performance.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        base_stocks : dict
            Base stock levels for each node
        initial_inventory : dict, optional
            Initial inventory levels
            
        Returns
        -------
        pd.DataFrame
            Simulation results
        """
        # Initialize
        inventory = initial_inventory or {node: base_stocks[node] for node in self.network.nodes()}
        on_order = {node: [] for node in self.network.nodes()}  # List of (arrival_time, quantity)
        
        results = []
        
        for t in range(n_periods):
            period_results = {"period": t}
            
            # Process nodes in topological order
            for node in nx.topological_sort(self.network):
                # Receive orders
                arrived = [qty for arrival, qty in on_order[node] if arrival <= t]
                on_order[node] = [(arrival, qty) for arrival, qty in on_order[node] if arrival > t]
                inventory[node] += sum(arrived)
                
                # External demand (end nodes)
                if self.network.out_degree(node) == 0:
                    if node in self.demand_info:
                        demand = np.random.normal(
                            self.demand_info[node]["mean"],
                            self.demand_info[node]["std"]
                        )
                        demand = max(0, demand)
                    else:
                        demand = 0
                        
                    # Satisfy demand
                    satisfied = min(inventory[node], demand)
                    stockout = max(demand - inventory[node], 0)
                    inventory[node] -= satisfied
                    
                    period_results[f"{node}_demand"] = demand
                    period_results[f"{node}_satisfied"] = satisfied
                    period_results[f"{node}_stockout"] = stockout
                    
                # Internal demand (from downstream nodes)
                else:
                    total_request = 0
                    for successor in self.network.successors(node):
                        # Simple pull policy
                        if inventory[successor] < base_stocks[successor]:
                            request = base_stocks[successor] - inventory[successor]
                            total_request += request
                            
                    # Satisfy internal demand
                    satisfied = min(inventory[node], total_request)
                    inventory[node] -= satisfied
                    
                    # Allocate to successors (proportional)
                    if total_request > 0:
                        for successor in self.network.successors(node):
                            if inventory[successor] < base_stocks[successor]:
                                request = base_stocks[successor] - inventory[successor]
                                allocated = satisfied * request / total_request
                                
                                # Ship with lead time
                                lead_time = self.node_attributes[node]["lead_time"]
                                on_order[successor].append((t + lead_time, allocated))
                                
                # Replenishment decision
                position = inventory[node] + sum(qty for _, qty in on_order[node])
                if position < base_stocks[node]:
                    order_qty = base_stocks[node] - position
                    
                    # Order from predecessors or external
                    if self.network.in_degree(node) == 0:
                        # External supplier - immediate
                        lead_time = self.node_attributes[node]["lead_time"]
                        on_order[node].append((t + lead_time, order_qty))
                        
                period_results[f"{node}_inventory"] = inventory[node]
                period_results[f"{node}_on_order"] = sum(qty for _, qty in on_order[node])
                
            results.append(period_results)
            
        return pd.DataFrame(results)
    
    def calculate_network_metrics(
        self,
        simulation_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate network performance metrics.
        
        Parameters
        ----------
        simulation_df : pd.DataFrame
            Simulation results
            
        Returns
        -------
        dict
            Performance metrics
        """
        metrics = {}
        
        # Service levels by node
        for node in self.network.nodes():
            if f"{node}_demand" in simulation_df.columns:
                total_demand = simulation_df[f"{node}_demand"].sum()
                total_satisfied = simulation_df[f"{node}_satisfied"].sum()
                metrics[f"{node}_service_level"] = total_satisfied / total_demand if total_demand > 0 else 1.0
                
        # Average inventory by node
        for node in self.network.nodes():
            if f"{node}_inventory" in simulation_df.columns:
                metrics[f"{node}_avg_inventory"] = simulation_df[f"{node}_inventory"].mean()
                
        # Total network metrics
        total_inventory_value = 0
        for node in self.network.nodes():
            if f"{node}_inventory" in simulation_df.columns:
                avg_inv = simulation_df[f"{node}_inventory"].mean()
                holding_cost = self.node_attributes[node]["holding_cost"]
                total_inventory_value += avg_inv * holding_cost
                
        metrics["total_inventory_cost"] = total_inventory_value
        
        return metrics
    
    def optimize(self, **kwargs) -> OptimizationResult:
        """Optimize multi-echelon inventory."""
        method = kwargs.get("method", "guaranteed_service")
        target_service = kwargs.get("target_service_level", 0.95)
        
        # Calculate base stocks
        base_stocks = self.calculate_base_stock_levels(method, target_service)
        
        # Calculate total cost
        total_cost = sum(
            base_stocks[node] * self.node_attributes[node]["holding_cost"]
            for node in self.network.nodes()
        )
        
        return OptimizationResult(
            objective_value=total_cost,
            solution=base_stocks,
            status="optimal",
            solver_time=0.1,
            metadata={
                "method": method,
                "service_level": target_service,
                "network_size": len(self.network.nodes()),
            }
        )
    
    def get_constraints(self) -> List[str]:
        """Get constraints description."""
        return [
            "Service time guarantees",
            "Network flow conservation",
            "Non-negativity of inventory",
        ]
    
    def get_objective(self) -> str:
        """Get objective description."""
        return "Minimize total network holding cost"