"""Facility location optimization for supply chain networks."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp
from geopy.distance import geodesic
from scipy.spatial.distance import cdist

from pymc_supply_chain.base import OptimizationResult, SupplyChainOptimizer


class FacilityLocationOptimizer(SupplyChainOptimizer):
    """Optimize facility locations in supply chain networks.
    
    Solves problems like:
    - Warehouse location selection
    - Distribution center placement
    - Manufacturing site selection
    - Multi-echelon facility networks
    """
    
    def __init__(
        self,
        demand_locations: pd.DataFrame,
        candidate_locations: pd.DataFrame,
        fixed_costs: Dict[str, float],
        transportation_cost_per_unit_distance: float = 1.0,
        capacity_constraints: Optional[Dict[str, float]] = None,
        single_sourcing: bool = False,
        existing_facilities: Optional[List[str]] = None,
    ):
        """Initialize facility location optimizer.
        
        Parameters
        ----------
        demand_locations : pd.DataFrame
            DataFrame with columns: location_id, latitude, longitude, demand
        candidate_locations : pd.DataFrame
            DataFrame with columns: location_id, latitude, longitude
        fixed_costs : dict
            Fixed cost for opening each candidate facility
        transportation_cost_per_unit_distance : float
            Cost per unit distance per unit demand
        capacity_constraints : dict, optional
            Capacity limit for each facility
        single_sourcing : bool
            Whether each demand point must be served by single facility
        existing_facilities : list, optional
            Already open facilities that must remain open
        """
        self.demand_locations = demand_locations
        self.candidate_locations = candidate_locations
        self.fixed_costs = fixed_costs
        self.transport_cost = transportation_cost_per_unit_distance
        self.capacity_constraints = capacity_constraints or {}
        self.single_sourcing = single_sourcing
        self.existing_facilities = existing_facilities or []
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distances()
        
    def _calculate_distances(self) -> np.ndarray:
        """Calculate distance matrix between candidates and demand points."""
        n_candidates = len(self.candidate_locations)
        n_demand = len(self.demand_locations)
        distances = np.zeros((n_candidates, n_demand))
        
        for i, (_, candidate) in enumerate(self.candidate_locations.iterrows()):
            for j, (_, demand) in enumerate(self.demand_locations.iterrows()):
                # Calculate geodesic distance
                coord1 = (candidate['latitude'], candidate['longitude'])
                coord2 = (demand['latitude'], demand['longitude'])
                distances[i, j] = geodesic(coord1, coord2).kilometers
                
        return distances
    
    def optimize(self, **kwargs) -> OptimizationResult:
        """Solve facility location problem.
        
        Parameters
        ----------
        **kwargs
            max_facilities: Maximum number of facilities to open
            min_facilities: Minimum number of facilities to open
            budget: Total budget constraint
            service_distance: Maximum service distance allowed
            
        Returns
        -------
        OptimizationResult
            Solution with selected facilities and assignments
        """
        # Extract parameters
        max_facilities = kwargs.get('max_facilities', len(self.candidate_locations))
        min_facilities = kwargs.get('min_facilities', 1)
        budget = kwargs.get('budget', np.inf)
        service_distance = kwargs.get('service_distance', np.inf)
        
        # Create problem
        prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
        
        # Decision variables
        # y[i] = 1 if facility i is opened
        facility_vars = {}
        for i, (idx, _) in enumerate(self.candidate_locations.iterrows()):
            loc_id = self.candidate_locations.loc[idx, 'location_id']
            if loc_id in self.existing_facilities:
                facility_vars[i] = 1  # Fixed to open
            else:
                facility_vars[i] = pulp.LpVariable(f"y_{i}", cat='Binary')
                
        # x[i,j] = fraction of demand j served by facility i
        assign_vars = {}
        for i in range(len(self.candidate_locations)):
            for j in range(len(self.demand_locations)):
                if self.single_sourcing:
                    assign_vars[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
                else:
                    assign_vars[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1)
                    
        # Objective: minimize total cost
        # Fixed costs
        fixed_cost_expr = pulp.lpSum([
            self.fixed_costs.get(
                self.candidate_locations.iloc[i]['location_id'], 
                0
            ) * facility_vars[i]
            for i in range(len(self.candidate_locations))
            if isinstance(facility_vars[i], pulp.LpVariable)
        ])
        
        # Transportation costs
        transport_cost_expr = pulp.lpSum([
            self.distance_matrix[i, j] * 
            self.transport_cost * 
            self.demand_locations.iloc[j]['demand'] * 
            assign_vars[i, j]
            for i in range(len(self.candidate_locations))
            for j in range(len(self.demand_locations))
        ])
        
        prob += fixed_cost_expr + transport_cost_expr
        
        # Constraints
        # 1. Demand satisfaction
        for j in range(len(self.demand_locations)):
            prob += pulp.lpSum([
                assign_vars[i, j] 
                for i in range(len(self.candidate_locations))
            ]) == 1
            
        # 2. Facility capacity
        for i in range(len(self.candidate_locations)):
            loc_id = self.candidate_locations.iloc[i]['location_id']
            if loc_id in self.capacity_constraints:
                prob += pulp.lpSum([
                    assign_vars[i, j] * self.demand_locations.iloc[j]['demand']
                    for j in range(len(self.demand_locations))
                ]) <= self.capacity_constraints[loc_id] * facility_vars[i]
            else:
                # If no capacity limit, still need facility to be open
                for j in range(len(self.demand_locations)):
                    prob += assign_vars[i, j] <= facility_vars[i]
                    
        # 3. Number of facilities
        prob += pulp.lpSum([
            facility_vars[i] 
            for i in range(len(self.candidate_locations))
            if isinstance(facility_vars[i], pulp.LpVariable)
        ]) <= max_facilities
        
        prob += pulp.lpSum([
            facility_vars[i] 
            for i in range(len(self.candidate_locations))
            if isinstance(facility_vars[i], pulp.LpVariable)
        ]) >= min_facilities
        
        # 4. Budget constraint
        if budget < np.inf:
            prob += fixed_cost_expr <= budget
            
        # 5. Service distance constraint
        if service_distance < np.inf:
            for j in range(len(self.demand_locations)):
                for i in range(len(self.candidate_locations)):
                    if self.distance_matrix[i, j] > service_distance:
                        prob += assign_vars[i, j] == 0
                        
        # Solve
        prob.solve()
        
        # Extract solution
        if pulp.LpStatus[prob.status] == "Optimal":
            # Selected facilities
            selected_facilities = []
            for i in range(len(self.candidate_locations)):
                if isinstance(facility_vars[i], pulp.LpVariable):
                    if facility_vars[i].varValue > 0.5:
                        selected_facilities.append(
                            self.candidate_locations.iloc[i]['location_id']
                        )
                elif facility_vars[i] == 1:
                    selected_facilities.append(
                        self.candidate_locations.iloc[i]['location_id']
                    )
                    
            # Assignments
            assignments = {}
            for j in range(len(self.demand_locations)):
                demand_id = self.demand_locations.iloc[j]['location_id']
                assignments[demand_id] = {}
                
                for i in range(len(self.candidate_locations)):
                    if assign_vars[i, j].varValue > 0.001:
                        facility_id = self.candidate_locations.iloc[i]['location_id']
                        assignments[demand_id][facility_id] = assign_vars[i, j].varValue
                        
            # Calculate costs
            total_fixed = sum(
                self.fixed_costs.get(fac, 0) 
                for fac in selected_facilities
            )
            total_transport = pulp.value(transport_cost_expr)
            
            return OptimizationResult(
                objective_value=pulp.value(prob.objective),
                solution={
                    "selected_facilities": selected_facilities,
                    "assignments": assignments,
                },
                status="optimal",
                solver_time=prob.solutionTime,
                metadata={
                    "fixed_cost": total_fixed,
                    "transport_cost": total_transport,
                    "n_facilities": len(selected_facilities),
                }
            )
        else:
            return OptimizationResult(
                objective_value=np.inf,
                solution={},
                status=pulp.LpStatus[prob.status],
                solver_time=prob.solutionTime,
                metadata={"error": "No feasible solution found"}
            )
            
    def analyze_solution(self, result: OptimizationResult) -> pd.DataFrame:
        """Analyze facility location solution.
        
        Parameters
        ----------
        result : OptimizationResult
            Solution from optimize()
            
        Returns
        -------
        pd.DataFrame
            Analysis of solution
        """
        if result.status != "optimal":
            return pd.DataFrame()
            
        selected = result.solution["selected_facilities"]
        assignments = result.solution["assignments"]
        
        # Facility utilization
        facility_stats = []
        
        for facility_id in selected:
            # Find facility index
            fac_idx = self.candidate_locations[
                self.candidate_locations['location_id'] == facility_id
            ].index[0]
            fac_pos = self.candidate_locations.index.get_loc(fac_idx)
            
            # Calculate assigned demand
            total_demand = 0
            n_customers = 0
            avg_distance = 0
            
            for demand_id, assign_dict in assignments.items():
                if facility_id in assign_dict:
                    # Find demand index
                    dem_idx = self.demand_locations[
                        self.demand_locations['location_id'] == demand_id
                    ].index[0]
                    dem_pos = self.demand_locations.index.get_loc(dem_idx)
                    
                    fraction = assign_dict[facility_id]
                    demand = self.demand_locations.iloc[dem_pos]['demand']
                    
                    total_demand += fraction * demand
                    n_customers += fraction
                    avg_distance += fraction * self.distance_matrix[fac_pos, dem_pos]
                    
            # Capacity utilization
            capacity = self.capacity_constraints.get(facility_id, np.inf)
            utilization = total_demand / capacity if capacity < np.inf else 0
            
            facility_stats.append({
                "facility_id": facility_id,
                "latitude": self.candidate_locations.iloc[fac_pos]['latitude'],
                "longitude": self.candidate_locations.iloc[fac_pos]['longitude'],
                "fixed_cost": self.fixed_costs.get(facility_id, 0),
                "total_demand": total_demand,
                "n_customers": n_customers,
                "capacity": capacity,
                "utilization": utilization,
                "avg_distance": avg_distance / n_customers if n_customers > 0 else 0,
            })
            
        return pd.DataFrame(facility_stats)
    
    def sensitivity_analysis(
        self,
        param_name: str,
        param_values: List[float],
        **base_kwargs
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on parameters.
        
        Parameters
        ----------
        param_name : str
            Parameter to vary
        param_values : list
            Values to test
        **base_kwargs
            Base parameters for optimization
            
        Returns
        -------
        pd.DataFrame
            Sensitivity results
        """
        results = []
        
        for value in param_values:
            # Update parameter
            kwargs = base_kwargs.copy()
            kwargs[param_name] = value
            
            # Solve
            result = self.optimize(**kwargs)
            
            if result.status == "optimal":
                results.append({
                    param_name: value,
                    "total_cost": result.objective_value,
                    "n_facilities": len(result.solution["selected_facilities"]),
                    "fixed_cost": result.metadata["fixed_cost"],
                    "transport_cost": result.metadata["transport_cost"],
                })
                
        return pd.DataFrame(results)
    
    def get_constraints(self) -> List[str]:
        """Get problem constraints."""
        constraints = [
            "All demand must be satisfied",
            "Facilities must be open to serve demand",
        ]
        
        if self.capacity_constraints:
            constraints.append("Facility capacity limits")
            
        if self.single_sourcing:
            constraints.append("Single sourcing requirement")
            
        return constraints
    
    def get_objective(self) -> str:
        """Get objective description."""
        return "Minimize total cost (fixed + transportation)"