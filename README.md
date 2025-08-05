# PyMC-Supply-Chain: Bayesian Supply Chain Optimization

<div align="center">

![PyMC-Supply-Chain Logo](docs/source/_static/supply-chain-logo.png)

[![PyPI Version](https://img.shields.io/pypi/v/pymc-supply-chain.svg)](https://pypi.python.org/pypi/pymc-supply-chain)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/pymc-supply-chain.svg)](https://pypi.org/project/pymc-supply-chain/)

</div>

## 🚀 Enterprise-Grade Supply Chain Analytics & Optimization

PyMC-Supply-Chain brings the power of **Bayesian modeling** and **probabilistic programming** to supply chain optimization. Built on top of PyMC, this comprehensive toolkit helps organizations make smarter, data-driven decisions under uncertainty.

### 🎯 Key Benefits

- **Handle Uncertainty**: Quantify and manage uncertainty in demand, lead times, and costs
- **Optimize Under Risk**: Make robust decisions that account for variability
- **End-to-End Solutions**: From demand forecasting to network design
- **Industry-Ready**: Production-grade implementations of proven algorithms
- **Flexible & Extensible**: Easily customize models for your specific needs

---

## 📦 Core Modules

### 1. **Demand Forecasting** (`pymc_supply_chain.demand`)
Advanced Bayesian demand forecasting with uncertainty quantification:

- **Base Demand Model**: Trend, seasonality, and external factors
- **Hierarchical Models**: Multi-location/product forecasting with pooling
- **Seasonal Models**: Multiple seasonality patterns with Fourier series
- **Intermittent Demand**: Croston's method and zero-inflated models for spare parts

```python
from pymc_supply_chain.demand import SeasonalDemandModel

# Create and fit model
model = SeasonalDemandModel(
    yearly_seasonality=10,
    weekly_seasonality=3,
    changepoint_prior_scale=0.05
)
model.fit(demand_data)

# Generate probabilistic forecasts
forecast = model.forecast(steps=30, include_uncertainty=True)
```

### 2. **Inventory Optimization** (`pymc_supply_chain.inventory`)
Sophisticated inventory management under uncertainty:

- **Newsvendor Model**: Single-period optimization with demand learning
- **EOQ Models**: Economic order quantity with extensions for discounts and backorders
- **Safety Stock**: Bayesian optimization with demand and lead time uncertainty
- **Multi-Echelon**: Network-wide inventory optimization

```python
from pymc_supply_chain.inventory import NewsvendorModel

# Optimize order quantity
newsvendor = NewsvendorModel(
    unit_cost=10,
    selling_price=25,
    salvage_value=5,
    demand_distribution="gamma"
)
newsvendor.fit(historical_demand)
optimal = newsvendor.calculate_optimal_quantity()
```

### 3. **Network Design** (`pymc_supply_chain.network`)
Strategic supply chain network optimization:

- **Facility Location**: Warehouse and DC placement optimization
- **Network Flow**: Multi-commodity flow optimization
- **Capacity Planning**: Right-sizing facilities under demand uncertainty

```python
from pymc_supply_chain.network import FacilityLocationOptimizer

# Optimize facility locations
optimizer = FacilityLocationOptimizer(
    demand_locations=customer_data,
    candidate_locations=candidate_sites,
    fixed_costs=facility_costs
)
result = optimizer.optimize(max_facilities=5, service_distance=200)
```

### 4. **Transportation & Routing** (`pymc_supply_chain.transportation`)
Vehicle routing and transportation optimization:

- **VRP Solver**: Vehicle routing with time windows and capacity
- **Fleet Sizing**: Optimal fleet composition
- **Mode Selection**: Multi-modal transportation decisions

### 5. **Risk Assessment** (`pymc_supply_chain.risk`)
Supply chain risk modeling and mitigation:

- **Disruption Modeling**: Supplier reliability and risk assessment
- **Scenario Analysis**: Multi-scenario planning
- **Resilience Metrics**: Network robustness evaluation

---

## 🚀 Quick Start

### Installation

```bash
pip install pymc-supply-chain
```

Or with conda:

```bash
conda install -c conda-forge pymc-supply-chain
```

### Basic Example: End-to-End Supply Chain Optimization

```python
import pandas as pd
from pymc_supply_chain import (
    DemandForecastModel,
    SafetyStockOptimizer,
    FacilityLocationOptimizer
)

# 1. Forecast demand
demand_model = DemandForecastModel(seasonality=12)
demand_model.fit(historical_data)
demand_forecast = demand_model.forecast(steps=90)

# 2. Optimize safety stock
safety_stock = SafetyStockOptimizer(
    holding_cost=2.0,
    stockout_cost=50.0,
    service_type="fill"
)
safety_stock.fit(demand_data)
optimal_ss = safety_stock.calculate_safety_stock(confidence_level=0.95)

# 3. Design distribution network
network_opt = FacilityLocationOptimizer(
    demand_locations=demand_forecast,
    candidate_locations=warehouse_sites,
    fixed_costs=warehouse_costs
)
network_design = network_opt.optimize(budget=1_000_000)
```

---

## 📊 Real-World Applications

### Manufacturing
- Production planning under demand uncertainty
- Raw material inventory optimization
- Multi-tier supplier network design

### Retail & E-commerce
- Store replenishment optimization
- Seasonal demand forecasting
- Distribution center location planning

### Healthcare
- Medical supply inventory management
- Hospital network optimization
- Emergency stockpile planning

### Logistics
- Fleet sizing and routing
- Cross-docking optimization
- Last-mile delivery planning

---

## 🔬 Advanced Features

### Uncertainty Quantification
All models provide full posterior distributions, not just point estimates:

```python
# Get full posterior predictive distribution
with model:
    posterior_samples = pm.sample_posterior_predictive(trace)
    
# Calculate risk metrics
value_at_risk = np.percentile(posterior_samples, 5)
conditional_value_at_risk = posterior_samples[posterior_samples <= value_at_risk].mean()
```

### Multi-Objective Optimization
Balance competing objectives like cost, service, and risk:

```python
optimizer.optimize(
    objectives=["cost", "service_level", "carbon_footprint"],
    weights=[0.5, 0.3, 0.2]
)
```

### What-If Scenario Analysis
Test strategies under different future scenarios:

```python
scenarios = {
    "baseline": {"demand_growth": 0.05, "fuel_cost": 3.0},
    "recession": {"demand_growth": -0.10, "fuel_cost": 2.5},
    "boom": {"demand_growth": 0.15, "fuel_cost": 4.0}
}

results = model.scenario_analysis(scenarios)
```

---

## 📈 Visualization & Reporting

Built-in visualization tools for supply chain insights:

```python
from pymc_supply_chain.visualization import (
    plot_network_flows,
    plot_inventory_levels,
    plot_service_cost_tradeoff
)

# Interactive network visualization
plot_network_flows(network_solution, demand_flows)

# Inventory analysis dashboard
plot_inventory_levels(inventory_simulation, service_targets)
```

---

## 🤝 Integration

### ERP/WMS Systems
- SAP integration modules
- Oracle SCM connectors
- WMS data adapters

### Business Intelligence
- Export to PowerBI/Tableau
- Automated reporting
- KPI dashboards

### Cloud Deployment
- AWS/Azure/GCP ready
- Containerized deployment
- REST API framework

---

## 🎓 Learning Resources

### Documentation
- [Getting Started Guide](https://pymc-supply-chain.readthedocs.io/en/latest/getting_started.html)
- [API Reference](https://pymc-supply-chain.readthedocs.io/en/latest/api.html)
- [Theory & Methods](https://pymc-supply-chain.readthedocs.io/en/latest/theory.html)

### Tutorials
- [Demand Forecasting Tutorial](examples/demand_forecasting_tutorial.ipynb)
- [Inventory Optimization Guide](examples/inventory_optimization.ipynb)
- [Network Design Walkthrough](examples/network_design.ipynb)

### Case Studies
- [Retail Chain Optimization](case_studies/retail_optimization.ipynb)
- [Manufacturing Network Redesign](case_studies/manufacturing_network.ipynb)
- [Healthcare Supply Chain](case_studies/healthcare_supply.ipynb)

---

## 🏗️ Architecture

PyMC-Supply-Chain follows a modular architecture:

```
pymc_supply_chain/
├── demand/          # Forecasting models
├── inventory/       # Stock optimization
├── network/         # Network design
├── transportation/  # Routing & logistics
├── risk/           # Risk assessment
├── visualization/   # Plotting tools
└── utils/          # Shared utilities
```

Each module provides:
- **Model builders** following consistent API patterns
- **Optimizers** for deterministic problems
- **Simulators** for testing strategies
- **Analyzers** for insights and diagnostics

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourcompany/pymc-supply-chain.git
cd pymc-supply-chain

# Create development environment
conda env create -f environment.yml
conda activate pymc-supply-chain-dev

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

---

## 🏢 Professional Services

For organizations looking to implement PyMC-Supply-Chain in production:

- **Consulting**: Custom model development and optimization
- **Training**: Workshops and certification programs
- **Support**: Enterprise support packages available

Contact: [sales@yourcompany.com](mailto:sales@yourcompany.com)

---

## 📄 License

PyMC-Supply-Chain is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built on top of these excellent projects:
- [PyMC](https://www.pymc.io/) - Probabilistic programming
- [ArviZ](https://arviz-devs.github.io/arviz/) - Exploratory analysis
- [NetworkX](https://networkx.org/) - Network analysis
- [PuLP](https://coin-or.github.io/pulp/) - Linear programming

---

## 📚 Citation

If you use PyMC-Supply-Chain in your research, please cite:

```bibtex
@software{pymc_supply_chain,
  title = {PyMC-Supply-Chain: Bayesian Supply Chain Optimization},
  author = {Your Company},
  year = {2025},
  url = {https://github.com/yourcompany/pymc-supply-chain}
}
```

---

<div align="center">

**Ready to transform your supply chain?**

[Get Started](https://pymc-supply-chain.readthedocs.io) | [View Examples](examples/) | [API Docs](https://pymc-supply-chain.readthedocs.io/en/latest/api.html)

</div>