from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.models.company import CompanyContext


@dataclass(slots=True)
class MonteCarloSimulator:
	"""Monte Carlo simulator for budget uncertainty.

	The simulation is intentionally vectorized (no per-iteration loops):
	- Sales volumes are simulated per product.
	- Material cost uncertainty is modeled via a per-iteration multiplier applied
	  to standard material cost per unit.
	"""

	company: CompanyContext
	baseline_forecasted_volumes: dict[str, float]

	def _validate_products(self) -> None:
		known = {p.name for p in self.company.products}
		unknown = set(self.baseline_forecasted_volumes) - known
		if unknown:
			raise ValueError(f"Baseline forecast contains unknown products: {sorted(unknown)}")

	def _standard_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return aligned arrays for (base_volumes, selling_prices, std_var_cost_per_unit).

		std_var_cost_per_unit = std_material_cost_per_unit + std_labor_cost_per_unit
		(material component kept separate inside run_simulation to apply multipliers).
		"""

		self._validate_products()

		product_names = [p.name for p in self.company.products]
		base_volumes = np.array([float(self.baseline_forecasted_volumes.get(name, 0.0)) for name in product_names])
		selling_prices = np.array([float(p.selling_price) for p in self.company.products])

		# Standard material cost per unit: sum_i (sigma_i * unit_cost_i)
		unit_cost_map = {
			name: float(info.get("unit_cost", 0.0)) for name, info in (self.company.raw_materials or {}).items()
		}

		std_material_cost_per_unit: list[float] = []
		for p in self.company.products:
			cost = 0.0
			for material, qty_per_unit in (p.bill_of_materials or {}).items():
				if material not in unit_cost_map:
					raise ValueError(f"Missing unit_cost for material '{material}'")
				cost += float(qty_per_unit) * float(unit_cost_map[material])
			std_material_cost_per_unit.append(cost)

		std_material_cost_per_unit_arr = np.array(std_material_cost_per_unit, dtype=float)

		# Standard labor cost per unit: (labor_minutes/60) * labor_rate_per_hour
		labor_rate = float(self.company.labor_rate_per_hour)
		labor_minutes = np.array([float(p.labor_minutes) for p in self.company.products])
		std_labor_cost_per_unit_arr = (labor_minutes / 60.0) * labor_rate

		# Keep both arrays for simulation.
		return base_volumes, selling_prices, std_material_cost_per_unit_arr + std_labor_cost_per_unit_arr

	def run_simulation(
		self,
		num_iterations: int = 10_000,
		volume_volatility: float = 0.10,
		price_volatility: float = 0.05,
		seed: int | None = None,
	) -> dict[str, Any]:
		"""Run a Monte Carlo simulation.

		Args:
			num_iterations: Number of simulated scenarios.
			volume_volatility: Std dev as a fraction of baseline volume.
			price_volatility: Std dev for material cost multiplier (mean=1.0).
			seed: Optional RNG seed for reproducibility.

		Returns:
			Dict with keys: metrics (dict) and profits (np.ndarray).
		"""

		if num_iterations <= 0:
			raise ValueError("num_iterations must be > 0")
		if volume_volatility < 0:
			raise ValueError("volume_volatility must be >= 0")
		if price_volatility < 0:
			raise ValueError("price_volatility must be >= 0")

		self._validate_products()

		rng = np.random.default_rng(seed)

		product_names = [p.name for p in self.company.products]
		base_volumes = np.array([float(self.baseline_forecasted_volumes.get(name, 0.0)) for name in product_names])
		selling_prices = np.array([float(p.selling_price) for p in self.company.products])

		unit_cost_map = {
			name: float(info.get("unit_cost", 0.0)) for name, info in (self.company.raw_materials or {}).items()
		}
		std_material_cost_per_unit = np.array(
			[
				sum(float(qty) * float(unit_cost_map[mat]) for mat, qty in (p.bill_of_materials or {}).items())
				for p in self.company.products
			],
			dtype=float,
		)

		labor_rate = float(self.company.labor_rate_per_hour)
		labor_minutes = np.array([float(p.labor_minutes) for p in self.company.products], dtype=float)
		std_labor_cost_per_unit = (labor_minutes / 60.0) * labor_rate

		# Simulate volumes: shape (N, P)
		vol_std = volume_volatility * base_volumes
		volumes = rng.normal(loc=base_volumes, scale=vol_std, size=(num_iterations, base_volumes.size))
		volumes = np.clip(volumes, 0.0, None)

		# Simulate material cost multiplier per iteration: shape (N,)
		material_multiplier = rng.normal(loc=1.0, scale=price_volatility, size=num_iterations)
		material_multiplier = np.clip(material_multiplier, 0.0, None)

		# Vectorized income equation: rho = R - VC - k
		# Revenue: sum_p (volume_np * selling_price_p)
		revenue = volumes @ selling_prices

		# Variable costs:
		# - Materials: multiplier_n * sum_p (volume_np * std_material_cost_per_unit_p)
		materials_base = volumes @ std_material_cost_per_unit
		materials_cost = material_multiplier * materials_base
		# - Labor: sum_p (volume_np * std_labor_cost_per_unit_p)
		labor_cost = volumes @ std_labor_cost_per_unit
		variable_costs = materials_cost + labor_cost

		fixed_overhead = float(self.company.fixed_overhead)
		profits = revenue - variable_costs - fixed_overhead

		metrics = {
			"expected_mean_profit": float(np.mean(profits)),
			"p05_worst_case_profit": float(np.percentile(profits, 5)),
			"p95_best_case_profit": float(np.percentile(profits, 95)),
			"probability_of_break_even": float(np.mean(profits > 0.0)),
		}

		return {"metrics": metrics, "profits": profits}
