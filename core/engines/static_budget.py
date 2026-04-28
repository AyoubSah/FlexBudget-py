from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.models.company import CompanyContext


@dataclass(slots=True)
class StaticBudgetEngine:
	company: CompanyContext
	forecasted_sales_units: dict[str, float]
	next_month_forecasted_sales_units: dict[str, float] | None = None

	def _products_df(self) -> pd.DataFrame:
		rows: list[dict[str, Any]] = []
		for p in self.company.products:
			rows.append(
				{
					"product": p.name,
					"selling_price": float(p.selling_price),
					"target_ending_inv_pct": float(p.target_ending_inv_pct),
					"labor_minutes": float(p.labor_minutes),
					"opening_inv": int(p.opening_inv),
				}
			)
		return pd.DataFrame(rows)

	def _sales_df(self, sales_units: dict[str, float] | None) -> pd.DataFrame:
		sales_units = sales_units or {}
		return pd.DataFrame(
			[{"product": str(name), "sales_units": float(units)} for name, units in sales_units.items()]
		)

	def _validate_products(self) -> None:
		known_products = {p.name for p in self.company.products}
		unknown_in_forecast = set(self.forecasted_sales_units) - known_products
		if unknown_in_forecast:
			raise ValueError(f"Forecast contains unknown products: {sorted(unknown_in_forecast)}")

		next_fcst = self.next_month_forecasted_sales_units or {}
		unknown_in_next = set(next_fcst) - known_products
		if unknown_in_next:
			raise ValueError(f"Next-month forecast contains unknown products: {sorted(unknown_in_next)}")

	def revenue_budget(self) -> pd.DataFrame:
		"""Revenue Budget: R = selling_price * sales_units."""
		self._validate_products()

		products = self._products_df()
		sales = self._sales_df(self.forecasted_sales_units)

		df = products.merge(sales, on="product", how="left")
		df["sales_units"] = df["sales_units"].fillna(0.0)
		df["revenue"] = df["selling_price"] * df["sales_units"]

		out = df[["product", "sales_units", "selling_price", "revenue"]].copy()
		out.loc[len(out)] = {
			"product": "TOTAL",
			"sales_units": out["sales_units"].sum(),
			"selling_price": float("nan"),
			"revenue": out["revenue"].sum(),
		}
		return out

	def production_budget(self) -> pd.DataFrame:
		"""Production Budget: qp = qs + (q_next * target_pct) - opening_inv."""
		out = self._production_budget_raw()

		# Integrity: production cannot be negative. If opening inventory exceeds needs,
		# cap production at zero (steady-state behavior).
		out["production_units"] = out["production_units"].clip(lower=0.0)
		return out

	def _production_budget_raw(self) -> pd.DataFrame:
		"""Internal production budget without clamping negative production."""
		self._validate_products()

		products = self._products_df()
		sales = self._sales_df(self.forecasted_sales_units)
		next_sales = self._sales_df(self.next_month_forecasted_sales_units)
		next_sales = next_sales.rename(columns={"sales_units": "next_month_sales_units"})

		df = products.merge(sales, on="product", how="left").merge(next_sales, on="product", how="left")
		df["sales_units"] = df["sales_units"].fillna(0.0)
		df["next_month_sales_units"] = df["next_month_sales_units"].fillna(0.0)

		df["target_ending_inv_units"] = df["next_month_sales_units"] * df["target_ending_inv_pct"]
		df["production_units"] = df["sales_units"] + df["target_ending_inv_units"] - df["opening_inv"]

		out = df[
			[
				"product",
				"sales_units",
				"next_month_sales_units",
				"target_ending_inv_pct",
				"target_ending_inv_units",
				"opening_inv",
				"production_units",
			]
		].copy()
		return out

	def validate_logic(self) -> bool:
		"""Return True if production would be negative (before clamping)."""
		raw = self._production_budget_raw()
		return bool((raw["production_units"] < 0).any())

	def direct_materials_budget(self) -> pd.DataFrame:
		"""Direct Materials Budget grouped by material."""
		prod_df = self.production_budget()[["product", "production_units"]].copy()

		rows: list[dict[str, Any]] = []
		for p in self.company.products:
			for material, qty_per_unit in (p.bill_of_materials or {}).items():
				rows.append(
					{
						"product": p.name,
						"material": str(material),
						"qty_per_unit": float(qty_per_unit),
					}
				)

		bom = pd.DataFrame(rows)
		if bom.empty:
			return pd.DataFrame(columns=["material", "qty_required", "unit_cost", "cost"])  # pragma: no cover

		df = bom.merge(prod_df, on="product", how="left")
		df["production_units"] = df["production_units"].fillna(0.0)
		df["qty_required"] = df["production_units"] * df["qty_per_unit"]

		unit_cost_map = {
			name: float(info.get("unit_cost", 0.0)) for name, info in (self.company.raw_materials or {}).items()
		}
		df["unit_cost"] = df["material"].map(unit_cost_map)

		missing = sorted(set(df.loc[df["unit_cost"].isna(), "material"].tolist()))
		if missing:
			raise ValueError(f"Missing unit_cost for materials: {missing}")

		df["cost"] = df["qty_required"] * df["unit_cost"]

		grouped = (
			df.groupby("material", as_index=False)
			.agg(qty_required=("qty_required", "sum"), unit_cost=("unit_cost", "first"), cost=("cost", "sum"))
			.sort_values("material")
		)
		grouped.loc[len(grouped)] = {
			"material": "TOTAL",
			"qty_required": grouped["qty_required"].sum(),
			"unit_cost": float("nan"),
			"cost": grouped["cost"].sum(),
		}
		return grouped

	def direct_labor_budget(self) -> pd.DataFrame:
		"""Direct Labor Budget: production_units * (labor_minutes/60) * labor_rate_per_hour."""
		prod = self.production_budget()[["product", "production_units"]].copy()
		products = self._products_df()[["product", "labor_minutes"]].copy()
		df = products.merge(prod, on="product", how="left")
		df["production_units"] = df["production_units"].fillna(0.0)

		labor_rate = float(self.company.labor_rate_per_hour)
		df["labor_hours"] = df["production_units"] * (df["labor_minutes"] / 60.0)
		df["labor_cost"] = df["labor_hours"] * labor_rate

		out = df[["product", "production_units", "labor_minutes", "labor_hours", "labor_cost"]].copy()
		out.loc[len(out)] = {
			"product": "TOTAL",
			"production_units": out["production_units"].sum(),
			"labor_minutes": float("nan"),
			"labor_hours": out["labor_hours"].sum(),
			"labor_cost": out["labor_cost"].sum(),
		}
		return out

	def income_statement(self) -> pd.DataFrame:
		"""Income Statement summary DataFrame."""
		revenue_df = self.revenue_budget()
		materials_df = self.direct_materials_budget()
		labor_df = self.direct_labor_budget()

		revenue_total = float(revenue_df.loc[revenue_df["product"] == "TOTAL", "revenue"].iloc[0])
		materials_total = float(materials_df.loc[materials_df["material"] == "TOTAL", "cost"].iloc[0])
		labor_total = float(labor_df.loc[labor_df["product"] == "TOTAL", "labor_cost"].iloc[0])

		material_cost_total = materials_total
		labor_cost_total = labor_total
		variable_costs = material_cost_total + labor_cost_total
		contribution_margin = revenue_total - variable_costs
		fixed_overhead = float(self.company.fixed_overhead)
		operating_income = contribution_margin - fixed_overhead

		return pd.DataFrame(
			{
				"Amount (€)": [
					revenue_total,
					material_cost_total,
					labor_cost_total,
					variable_costs,
					contribution_margin,
					fixed_overhead,
					operating_income,
				]
			},
			index=[
				"Revenue",
				"Total Material Cost",
				"Total Labor Cost",
				"Variable Costs",
				"Contribution Margin",
				"Fixed Overhead",
				"Operating Income",
			],  
		)
