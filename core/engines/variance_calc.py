from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.engines.static_budget import StaticBudgetEngine


@dataclass(slots=True)
class VarianceEngine:
	"""Variance analysis engine (Static vs Flexible vs Actuals).

	Conventions:
	- Volume variance = Flexible Budget - Static Budget
	- Spending/Efficiency variance = Actuals - Flexible Budget
	- Favorable variances will naturally be:
		- positive for Revenue (higher actual/flexible is favorable)
		- negative for Costs (lower actual/flexible is favorable)
	"""

	static_engine: StaticBudgetEngine
	actuals_df: pd.DataFrame
	month: str | None = None

	def _actuals_filtered(self) -> pd.DataFrame:
		df = self.actuals_df.copy()
		if self.month is not None and "Month" in df.columns:
			df = df.loc[df["Month"].astype(str) == str(self.month)].copy()
		return df

	def _actuals_by_product(self) -> pd.DataFrame:
		df = self._actuals_filtered()

		rename_map = {
			"Product": "product",
			"Actual_Sales_Units": "actual_sales_units",
			"Actual_Production_Units": "actual_production_units",
			"Actual_Material_Cost": "actual_material_cost",
			"Actual_Labor_Cost": "actual_labor_cost",
			"Actual_Revenue": "actual_revenue",
		}
		for k, v in rename_map.items():
			if k in df.columns:
				df = df.rename(columns={k: v})

		required = {"product", "actual_sales_units", "actual_production_units"}
		missing_required = required - set(df.columns)
		if missing_required:
			raise ValueError(f"Actuals DF missing required columns: {sorted(missing_required)}")

		# Optional cost/revenue columns.
		for col in ("actual_material_cost", "actual_labor_cost", "actual_revenue"):
			if col not in df.columns:
				df[col] = 0.0

		grouped = (
			df.groupby("product", as_index=False)
			.agg(
				actual_sales_units=("actual_sales_units", "sum"),
				actual_production_units=("actual_production_units", "sum"),
				actual_revenue=("actual_revenue", "sum"),
				actual_material_cost=("actual_material_cost", "sum"),
				actual_labor_cost=("actual_labor_cost", "sum"),
			)
			.sort_values("product")
		)
		return grouped

	def _static_summary(self) -> pd.Series:
		revenue_df = self.static_engine.revenue_budget()
		materials_df = self.static_engine.direct_materials_budget()
		labor_df = self.static_engine.direct_labor_budget()

		revenue_total = float(revenue_df.loc[revenue_df["product"] == "TOTAL", "revenue"].iloc[0])
		materials_total = float(materials_df.loc[materials_df["material"] == "TOTAL", "cost"].iloc[0])
		labor_total = float(labor_df.loc[labor_df["product"] == "TOTAL", "labor_cost"].iloc[0])
		fixed_overhead = float(self.static_engine.company.fixed_overhead)

		variable_costs = materials_total + labor_total
		contribution_margin = revenue_total - variable_costs
		operating_income = contribution_margin - fixed_overhead

		return pd.Series(
			{
				"Revenue": revenue_total,
				"Direct Materials": materials_total,
				"Direct Labor": labor_total,
				"Variable Costs (Materials + Labor)": variable_costs,
				"Contribution Margin": contribution_margin,
				"Fixed Overhead": fixed_overhead,
				"Operating Income": operating_income,
			}
		)

	def _flexible_budget_summary(self) -> pd.Series:
		"""Flexible budget using actual volumes but standard prices/standards."""
		company = self.static_engine.company

		actuals = self._actuals_by_product()
		products = self.static_engine._products_df()  # standard parameters

		# Flexible revenue: actual sales units * standard selling price
		rev = products[["product", "selling_price"]].merge(actuals[["product", "actual_sales_units"]], on="product")
		rev["flex_revenue"] = rev["selling_price"] * rev["actual_sales_units"]
		revenue_total = float(rev["flex_revenue"].sum())

		# Flexible materials: actual production units * standard sigma * standard unit costs
		bom_rows: list[dict[str, Any]] = []
		for p in company.products:
			for material, qty_per_unit in (p.bill_of_materials or {}).items():
				bom_rows.append(
					{
						"product": p.name,
						"material": str(material),
						"qty_per_unit": float(qty_per_unit),
					}
				)
		bom = pd.DataFrame(bom_rows)
		if bom.empty:
			materials_total = 0.0
		else:
			unit_cost_map = {
				name: float(info.get("unit_cost", 0.0)) for name, info in (company.raw_materials or {}).items()
			}
			mat = bom.merge(actuals[["product", "actual_production_units"]], on="product", how="left")
			mat["actual_production_units"] = mat["actual_production_units"].fillna(0.0)
			mat["qty_required"] = mat["actual_production_units"] * mat["qty_per_unit"]
			mat["unit_cost"] = mat["material"].map(unit_cost_map)

			missing = sorted(set(mat.loc[mat["unit_cost"].isna(), "material"].tolist()))
			if missing:
				raise ValueError(f"Missing unit_cost for materials: {missing}")

			mat["flex_material_cost"] = mat["qty_required"] * mat["unit_cost"]
			materials_total = float(mat["flex_material_cost"].sum())

		# Flexible labor: actual production units * standard labor minutes * standard labor rate
		lab = products[["product", "labor_minutes"]].merge(
			actuals[["product", "actual_production_units"]], on="product", how="left"
		)
		lab["actual_production_units"] = lab["actual_production_units"].fillna(0.0)
		labor_rate = float(company.labor_rate_per_hour)
		lab["flex_labor_cost"] = lab["actual_production_units"] * (lab["labor_minutes"] / 60.0) * labor_rate
		labor_total = float(lab["flex_labor_cost"].sum())

		fixed_overhead = float(company.fixed_overhead)
		variable_costs = materials_total + labor_total
		contribution_margin = revenue_total - variable_costs
		operating_income = contribution_margin - fixed_overhead

		return pd.Series(
			{
				"Revenue": revenue_total,
				"Direct Materials": materials_total,
				"Direct Labor": labor_total,
				"Variable Costs (Materials + Labor)": variable_costs,
				"Contribution Margin": contribution_margin,
				"Fixed Overhead": fixed_overhead,
				"Operating Income": operating_income,
			}
		)

	def _actuals_summary(self) -> pd.Series:
		company = self.static_engine.company
		actuals = self._actuals_by_product()

		# Revenue: prefer provided actual_revenue; if missing/zero, fall back to standard price * actual sales.
		if "actual_revenue" in actuals.columns and float(actuals["actual_revenue"].sum()) != 0.0:
			revenue_total = float(actuals["actual_revenue"].sum())
		else:
			products = self.static_engine._products_df()[["product", "selling_price"]]
			rev = products.merge(actuals[["product", "actual_sales_units"]], on="product", how="inner")
			revenue_total = float((rev["selling_price"] * rev["actual_sales_units"]).sum())

		materials_total = float(actuals.get("actual_material_cost", pd.Series([0.0])).sum())
		labor_total = float(actuals.get("actual_labor_cost", pd.Series([0.0])).sum())

		# If an actual fixed overhead column exists, use it; otherwise assume it equals standard fixed overhead.
		fixed_overhead = float(company.fixed_overhead)
		df = self._actuals_filtered()
		if "Actual_Fixed_Overhead" in df.columns:
			fixed_overhead = float(pd.to_numeric(df["Actual_Fixed_Overhead"], errors="coerce").fillna(0.0).sum())

		variable_costs = materials_total + labor_total
		contribution_margin = revenue_total - variable_costs
		operating_income = contribution_margin - fixed_overhead

		return pd.Series(
			{
				"Revenue": revenue_total,
				"Direct Materials": materials_total,
				"Direct Labor": labor_total,
				"Variable Costs (Materials + Labor)": variable_costs,
				"Contribution Margin": contribution_margin,
				"Fixed Overhead": fixed_overhead,
				"Operating Income": operating_income,
			}
		)

	def variance_report(self) -> pd.DataFrame:
		"""Return combined variance report.

		Columns: [Static Budget, Volume Variance, Flexible Budget, Spending Variance, Actuals]
		"""
		static_s = self._static_summary()
		flexible_s = self._flexible_budget_summary()
		actuals_s = self._actuals_summary()

		# Sales volume (units): static from forecast, flexible/actuals from actual sales
		static_units = float(sum(self.static_engine.forecasted_sales_units.values()))
		actuals_by_prod = self._actuals_by_product()
		flex_units = float(actuals_by_prod["actual_sales_units"].sum())
		actual_units = flex_units

		numeric_static = pd.Series(
			{
				"Sales Volume (Units)": static_units,
				"Revenue (€)": static_s["Revenue"],
				"Direct Materials Cost (€)": static_s["Direct Materials"],
				"Direct Labor Cost (€)": static_s["Direct Labor"],
				"Variable Costs (€)": static_s["Variable Costs (Materials + Labor)"],
				"Fixed Overhead (€)": static_s["Fixed Overhead"],
				"Operating Income (€)": static_s["Operating Income"],
			}
		)

		numeric_flexible = pd.Series(
			{
				"Sales Volume (Units)": flex_units,
				"Revenue (€)": flexible_s["Revenue"],
				"Direct Materials Cost (€)": flexible_s["Direct Materials"],
				"Direct Labor Cost (€)": flexible_s["Direct Labor"],
				"Variable Costs (€)": flexible_s["Variable Costs (Materials + Labor)"],
				"Fixed Overhead (€)": flexible_s["Fixed Overhead"],
				"Operating Income (€)": flexible_s["Operating Income"],
			}
		)

		numeric_actuals = pd.Series(
			{
				"Sales Volume (Units)": actual_units,
				"Revenue (€)": actuals_s["Revenue"],
				"Direct Materials Cost (€)": actuals_s["Direct Materials"],
				"Direct Labor Cost (€)": actuals_s["Direct Labor"],
				"Variable Costs (€)": actuals_s["Variable Costs (Materials + Labor)"],
				"Fixed Overhead (€)": actuals_s["Fixed Overhead"],
				"Operating Income (€)": actuals_s["Operating Income"],
			}
		)

		volume_var = numeric_flexible - numeric_static
		spending_var = numeric_actuals - numeric_flexible

		# Status labeling rules
		revenue_like_rows = {"Revenue (€)", "Operating Income (€)"}
		cost_rows = {
			"Direct Materials Cost (€)",
			"Direct Labor Cost (€)",
			"Variable Costs (€)",
			"Fixed Overhead (€)",
		}

		def _variance_status(line_item: str, variance_value: float, is_spending: bool) -> str:
			# For the units row, statuses are not meaningful.
			if line_item == "Sales Volume (Units)":
				return ""
			# Revenue-like: positive variance is favorable.
			if line_item in revenue_like_rows:
				return "Favorable (F)" if variance_value > 0 else "Unfavorable (U)"
			# Costs: negative variance (lower cost) is favorable.
			if line_item in cost_rows:
				return "Favorable (F)" if variance_value < 0 else "Unfavorable (U)"
			# Fallback: treat as revenue-like
			return "Favorable (F)" if variance_value > 0 else "Unfavorable (U)"

		volume_status = pd.Series(
			{idx: _variance_status(idx, float(volume_var[idx]), is_spending=False) for idx in numeric_static.index}
		)
		spending_status = pd.Series(
			{idx: _variance_status(idx, float(spending_var[idx]), is_spending=True) for idx in numeric_static.index}
		)

		out = pd.DataFrame(
			{
				"Static Budget (€)": numeric_static,
				"Volume Variance (€)": volume_var,
				"Volume Status": volume_status,
				"Flexible Budget (€)": numeric_flexible,
				"Spending Variance (€)": spending_var,
				"Spending Status": spending_status,
				"Actuals (€)": numeric_actuals,
			}
		)

		# Ensure requested column order
		out = out[
			[
				"Static Budget (€)",
				"Volume Variance (€)",
				"Volume Status",
				"Flexible Budget (€)",
				"Spending Variance (€)",
				"Spending Status",
				"Actuals (€)",
			]
		]
		return out
