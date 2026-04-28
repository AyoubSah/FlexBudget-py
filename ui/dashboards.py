from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _extract_operating_income_row(variance_df: pd.DataFrame) -> pd.Series:
	"""Return the Operating Income row as a Series.

	Supports either a labeled index (preferred) or a column named 'Line Item'.
	"""

	for key in ("Operating Income (€)", "Operating Income"):
		if key in variance_df.index:
			row = variance_df.loc[key]
			if isinstance(row, pd.DataFrame):
				# In case of duplicate index labels.
				row = row.iloc[0]
			return row

	# Backward-compatible support if the report has a line-item column.
	if "Line Item" in variance_df.columns:
		matches = variance_df.loc[
			variance_df["Line Item"].astype(str).isin(["Operating Income (€)", "Operating Income"])
		]
		if not matches.empty:
			return matches.iloc[0]

	raise KeyError("Could not find Operating Income row in variance_df")


def draw_waterfall_chart(variance_df: pd.DataFrame, *, differences_only: bool = False) -> go.Figure:
	"""Plot Operating Income waterfall.

	Modes:
	- Absolute (default): Static -> Volume -> Spending -> Actuals.
	- Differences only: start at 0 and show only variances summing to Net Change.
	"""

	op = _extract_operating_income_row(variance_df)

	# Support both old and new column naming.
	static_col = "Static Budget (€)" if "Static Budget (€)" in variance_df.columns else "Static Budget"
	volume_col = "Volume Variance (€)" if "Volume Variance (€)" in variance_df.columns else "Volume Variance"
	spending_col = "Spending Variance (€)" if "Spending Variance (€)" in variance_df.columns else "Spending Variance"
	actuals_col = "Actuals (€)" if "Actuals (€)" in variance_df.columns else "Actuals"

	static_val = float(op[static_col])
	volume_var = float(op[volume_col])
	spending_var = float(op[spending_col])
	actual_val = float(op[actuals_col])
	net_change = volume_var + spending_var

	if differences_only:
		measure = ["relative", "relative", "total"]
		x = ["Volume Variance", "Spending Variance", "Net Change"]
		y = [volume_var, spending_var, net_change]
		yaxis_title = "Amount (€)"
		title = "Operating Income Variances (Differences Only)"
	else:
		measure = ["absolute", "relative", "relative", "total"]
		x = ["Static Budget", "Volume Variance", "Spending Variance", "Actuals"]
		y = [static_val, volume_var, spending_var, actual_val]
		yaxis_title = "Amount (€)"
		title = "Operating Income Waterfall"

	fig = go.Figure(
		go.Waterfall(
			name="Operating Income",
			orientation="v",
			measure=measure,
			x=x,
			y=y,
			increasing={"marker": {"color": "#2ECC71"}},  # Emerald Green
			decreasing={"marker": {"color": "#E74C3C"}},  # Alizarin Red
			totals={"marker": {"color": "#3498DB"}},  # Belize Blue
			connector={"line": {"color": "rgba(0,0,0,0.25)"}},
		)
	)

	fig.update_layout(
		title=title,
		showlegend=False,
		yaxis_title=yaxis_title,
		margin=dict(l=10, r=10, t=50, b=10),
	)
	return fig


def draw_profit_distribution(profit_array: np.ndarray, metrics_dict: dict[str, Any]) -> go.Figure:
	"""Plot Monte Carlo profit distribution with break-even and expected mean."""

	profits = np.asarray(profit_array, dtype=float)
	mean_profit = float(metrics_dict.get("expected_mean_profit", np.mean(profits)))
	prob_be = float(metrics_dict.get("probability_of_break_even", float(np.mean(profits > 0.0))))
	p95_profit = float(metrics_dict.get("p95_best_case_profit", float(np.percentile(profits, 95))))

	with st.container():
		c1, c2, c3 = st.columns(3)
		c1.metric("Expected Profit", f"€{mean_profit:,.2f}")
		c2.metric("Break-Even Probability (%)", f"{(prob_be * 100):.2f}%")
		c3.metric("95th Percentile (Best Case)", f"€{p95_profit:,.2f}")

		fig = px.histogram(
			x=profits,
			nbins=60,
			labels={"x": "Profit (€)"},
			title="Profit Distribution (Monte Carlo)",
		)

		# Break-even line
		fig.add_vline(
			x=0,
			line_width=2,
			line_dash="dash",
			line_color="#E74C3C",
			annotation_text="Break-even (0)",
			annotation_position="top left",
		)

		# Expected mean line
		fig.add_vline(
			x=mean_profit,
			line_width=2,
			line_dash="dash",
			line_color="#2ECC71",
			annotation_text="Expected mean",
			annotation_position="top right",
		)

		fig.add_annotation(
			xref="paper",
			yref="paper",
			x=0.01,
			y=0.99,
			showarrow=False,
			align="left",
			text=(
				f"Probability of Break-Even: {(prob_be * 100):.2f}%<br>"
				f"Expected Mean Profit: €{mean_profit:,.2f}"
			),
			bgcolor="rgba(255,255,255,0.8)",
			bordercolor="rgba(0,0,0,0.2)",
			borderwidth=1,
		)

		fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
		st.plotly_chart(fig, use_container_width=True)

	return fig


def draw_variance_analysis_bars(variance_df: pd.DataFrame) -> go.Figure:
	"""Grouped horizontal bars for volume vs spending variances.

	Excludes Sales Volume and any TOTAL rows to keep the scale focused on errors.
	"""

	volume_col = "Volume Variance (€)" if "Volume Variance (€)" in variance_df.columns else "Volume Variance"
	spending_col = (
		"Spending Variance (€)" if "Spending Variance (€)" in variance_df.columns else "Spending Variance"
	)

	missing = [c for c in (volume_col, spending_col) if c not in variance_df.columns]
	if missing:
		raise KeyError(f"variance_df missing required columns: {missing}")

	# Get line items from index (preferred) or fallback to a column.
	if variance_df.index is not None and variance_df.index.nlevels == 1:
		line_items = variance_df.index.to_series().astype(str).str.strip()
		base = variance_df.copy()
		base = base.assign(**{"Line Item": line_items.values})
	else:
		base = variance_df.copy()
		if "Line Item" not in base.columns:
			raise KeyError("variance_df must have a labeled index or a 'Line It" \
			"em' column")

	li = base["Line Item"].astype(str).str.strip()
	mask_sales_volume = li.str.contains("Sales Volume", case=False, na=False)
	mask_total = li.str.fullmatch("TOTAL", case=False, na=False) | li.str.contains("\bTOTAL\b", case=False, na=False)

	plot_df = base.loc[~(mask_sales_volume | mask_total), ["Line Item", volume_col, spending_col]].copy()

	# Long format for grouped bars.
	long_df = plot_df.melt(
		id_vars=["Line Item"],
		value_vars=[volume_col, spending_col],
		var_name="Variance Type",
		value_name="Variance (€)",
	)

	fig = px.bar(
		long_df,
		y="Line Item",
		x="Variance (€)",
		color="Variance Type",
		barmode="group",
		orientation="h",
		title="Variance Analysis (Volume vs Spending)",
	)
	fig.update_layout(
		xaxis_title="Variance (€)",
		yaxis_title="",
		margin=dict(l=10, r=10, t=50, b=10),
		legend_title_text="",
	)
	return fig


def draw_detailed_variance_bar_chart(variance_df: pd.DataFrame) -> go.Figure:
	"""Detailed grouped variance bars (Volume vs Spending).

	Filters out 'Sales Volume (Units)' and 'Operating Income (€)' rows to focus on
	category-level variances only.
	"""

	# Choose variance columns (support both old and new naming).
	volume_col = "Volume Variance (€)" if "Volume Variance (€)" in variance_df.columns else "Volume Variance"
	spending_col = (
		"Spending Variance (€)" if "Spending Variance (€)" in variance_df.columns else "Spending Variance"
	)

	missing = [c for c in (volume_col, spending_col) if c not in variance_df.columns]
	if missing:
		raise KeyError(f"variance_df missing required columns: {missing}")

	df = variance_df.copy()
	# Data cleaning: filter out specific summary rows.
	idx = df.index.to_series().astype(str).str.strip()
	df = df.loc[~idx.isin(["Sales Volume (Units)", "Operating Income (€)"])].copy()

	# Data transformation: wide -> long using reset_index().melt().
	df.index.name = "Line Item"
	long_df = (
		df.reset_index()
		.melt(
			id_vars=["Line Item"],
			value_vars=[volume_col, spending_col],
			var_name="variable",
			value_name="value",
		)
		.dropna(subset=["value"])
	)

	fig = px.bar(
		long_df,
		y="Line Item",
		x="value",
		color="variable",
		barmode="group",
		orientation="h",
		template="plotly_white",
		color_discrete_map={
			volume_col: "#3498DB",  # Volume
			spending_col: "#9B59B6",  # Spending
		},
		title="Detailed Variance Analysis (Volume vs Spending)",
	)

	fig.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.4)")
	fig.update_layout(
		xaxis_title="Variance (€)",
		yaxis_title="",
		legend_title_text="",
		legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
		margin=dict(l=10, r=10, t=50, b=10),
	)
	return fig

