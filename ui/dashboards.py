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


def draw_waterfall_chart(variance_df: pd.DataFrame) -> go.Figure:
	"""Plot Operating Income bridge: Static -> Volume -> Spending -> Actuals."""

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

	fig = go.Figure(
		go.Waterfall(
			name="Operating Income",
			orientation="v",
			measure=["absolute", "relative", "relative", "total"],
			x=["Static Budget", "Volume Variance", "Spending Variance", "Actuals"],
			y=[static_val, volume_var, spending_var, actual_val],
			increasing={"marker": {"color": "#2ECC71"}},  # Emerald Green
			decreasing={"marker": {"color": "#E74C3C"}},  # Alizarin Red
			totals={"marker": {"color": "#3498DB"}},  # Belize Blue
			connector={"line": {"color": "rgba(0,0,0,0.25)"}},
		)
	)

	fig.update_layout(
		title="Operating Income Waterfall",
		showlegend=False,
		yaxis_title="Amount",
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

