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

	if "Operating Income" in variance_df.index:
		row = variance_df.loc["Operating Income"]
		if isinstance(row, pd.DataFrame):
			# In case of duplicate index labels.
			row = row.iloc[0]
		return row

	if "Line Item" in variance_df.columns:
		matches = variance_df.loc[variance_df["Line Item"].astype(str) == "Operating Income"]
		if not matches.empty:
			return matches.iloc[0]

	raise KeyError("Could not find 'Operating Income' row in variance_df")


def draw_waterfall_chart(variance_df: pd.DataFrame) -> go.Figure:
	"""Plot Operating Income bridge: Static -> Volume -> Spending -> Actuals."""

	op = _extract_operating_income_row(variance_df)

	static_val = float(op["Static Budget"])
	volume_var = float(op["Volume Variance"])
	spending_var = float(op["Spending Variance"])
	actual_val = float(op["Actuals"])

	fig = go.Figure(
		go.Waterfall(
			name="Operating Income",
			orientation="v",
			measure=["absolute", "relative", "relative", "total"],
			x=["Static Budget", "Volume Variance", "Spending Variance", "Actuals"],
			y=[static_val, volume_var, spending_var, actual_val],
			increasing={"marker": {"color": "green"}},
			decreasing={"marker": {"color": "red"}},
			totals={"marker": {"color": "gray"}},
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

	fig = px.histogram(
		x=profits,
		nbins=60,
		labels={"x": "Profit"},
		title="Profit Distribution (Monte Carlo)",
	)

	# Break-even line
	fig.add_vline(
		x=0,
		line_width=2,
		line_dash="dash",
		line_color="red",
		annotation_text="Break-even (0)",
		annotation_position="top left",
	)

	# Expected mean line
	fig.add_vline(
		x=mean_profit,
		line_width=2,
		line_dash="dash",
		line_color="green",
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
			f"Probability of Break-Even: {prob_be:.1%}<br>"
			f"Expected Mean Profit: {mean_profit:,.0f}"
		),
		bgcolor="rgba(255,255,255,0.8)",
		bordercolor="rgba(0,0,0,0.2)",
		borderwidth=1,
	)

	fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
	return fig

