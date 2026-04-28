import pandas as pd
import streamlit as st

from pathlib import Path

from core.engines.static_budget import StaticBudgetEngine
from core.engines.stochastic_mc import MonteCarloSimulator
from core.engines.variance_calc import VarianceEngine
from core.models.company import CompanyContext
from ui.config_editor import render_config_editor
from ui.dashboards import draw_detailed_variance_bar_chart, draw_profit_distribution, draw_waterfall_chart


def get_next_month(current_month: str, all_months: list[str]) -> str:
	"""Return the next month in the provided sequence.

	If current_month is the last element (or not found), return current_month (steady-state).
	"""
	try:
		idx = all_months.index(str(current_month))
	except ValueError:
		return str(current_month)
	if idx >= len(all_months) - 1:
		return str(current_month)
	return str(all_months[idx + 1])


def _fmt_currency(value: object) -> str:
	try:
		n = float(value)  # type: ignore[arg-type]
	except Exception:
		return ""
	if pd.isna(n):
		return ""
	if n < 0:
		return f"-€{abs(n):,.2f}"
	return f"€{n:,.2f}"


def _fmt_units(value: object) -> str:
	try:
		n = float(value)  # type: ignore[arg-type]
	except Exception:
		return ""
	if pd.isna(n):
		return ""
	if n < 0:
		return f"-{abs(n):,.0f}"
	return f"{n:,.0f}"


def _fmt_percent(value: object) -> str:
	try:
		n = float(value)  # type: ignore[arg-type]
	except Exception:
		return ""
	if pd.isna(n):
		return ""
	return f"{(n * 100):.1f}%"


def _style_fav_unfav(value: object) -> str:
	text = str(value)
	if "Favorable" in text:
		return "color: green; font-weight: 600;"
	if "Unfavorable" in text:
		return "color: red; font-weight: 600;"
	return ""


def render_dataframe(
	df: pd.DataFrame,
	*,
	currency_cols: list[str] | None = None,
	unit_cols: list[str] | None = None,
	pct_cols: list[str] | None = None,
	fav_unfav_cols: list[str] | None = None,
	units_index_rows: list[str] | None = None,
) -> None:
	"""Render a dataframe with consistent formatting and optional variance coloring."""
	currency_cols = currency_cols or []
	unit_cols = unit_cols or []
	pct_cols = pct_cols or []
	fav_unfav_cols = fav_unfav_cols or []
	units_index_rows = units_index_rows or []

	styler = df.style

	# Column-based formatting
	for col in currency_cols:
		if col in df.columns:
			styler = styler.format(_fmt_currency, subset=[col])
	for col in unit_cols:
		if col in df.columns:
			styler = styler.format(_fmt_units, subset=[col])
	for col in pct_cols:
		if col in df.columns:
			styler = styler.format(_fmt_percent, subset=[col])

	# Row-based overrides (e.g., units row inside an otherwise currency table)
	for row_name in units_index_rows:
		if row_name in df.index:
			numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
			if numeric_cols:
				styler = styler.format(_fmt_units, subset=pd.IndexSlice[[row_name], numeric_cols])

	# Favorable / Unfavorable coloring
	for col in fav_unfav_cols:
		if col in df.columns:
			styler = styler.map(_style_fav_unfav, subset=[col])

	st.dataframe(styler, use_container_width=True)


st.set_page_config(page_title="FlexBudget-Py", layout="wide")
st.title("FlexBudget-Py")


page = st.sidebar.radio(
	"Navigation",
	[
		"1. Parameters Setup",
		"2. Master Budget",
		"3. Variance Analysis",
		"4. Risk Simulation",
	],
)


st.sidebar.subheader("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Actuals CSV", type=["csv"])


ctx: CompanyContext | None
try:
	ctx = CompanyContext()
	exc: Exception | None = None
except Exception as e:
	ctx = None
	exc = e

available_months = list(ctx.forecasted_sales_units.keys()) if ctx is not None else []
selected_month: str | None = None
next_month: str | None = None

st.sidebar.subheader("Month")
if available_months:
	selected_month = st.sidebar.selectbox("Selected Month", options=available_months)
	next_month = get_next_month(selected_month, available_months)
	st.sidebar.caption(f"Next month: {next_month}")
else:
	if exc is not None:
		st.sidebar.error(f"Failed to load config: {exc}")
	st.sidebar.warning("No forecasted months found in config. Add them in Parameters Setup.")


if page == "1. Parameters Setup":
	render_config_editor()


elif page == "2. Master Budget":
	if ctx is None:
		st.error("Configuration could not be loaded. Fix it in Parameters Setup.")
		st.stop()
	if selected_month is None or next_month is None:
		st.error("No forecasted months found in config forecasted_sales_units")
		st.stop()

	try:
		forecast_current = ctx.forecasted_sales_units[selected_month]
		forecast_next = ctx.forecasted_sales_units.get(next_month, forecast_current)
	except Exception:
		st.error(f"Missing '{selected_month}' in config forecasted_sales_units")
		st.stop()

	engine = StaticBudgetEngine(
		company=ctx,
		forecasted_sales_units=forecast_current,
		next_month_forecasted_sales_units=forecast_next,
	)
	if engine.validate_logic():
		st.warning(
			"Warning: Opening inventory is higher than planned sales + target ending inventory. Production is zero."
		)

	t1, t2, t3 = st.tabs(["1. Sales & Production", "2. Direct Costs", "3. Income Statement"])

	with t1:
		st.subheader("Revenue Budget")
		rev_df = engine.revenue_budget().rename(
			columns={
				"product": "Product",
				"sales_units": "Sales Volume (Units)",
				"selling_price": "Selling Price (€)",
				"revenue": "Revenue (€)",
			}
		)
		render_dataframe(
			rev_df,
			currency_cols=["Selling Price (€)", "Revenue (€)"],
			unit_cols=["Sales Volume (Units)"],
		)

		st.subheader("Production Budget")
		prod_df = engine.production_budget().rename(
			columns={
				"product": "Product",
				"sales_units": "Sales Volume (Units)",
				"next_month_sales_units": "Next Month Sales (Units)",
				"target_ending_inv_pct": "Target Ending Inv (%)",
				"target_ending_inv_units": "Target Ending Inv (Units)",
				"opening_inv": "Opening Inv (Units)",
				"production_units": "Production (Units)",
			}
		)
		render_dataframe(
			prod_df,
			unit_cols=[
				"Sales Volume (Units)",
				"Next Month Sales (Units)",
				"Target Ending Inv (Units)",
				"Opening Inv (Units)",
				"Production (Units)",
			],
		)

	with t2:
		st.subheader("Direct Materials Budget")
		mat_df = engine.direct_materials_budget().rename(
			columns={
				"material": "Material",
				"qty_required": "Qty Required (Units)",
				"unit_cost": "Unit Cost (€/unit)",
				"cost": "Cost (€)",
			}
		)
		render_dataframe(
			mat_df,
			currency_cols=["Unit Cost (€/unit)", "Cost (€)"],
			unit_cols=["Qty Required (Units)"],
		)

		st.subheader("Direct Labor Budget")
		labor_df = engine.direct_labor_budget().rename(
			columns={
				"product": "Product",
				"production_units": "Production (Units)",
				"labor_minutes": "Labor (Minutes/unit)",
				"labor_hours": "Labor (Hours)",
				"labor_cost": "Labor Cost (€)",
			}
		)
		render_dataframe(
			labor_df,
			currency_cols=["Labor Cost (€)"],
			unit_cols=["Production (Units)"],
		)

	with t3:
		st.subheader("Income Statement")
		inc_df = engine.income_statement().copy()
		render_dataframe(inc_df, currency_cols=["Amount (€)"])


elif page == "3. Variance Analysis":
	if ctx is None:
		st.error("Configuration could not be loaded. Fix it in Parameters Setup.")
		st.stop()
	if selected_month is None or next_month is None:
		st.error("No forecasted months found in config forecasted_sales_units")
		st.stop()

	try:
		forecast_current = ctx.forecasted_sales_units[selected_month]
		forecast_next = ctx.forecasted_sales_units.get(next_month, forecast_current)
	except Exception:
		st.error(f"Missing '{selected_month}' in config forecasted_sales_units")
		st.stop()

	static_engine = StaticBudgetEngine(
		company=ctx,
		forecasted_sales_units=forecast_current,
		next_month_forecasted_sales_units=forecast_next,
	)

	required_cols = [
		"Month",
		"Product",
		"Actual_Sales_Units",
		"Actual_Production_Units",
		"Actual_Material_Cost",
		"Actual_Labor_Cost",
		"Actual_Revenue",
	]

	default_actuals_path = Path("data/actuals.csv")
	known_products = {p.name for p in ctx.products}

	def _load_default_actuals() -> pd.DataFrame:
		st.toast("Loading default actuals from data/actuals.csv", icon="ℹ️")
		st.info("Using default actuals data. Upload your own CSV for custom analysis.")
		st.sidebar.info("💡 Running with sample Apple Inc. data")
		return pd.read_csv(default_actuals_path)

	actuals_df: pd.DataFrame
	if uploaded_file is None:
		actuals_df = _load_default_actuals()
	else:
		try:
			uploaded_bytes = uploaded_file.getvalue()
			actuals_df = pd.read_csv(uploaded_file)
		except Exception:
			msg = "⚠️ Upload could not be read. Falling back to default actuals."
			st.sidebar.error(msg)
			st.error(msg)
			st.toast("Upload failed — using default actuals", icon="❌")
			actuals_df = _load_default_actuals()
		else:
			missing_cols = [c for c in required_cols if c not in actuals_df.columns]
			if missing_cols:
				msg = "⚠️ Invalid CSV Format. Falling back to default actuals."
				st.sidebar.error(msg)
				st.error(msg)
				st.toast("Invalid CSV format — using default actuals", icon="❌")
				actuals_df = _load_default_actuals()
			else:
				unknown_products = sorted(set(actuals_df["Product"].astype(str)) - known_products)
				if unknown_products:
					msg = (
						"⚠️ Product mismatch. Falling back to default actuals. "
						f"Unknown products: {unknown_products}"
					)
					st.sidebar.error(msg)
					st.error(msg)
					st.toast("Product mismatch — using default actuals", icon="❌")
					actuals_df = _load_default_actuals()
				else:
					# Persist valid upload locally for future runs.
					default_actuals_path.parent.mkdir(parents=True, exist_ok=True)
					default_actuals_path.write_bytes(uploaded_bytes)
					st.sidebar.success("✅ Using custom actuals data")
					st.toast("Custom actuals saved to data/actuals.csv", icon="✅")

	# Filter actuals to selected month and block variance analysis if missing.
	if "Month" not in actuals_df.columns:
		st.warning(f"No actual data found for {selected_month}. Variance analysis unavailable.")
		st.stop()

	actuals_df = actuals_df.loc[actuals_df["Month"].astype(str) == str(selected_month)].copy()
	if actuals_df.empty:
		st.warning(f"No actual data found for {selected_month}. Variance analysis unavailable.")
		st.stop()

	variance_engine = VarianceEngine(static_engine, actuals_df, month=None)
	variance_df = variance_engine.variance_report()

	st.subheader("Variance Report")
	render_dataframe(
		variance_df,
		currency_cols=[
			"Static Budget (€)",
			"Volume Variance (€)",
			"Flexible Budget (€)",
			"Spending Variance (€)",
			"Actuals (€)",
		],
		pct_cols=["Volume Var %", "Spending Var %"],
		fav_unfav_cols=["Volume Status", "Spending Status"],
		units_index_rows=["Sales Volume (Units)"],
	)

	st.subheader("Operating Income Waterfall (€)")
	tab_bridge, tab_detail = st.tabs(["📊 Operating Income Bridge", "🔍 Detailed Category Breakdown"])

	with tab_bridge:
		st.subheader("Operating Income Waterfall (€)")
		mode = st.radio(
			"Waterfall mode",
			options=["Show Absolute Values", "Show Differences Only"],
			horizontal=True,
		)
		fig = draw_waterfall_chart(variance_df, differences_only=(mode == "Show Differences Only"))
		st.plotly_chart(fig, use_container_width=True)

	with tab_detail:
		st.subheader("Variance Analysis by Line Item")
		fig_detail = draw_detailed_variance_bar_chart(variance_df)
		st.plotly_chart(fig_detail, use_container_width=True)
		st.caption(
			"This chart enables Attention-Directing by isolating Volume and Spending variances for each category, "
			"allowing for a precise comparison of impacts that are otherwise hidden by the scale of total budget figures."
		)


elif page == "4. Risk Simulation":
	if ctx is None:
		st.error("Configuration could not be loaded. Fix it in Parameters Setup.")
		st.stop()
	if selected_month is None:
		st.error("No forecasted months found in config forecasted_sales_units")
		st.stop()

	try:
		baseline = ctx.forecasted_sales_units[selected_month]
	except Exception:
		st.error(f"Missing '{selected_month}' in config forecasted_sales_units")
		st.stop()

	num_iterations = st.sidebar.slider("num_iterations", min_value=1000, max_value=50000, value=10000, step=1000)
	volume_volatility = st.sidebar.slider(
		"volume_volatility", min_value=0.0, max_value=0.50, value=0.10, step=0.01
	)
	price_volatility = st.sidebar.slider(
		"price_volatility", min_value=0.0, max_value=0.50, value=0.05, step=0.01
	)

	run = st.sidebar.button("Run Simulation")
	simulator = MonteCarloSimulator(ctx, baseline)

	if run:
		st.toast("Running Monte Carlo simulation…", icon="⏳")
		out = simulator.run_simulation(
			num_iterations=int(num_iterations),
			volume_volatility=float(volume_volatility),
			price_volatility=float(price_volatility),
		)
		st.toast("Simulation completed", icon="✅")
		metrics = out["metrics"]
		profits = out["profits"]

		st.subheader("Simulation Metrics")
		c1, c2, c3, c4 = st.columns(4)
		c1.metric("Expected Mean Profit (€)", _fmt_currency(metrics["expected_mean_profit"]))
		c2.metric("5th Percentile (Worst) (€)", _fmt_currency(metrics["p05_worst_case_profit"]))
		c3.metric("95th Percentile (Best) (€)", _fmt_currency(metrics["p95_best_case_profit"]))
		c4.metric("Probability Break-Even (%)", f"{(metrics['probability_of_break_even'] * 100):.2f}%")

		st.subheader("Profit Distribution (€)")
		draw_profit_distribution(profits, metrics)