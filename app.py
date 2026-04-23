import pandas as pd
import streamlit as st

from pathlib import Path

from core.engines.static_budget import StaticBudgetEngine
from core.engines.stochastic_mc import MonteCarloSimulator
from core.engines.variance_calc import VarianceEngine
from core.models.company import CompanyContext
from ui.config_editor import render_config_editor
from ui.dashboards import draw_profit_distribution, draw_waterfall_chart


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


if page == "1. Parameters Setup":
	render_config_editor()


elif page == "2. Master Budget":
	ctx = CompanyContext()
	try:
		fcst_oct = ctx.forecasted_sales_units["Oct_2026"]
		fcst_nov = ctx.forecasted_sales_units["Nov_2026"]
	except Exception:
		st.error("Missing 'Oct_2026' or 'Nov_2026' in config forecasted_sales_units")
		st.stop()

	engine = StaticBudgetEngine(
		company=ctx,
		forecasted_sales_units=fcst_oct,
		next_month_forecasted_sales_units=fcst_nov,
	)

	st.subheader("Revenue Budget")
	st.dataframe(engine.revenue_budget(), use_container_width=True)

	st.subheader("Production Budget")
	st.dataframe(engine.production_budget(), use_container_width=True)

	st.subheader("Direct Materials Budget")
	st.dataframe(engine.direct_materials_budget(), use_container_width=True)

	st.subheader("Direct Labor Budget")
	st.dataframe(engine.direct_labor_budget(), use_container_width=True)

	st.subheader("Income Statement")
	st.dataframe(engine.income_statement(), use_container_width=True)


elif page == "3. Variance Analysis":
	ctx = CompanyContext()
	try:
		fcst_oct = ctx.forecasted_sales_units["Oct_2026"]
		fcst_nov = ctx.forecasted_sales_units.get("Nov_2026", {})
	except Exception:
		st.error("Missing 'Oct_2026' in config forecasted_sales_units")
		st.stop()

	static_engine = StaticBudgetEngine(
		company=ctx,
		forecasted_sales_units=fcst_oct,
		next_month_forecasted_sales_units=fcst_nov,
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

	variance_engine = VarianceEngine(static_engine, actuals_df, month="Oct_2026")
	variance_df = variance_engine.variance_report()

	st.subheader("Variance Report")
	st.dataframe(variance_df, use_container_width=True)

	st.subheader("Operating Income Waterfall")
	fig = draw_waterfall_chart(variance_df)
	st.plotly_chart(fig, use_container_width=True)


elif page == "4. Risk Simulation":
	ctx = CompanyContext()
	try:
		baseline = ctx.forecasted_sales_units["Oct_2026"]
	except Exception:
		st.error("Missing 'Oct_2026' in config forecasted_sales_units")
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
		c1.metric("Expected Mean Profit", f"{metrics['expected_mean_profit']:,.0f}")
		c2.metric("5th Percentile (Worst)", f"{metrics['p05_worst_case_profit']:,.0f}")
		c3.metric("95th Percentile (Best)", f"{metrics['p95_best_case_profit']:,.0f}")
		c4.metric("Probability Break-Even", f"{metrics['probability_of_break_even']:.1%}")

		st.subheader("Profit Distribution")
		draw_profit_distribution(profits, metrics)