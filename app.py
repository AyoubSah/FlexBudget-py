import pandas as pd
import streamlit as st

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

	actuals_df = pd.read_csv("data/actuals.csv")
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
		out = simulator.run_simulation(
			num_iterations=int(num_iterations),
			volume_volatility=float(volume_volatility),
			price_volatility=float(price_volatility),
		)
		metrics = out["metrics"]
		profits = out["profits"]

		st.subheader("Simulation Metrics")
		c1, c2, c3, c4 = st.columns(4)
		c1.metric("Expected Mean Profit", f"{metrics['expected_mean_profit']:,.0f}")
		c2.metric("5th Percentile (Worst)", f"{metrics['p05_worst_case_profit']:,.0f}")
		c3.metric("95th Percentile (Best)", f"{metrics['p95_best_case_profit']:,.0f}")
		c4.metric("Probability Break-Even", f"{metrics['probability_of_break_even']:.1%}")

		st.subheader("Profit Distribution")
		fig = draw_profit_distribution(profits, metrics)
		st.plotly_chart(fig, use_container_width=True)