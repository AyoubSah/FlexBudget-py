from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.models.company import CompanyContext
from core.models.product import Product


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    return json.loads(text)


def _init_session_state(config_path: str) -> None:
	path = Path(config_path)

	if st.session_state.get("_config_path") != str(path):
		for key in (
			"config_data",
			"company_ctx",
			"raw_materials_table",
			"products_table",
			"labor_rate_per_hour_input",
			"fixed_overhead_input",
			"raw_materials_editor",
			"products_editor",
		):
			st.session_state.pop(key, None)
		st.session_state["_config_path"] = str(path)

	if "config_data" not in st.session_state:
		try:
			st.session_state["config_data"] = _read_json(path)
		except json.JSONDecodeError as exc:
			st.error(f"Invalid JSON in {path}: {exc}")
			st.stop()

	if "company_ctx" not in st.session_state:
		st.session_state["company_ctx"] = CompanyContext(path)

	if "raw_materials_table" not in st.session_state:
		ctx: CompanyContext = st.session_state["company_ctx"]
		rows = [
			{"material": name, "unit_cost": float(info.get("unit_cost", 0.0))}
			for name, info in ctx.raw_materials.items()
		]
		st.session_state["raw_materials_table"] = pd.DataFrame(rows)

	if "products_table" not in st.session_state:
		ctx: CompanyContext = st.session_state["company_ctx"]
		rows: list[dict[str, Any]] = []
		for i, p in enumerate(ctx.products):
			rows.append(
				{
					"row_id": i,
					"name": p.name,
					"selling_price": p.selling_price,
					"target_ending_inv_pct": p.target_ending_inv_pct,
					"labor_minutes": p.labor_minutes,
					"opening_inv": p.opening_inv,
				}
			)
		st.session_state["products_table"] = pd.DataFrame(rows)


def render_company_config_editor(config_path: str = "data/config.json") -> None:
	"""Streamlit UI component to edit CompanyContext interactively."""
	_init_session_state(config_path)
	ctx: CompanyContext = st.session_state["company_ctx"]

	st.subheader("Global Parameters")
	ctx.labor_rate_per_hour = float(
		st.number_input(
			"Labor rate (€/hour)",
			value=float(ctx.labor_rate_per_hour),
			min_value=0.0,
			step=0.5,
			key="labor_rate_per_hour_input",
		)
	)
	ctx.fixed_overhead = float(
		st.number_input(
			"Fixed overhead (€)",
			value=float(ctx.fixed_overhead),
			min_value=0.0,
			step=100.0,
			key="fixed_overhead_input",
		)
	)

	st.subheader("Raw Materials")
	raw_edited = st.data_editor(
		st.session_state["raw_materials_table"],
		num_rows="dynamic",
		use_container_width=True,
		column_config={
			"material": st.column_config.TextColumn("Material"),
			"unit_cost": st.column_config.NumberColumn("Unit cost (€/unit)", format="%.4f"),
		},
		key="raw_materials_editor",
	)
	st.session_state["raw_materials_table"] = raw_edited

	st.subheader("Products")
	products_edited = st.data_editor(
		st.session_state["products_table"],
		num_rows="dynamic",
		disabled=["row_id"],
		use_container_width=True,
		column_config={
			"row_id": st.column_config.NumberColumn("Row"),
			"name": st.column_config.TextColumn("Product"),
			"selling_price": st.column_config.NumberColumn("Selling price (€/unit)", format="%.2f"),
			"target_ending_inv_pct": st.column_config.NumberColumn("Target ending inv (%)", format="%.2f"),
			"labor_minutes": st.column_config.NumberColumn("Labor (min/unit)", format="%.2f"),
			"opening_inv": st.column_config.NumberColumn("Opening inv (units)", format="%d"),
		},
		key="products_editor",
	)
	st.session_state["products_table"] = products_edited

	if st.button("Save Changes", type="primary"):
		st.toast("Saving configuration…", icon="⏳")
		new_raw_materials: dict[str, dict[str, float]] = {}
		for _, row in raw_edited.iterrows():
			material = str(row.get("material", "")).strip()
			if not material:
				continue
			unit_cost_raw = row.get("unit_cost", 0.0)
			unit_cost = float(unit_cost_raw) if unit_cost_raw is not None else 0.0
			new_raw_materials[material] = {"unit_cost": unit_cost}
		ctx.raw_materials = new_raw_materials

		existing_boms_by_row_id = {i: p.bill_of_materials for i, p in enumerate(ctx.products)}
		new_products: list[Product] = []
		for _, row in products_edited.iterrows():
			name = str(row.get("name", "")).strip()
			if not name:
				continue

			row_id = row.get("row_id")
			bill_of_materials: dict[str, float] = {}
			try:
				if pd.notna(row_id):
					bill_of_materials = dict(existing_boms_by_row_id.get(int(row_id), {}))
			except Exception:
				bill_of_materials = {}

			new_products.append(
				Product(
					name=name,
					selling_price=float(row.get("selling_price", 0.0) or 0.0),
					target_ending_inv_pct=float(row.get("target_ending_inv_pct", 0.0) or 0.0),
					labor_minutes=float(row.get("labor_minutes", 0.0) or 0.0),
					opening_inv=int(row.get("opening_inv", 0) or 0),
					bill_of_materials=bill_of_materials,
				)
			)
		ctx.products = new_products

		ctx.save_to_json()
		st.toast("Configuration saved", icon="✅")

		# Refresh JSON snapshot in session_state for other app components
		try:
			st.session_state["config_data"] = _read_json(Path(config_path))
		except json.JSONDecodeError as exc:
			st.toast("Save failed: config JSON is invalid", icon="❌")
			st.error(f"Config saved but now JSON is invalid in {config_path}: {exc}")
			st.stop()
		st.rerun()



def render_config_editor(config_path: str = "data/config.json") -> None:
	"""Backward-compatible alias for app code that imports render_config_editor."""
	render_company_config_editor(config_path=config_path)
