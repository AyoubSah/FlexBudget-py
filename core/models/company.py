from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .product import Product

class CompanyContext:
    def __init__(self, json_path: str | Path = "data/config.json") -> None:
        self.json_path = Path(json_path)

        data = self._load_json(self.json_path)

        # Assuming these are stored at the root of the JSON or inside a global_parameters dict
        global_params = data.get("global_parameters", data)
        self.labor_rate_per_hour: float = float(global_params.get("labor_rate_per_hour", 0.0))
        self.fixed_overhead: float = float(global_params.get("fixed_overhead", 0.0))

        raw_materials_data = data.get("raw_materials", {}) or {}
        self.raw_materials: dict[str, dict[str, float]] = {}
        for name, value in raw_materials_data.items():
            if isinstance(value, dict):
                unit_cost = float(value.get("unit_cost", 0.0))
            else:
                unit_cost = float(value)
            self.raw_materials[str(name)] = {"unit_cost": unit_cost}

        # Handle products whether they are in a list or a dictionary
        products_data = data.get("products",[])
        if isinstance(products_data, dict):
            # If JSON stored them as a dict, inject the name into the dict before parsing
            self.products = []
            for name, p_data in products_data.items():
                p_data["name"] = name
                self.products.append(Product.from_dict(p_data))
        else:
            self.products = [Product.from_dict(p) for p in products_data]

        # EXPLICITLY CAPTURE FORECASTED SALES
        self.forecasted_sales_units: dict[str, dict[str, int]] = data.get("forecasted_sales_units", {})

        known = {"labor_rate_per_hour", "fixed_overhead", "global_parameters", "raw_materials", "products", "forecasted_sales_units"}
        self._extra: dict[str, Any] = {k: v for k, v in data.items() if k not in known}

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}") from exc

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Top-level JSON must be an object in {path}")
        return data

    def save_to_json(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            **self._extra,
            "global_parameters": {
                "labor_rate_per_hour": float(self.labor_rate_per_hour),
                "fixed_overhead": float(self.fixed_overhead)
            },
            "raw_materials": {
                name: {"unit_cost": float(value.get("unit_cost", 0.0))}
                for name, value in self.raw_materials.items()
            },
            "products":[p.to_dict() for p in self.products],
            "forecasted_sales_units": self.forecasted_sales_units # Save it back
        }

        self.json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )