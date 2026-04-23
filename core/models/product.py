from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Product:
	name: str
	selling_price: float
	target_ending_inv_pct: float
	labor_minutes: float
	opening_inv: int
	# Bill of materials (a.k.a. sigma): component name -> quantity consumed per finished unit
	bill_of_materials: dict[str, float] = field(default_factory=dict)

	def __post_init__(self) -> None:
		self.name = str(self.name)
		self.selling_price = float(self.selling_price)
		self.target_ending_inv_pct = float(self.target_ending_inv_pct)
		self.labor_minutes = float(self.labor_minutes)
		self.opening_inv = int(self.opening_inv)

		# Normalize bill_of_materials values to floats (JSON may provide ints).
		self.bill_of_materials = {
			str(component): float(qty) for component, qty in (self.bill_of_materials or {}).items()
		}

	@staticmethod
	def from_dict(data: dict[str, Any]) -> "Product":
		bill_of_materials = (
			data.get("bill_of_materials")
			if data.get("bill_of_materials") is not None
			else data.get("sigma")
		)
		if bill_of_materials is None:
			bill_of_materials = {}

		return Product(
			name=data["name"],
			selling_price=data["selling_price"],
			target_ending_inv_pct=data["target_ending_inv_pct"],
			labor_minutes=data["labor_minutes"],
			opening_inv=data["opening_inv"],
			bill_of_materials=dict(bill_of_materials),
		)

	def to_dict(self) -> dict[str, Any]:
		return {
			"name": self.name,
			"selling_price": self.selling_price,
			"target_ending_inv_pct": self.target_ending_inv_pct,
			"labor_minutes": self.labor_minutes,
			"opening_inv": self.opening_inv,
			"bill_of_materials": dict(self.bill_of_materials),
		}
