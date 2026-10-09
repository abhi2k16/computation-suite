# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
units.py -- unit LABELS (no conversion engine).

The solver is unit-agnostic: it multiplies the numbers you give it. A `UnitSystem` only
records what those numbers mean so results, plots and tables can say so, and so that two
objects built in different systems can be compared before they are combined::

    system.units = units.SI                       # or units.MM_N_TONNE, or UnitSystem(...)
    U = system.solve_static()
    U.units                                       # "m"
    U.plot("uy")                                  # colour bar reads "uy [m]"

Nothing is converted. If you assemble a stiffness in MPa and mm but apply a load in N and m,
the labels will not stop you; `check_consistent` only catches two *labelled* objects that
disagree. Consistent sets: SI (m, N, kg, s -> Pa) and MM_N_TONNE (mm, N, tonne, s -> MPa).
"""
from __future__ import annotations
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"

from dataclasses import dataclass

__all__ = ["UnitSystem", "SI", "MM_N_TONNE", "PRESETS", "resolve", "check_consistent"]


@dataclass(frozen=True)
class UnitSystem:
    """A consistent set of unit labels.

    Parameters
    ----------
    length, force, mass, time : str
        Base unit labels, e.g. ``"m", "N", "kg", "s"``.
    stress : str, optional
        Label for force/length^2; derived when omitted (``"Pa"`` for SI, otherwise
        ``"N/mm^2"``-style text built from ``force`` and ``length``).
    """
    length: str = "m"
    force: str = "N"
    mass: str = "kg"
    time: str = "s"
    stress: str = ""

    def __post_init__(self):
        if not self.stress:
            derived = "Pa" if (self.force, self.length) == ("N", "m") else \
                      "MPa" if (self.force, self.length) == ("N", "mm") else f"{self.force}/{self.length}^2"
            object.__setattr__(self, "stress", derived)

    @property
    def density(self):
        return f"{self.mass}/{self.length}^3"

    @property
    def frequency(self):
        return "Hz" if self.time == "s" else f"1/{self.time}"

    def unit_of(self, quantity):
        """Label for a named quantity: length, displacement, force, mass, time, stress,
        modulus, density, frequency, rotation (``"rad"``)."""
        table = {"length": self.length, "displacement": self.length, "force": self.force,
                 "mass": self.mass, "time": self.time, "stress": self.stress,
                 "modulus": self.stress, "density": self.density, "frequency": self.frequency,
                 "rotation": "rad"}
        try:
            return table[quantity]
        except KeyError:
            raise ValueError(f"unknown quantity {quantity!r}; choose from {sorted(table)}") from None

    def __str__(self):
        return f"({self.length}, {self.force}, {self.mass}, {self.time}) -> {self.stress}"


SI = UnitSystem("m", "N", "kg", "s", "Pa")
MM_N_TONNE = UnitSystem("mm", "N", "tonne", "s", "MPa")
PRESETS = {"SI": SI, "MM_N_TONNE": MM_N_TONNE}


def resolve(units):
    """None, a UnitSystem, or a preset name ('SI', 'MM_N_TONNE') -> UnitSystem or None."""
    if units is None or isinstance(units, UnitSystem):
        return units
    if isinstance(units, str) and units.upper() in PRESETS:
        return PRESETS[units.upper()]
    raise ValueError(f"units must be None, a UnitSystem or one of {sorted(PRESETS)}; got {units!r}")


def check_consistent(*labelled, who="check_consistent"):
    """Raise ValueError if two objects that carry a ``units`` attribute disagree.
    Objects with ``units`` of None are skipped (unlabelled)."""
    seen = [(type(o).__name__, o.units) for o in labelled if getattr(o, "units", None) is not None]
    for name, u in seen[1:]:
        if u != seen[0][1]:
            raise ValueError(f"{who}: unit systems differ -- {seen[0][0]} uses {seen[0][1]} but "
                             f"{name} uses {u}. Convert the numbers yourself; labels are not converted.")
