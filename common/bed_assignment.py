"""`BedAssignment` stub (C7).

Synthetic admissions service mapping `patient_id -> (unit, bed)`. A `Unit` holds <= 25 beds;
each patient has a unique current bed and a bed holds <= 1 patient. When a unit fills, assignment
**overflows to the next unit** (auto-create). `clear_unit` / `clear_all` reset for tests.

In-memory only — the real form is an ADT/admissions feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BEDS_PER_UNIT = 25


@dataclass
class BedAssignmentStub:
    beds_per_unit: int = BEDS_PER_UNIT
    # patient_id -> (unit, bed)
    _assignments: dict[str, tuple[str, str]] = field(default_factory=dict)
    # ordered list of occupied bed keys per unit, to find the next free bed
    _unit_occupancy: dict[str, set[str]] = field(default_factory=dict)

    def assign(self, patient_id: str) -> tuple[str, str]:
        """Return the patient's current (unit, bed), assigning one on first call."""
        if patient_id in self._assignments:
            return self._assignments[patient_id]

        unit_idx = 1
        while True:
            unit = f"Unit{unit_idx}"
            occupied = self._unit_occupancy.setdefault(unit, set())
            if len(occupied) < self.beds_per_unit:
                bed_idx = self._first_free_bed(unit, occupied)
                bed = f"{unit}-Bed{bed_idx:02d}"
                occupied.add(bed)
                self._assignments[patient_id] = (unit, bed)
                return unit, bed
            unit_idx += 1  # overflow to the next unit

    def _first_free_bed(self, unit: str, occupied: set[str]) -> int:
        for i in range(1, self.beds_per_unit + 1):
            if f"{unit}-Bed{i:02d}" not in occupied:
                return i
        raise RuntimeError(f"{unit} unexpectedly full")  # pragma: no cover

    def current(self, patient_id: str) -> tuple[str, str] | None:
        return self._assignments.get(patient_id)

    def clear_unit(self, unit: str) -> None:
        occupied = self._unit_occupancy.pop(unit, set())
        for pid in [p for p, (u, _) in self._assignments.items() if u == unit]:
            del self._assignments[pid]
        occupied.clear()

    def clear_all(self) -> None:
        self._assignments.clear()
        self._unit_occupancy.clear()
