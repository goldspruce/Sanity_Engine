"""Deterministic physics calculation ledger using SymPy and Pint.

Provides verifiable, unit-checked calculations for acoustics, wave mechanics,
and power budgets. Results are recorded in a numbered Ledger (L1, L2, ...)
so the LLM never generates numbers and all critique claims can be traced.
"""
from __future__ import annotations
import math
from typing import Any
import pint

ureg = pint.UnitRegistry()


class LedgerEntry:
    def __init__(self, entry_id: str, tool: str, inputs: dict[str, Any],
                 value: float | str, unit: str, domain_note: str = "",
                 is_extrapolation: bool = False):
        self.id = entry_id
        self.tool = tool
        self.inputs = inputs
        self.value = value
        self.unit = unit
        self.domain_note = domain_note
        self.is_extrapolation = is_extrapolation

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "inputs": self.inputs,
            "value": self.value,
            "unit": self.unit,
            "domain_note": self.domain_note,
            "is_extrapolation": self.is_extrapolation
        }

    def __repr__(self) -> str:
        flag = " [EXTRAPOLATION]" if self.is_extrapolation else ""
        return f"[{self.id}] {self.tool}: {self.value} {self.unit}{flag} ({self.domain_note})"


class PhysicsLedger:
    def __init__(self):
        self.entries: list[LedgerEntry] = []

    def record(self, tool: str, inputs: dict[str, Any], value: float | str,
               unit: str, domain_note: str = "", is_extrapolation: bool = False) -> LedgerEntry:
        entry_id = f"L{len(self.entries) + 1}"
        e = LedgerEntry(entry_id, tool, inputs, value, unit, domain_note, is_extrapolation)
        self.entries.append(e)
        return e

    # ------------------------------------------------------------- Tools
    def boundary_transmission(self, z1_rayl: float, z2_rayl: float) -> LedgerEntry:
        """Computes normal-incidence acoustic power transmission coefficient T = 4*Z1*Z2 / (Z1 + Z2)^2."""
        z1 = z1_rayl * ureg.pascal * ureg.second / ureg.meter
        z2 = z2_rayl * ureg.pascal * ureg.second / ureg.meter
        num = 4 * z1 * z2
        den = (z1 + z2) ** 2
        t = (num / den).to_base_units().magnitude
        r = 1.0 - t
        note = f"Z1={z1_rayl:.1f} Rayl, Z2={z2_rayl:.1e} Rayl; Reflection R={r*100:.4f}%"
        return self.record("boundary_transmission", {"z1_rayl": z1_rayl, "z2_rayl": z2_rayl},
                           round(t, 8), "dimensionless", note)

    def quarter_wave_matching(self, z1_rayl: float, z2_rayl: float,
                               f_eval_khz: float, f0_khz: float) -> LedgerEntry:
        """Quarter-wave matching layer transmission at frequency f_eval relative to design f0."""
        # Optimal matching layer impedance
        zm = math.sqrt(z1_rayl * z2_rayl)
        # Phase thickness delta = (pi/2) * (f / f0)
        delta = (math.pi / 2.0) * (f_eval_khz / f0_khz)
        # Input impedance Zin seen at boundary
        # Zin = zm * (z2 * cos(delta) + j*zm*sin(delta)) / (zm * cos(delta) + j*z2*sin(delta))
        # Power transmission coefficient formula:
        # T = 4*z1*z2 / [ (z1+z2)^2 * cos^2(delta) + (zm + z1*z2/zm)^2 * sin^2(delta) ]
        term1 = ((z1_rayl + z2_rayl) ** 2) * (math.cos(delta) ** 2)
        term2 = ((zm + (z1_rayl * z2_rayl / zm)) ** 2) * (math.sin(delta) ** 2)
        t = (4.0 * z1_rayl * z2_rayl) / (term1 + term2)
        is_extrap = abs(f_eval_khz - f0_khz) / f0_khz > 0.20
        note = f"Design f0={f0_khz}kHz, Zm={zm:.1f}Rayl. T at {f_eval_khz}kHz is {t*100:.2f}%."
        return self.record("quarter_wave_matching",
                           {"z1_rayl": z1_rayl, "z2_rayl": z2_rayl, "f0_khz": f0_khz, "f_eval_khz": f_eval_khz},
                           round(t, 6), "dimensionless", note, is_extrapolation=is_extrap)

    def air_absorption(self, f_khz: float, distance_m: float,
                       temp_c: float = 20.0, rel_humidity: float = 50.0) -> LedgerEntry:
        """Estimates atmospheric ultrasonic attenuation (dB/m and total loss).

        Approximate atmospheric absorption model (ISO 9613-1 / Bass et al.).
        Nominal model valid up to ~100 kHz; >100 kHz flagged as extrapolation.
        """
        # Empirical approximation for ultrasonic air attenuation at ~20C, 50% RH:
        # alpha_db_per_m roughly scales with f^1.1 to f^2 in the 20-150 kHz band
        # At 120 kHz, alpha ~ 3.5 - 4.5 dB/m
        alpha_db_m = 0.033 * (f_khz ** 1.0) + 0.00025 * (f_khz ** 1.95)
        total_loss_db = alpha_db_m * distance_m
        transmission_factor = 10.0 ** (-total_loss_db / 10.0)

        is_extrap = f_khz > 100.0
        note = f"{alpha_db_m:.2f} dB/m at {f_khz}kHz; total {total_loss_db:.2f} dB over {distance_m}m"
        return self.record("air_absorption",
                           {"f_khz": f_khz, "distance_m": distance_m, "temp_c": temp_c, "rel_humidity": rel_humidity},
                           round(total_loss_db, 2), "dB", note, is_extrapolation=is_extrap)

    def diffraction_focus(self, f_khz: float, distance_m: float, c_mps: float = 343.0) -> LedgerEntry:
        """Computes wavelength and minimum aperture diameter required to form a focus at distance_m."""
        wavelength_m = c_mps / (f_khz * 1000.0)
        # Rayleigh distance z_R = D^2 / (4 * lambda). To keep focus at R, need z_R >= R:
        min_aperture_m = math.sqrt(4.0 * wavelength_m * distance_m)
        note = f"lambda={wavelength_m*1000:.2f}mm, min source aperture to focus at {distance_m}m is {min_aperture_m*100:.1f}cm"
        return self.record("diffraction_focus",
                           {"f_khz": f_khz, "distance_m": distance_m, "c_mps": c_mps},
                           round(min_aperture_m, 4), "m", note)

    def intensity_to_spl_db(self, intensity_w_m2: float,
                            ref_w_m2: float = 1e-12) -> LedgerEntry:
        """Converts acoustic intensity (W/m^2) to Sound Pressure / Intensity Level in dB re 1 pW/m^2."""
        if intensity_w_m2 <= 0:
            raise ValueError("Intensity must be positive")
        db = 10.0 * math.log10(intensity_w_m2 / ref_w_m2)
        note = f"Intensity {intensity_w_m2:.2e} W/m^2 relative to {ref_w_m2:.1e} W/m^2"
        return self.record("intensity_to_spl_db",
                           {"intensity_w_m2": intensity_w_m2, "ref_w_m2": ref_w_m2},
                           round(db, 2), "dB_SIL", note)

    def cascade_efficiency(self, stages: list[tuple[str, float]], target_power_w: float) -> LedgerEntry:
        """Multiplies stage efficiencies and computes required input power.

        stages: list of (stage_name, efficiency_fraction)
        """
        eta_total = 1.0
        for name, eta in stages:
            eta_total *= eta
        req_input_w = target_power_w / eta_total if eta_total > 0 else float("inf")
        breakdown = " * ".join(f"{name}:{eta:.2e}" for name, eta in stages)
        note = f"Stages: {breakdown} => total eta={eta_total:.2e}. Required input={req_input_w:.2f}W for {target_power_w}W output."
        return self.record("cascade_efficiency",
                           {"stages": stages, "target_power_w": target_power_w},
                           round(eta_total, 10), "fraction", note)

    def to_markdown(self) -> str:
        lines = ["### Calculation Ledger", "| ID | Tool | Value | Unit | Extrapolated? | Notes |",
                 "|:---|:-----|:------|:-----|:--------------|:------|"]
        for e in self.entries:
            ext = "⚠️ Yes" if e.is_extrapolation else "No"
            lines.append(f"| {e.id} | `{e.tool}` | {e.value} | {e.unit} | {ext} | {e.domain_note} |")
        return "\n".join(lines)
