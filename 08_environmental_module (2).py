"""
Environmental Module — CO2 Emission Estimation
====================================================
Compares estimated tank-to-wheel CO2 emissions across ethanol blends.

IMPORTANT CAVEAT (per roadmap): higher ethanol content does NOT automatically
mean proportionally lower total emissions. Ethanol has lower carbon content
per liter of fuel, but blended-fuel vehicles also consume MORE liters per km
(see the ethanol penalty in the dataset generator), so the net effect must be
computed, not assumed. This module computes it explicitly rather than assuming
a direction.

Emission factors used (approximate, tank-to-wheel, combustion only):
    Gasoline (E0):  2.31 kg CO2 / L
    Ethanol (E100): 1.51 kg CO2 / L   (lower carbon content per liter)
These are simplified, widely-cited approximations for illustration; a rigorous
lifecycle analysis (feedstock, production, land use) is out of scope here and
should be cited from a source like NITI Aayog / EPA if used in a real report.
"""

import pandas as pd

GASOLINE_CO2_KG_PER_L = 2.31
ETHANOL_CO2_KG_PER_L = 1.51


def blend_co2_factor(ethanol_pct: float) -> float:
    """kg CO2 per liter of blended fuel, linear interpolation by blend %."""
    e_frac = ethanol_pct / 100
    return (1 - e_frac) * GASOLINE_CO2_KG_PER_L + e_frac * ETHANOL_CO2_KG_PER_L


def estimate_trip_emissions(fuel_consumed_L: float, ethanol_pct: float) -> float:
    """Total kg CO2 for a trip given predicted fuel consumption and blend %."""
    return round(fuel_consumed_L * blend_co2_factor(ethanol_pct), 3)


def compare_blends_for_distance(distance_km: float, base_consumption_rate_Lpkm: float = 0.06):
    """
    Compares total CO2 for the SAME distance across blends, using the same
    ethanol consumption penalty built into the dataset generator, so higher
    blends correctly consume more liters per km before we apply the lower
    per-liter carbon factor.
    """
    ethanol_penalty = {0: 0.0, 10: 0.03, 20: 0.06, 85: 0.24}
    rows = []
    for blend, penalty in ethanol_penalty.items():
        rate = base_consumption_rate_Lpkm + penalty
        fuel_used = rate * distance_km
        co2_kg = estimate_trip_emissions(fuel_used, blend)
        rows.append({
            "Ethanol_Blend": f"E{blend}",
            "Fuel_Used_L": round(fuel_used, 2),
            "CO2_Factor_kg_per_L": round(blend_co2_factor(blend), 3),
            "Total_CO2_kg": co2_kg,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    comparison = compare_blends_for_distance(distance_km=100)
    print("CO2 comparison for a 100 km trip, same driving conditions:\n")
    print(comparison.to_string(index=False))
    print(
        "\nNote: E85 uses far more fuel per km (per the dataset's ethanol penalty), "
        "which can outweigh its lower per-liter carbon content — illustrating why "
        "the net effect must be calculated rather than assumed."
    )
