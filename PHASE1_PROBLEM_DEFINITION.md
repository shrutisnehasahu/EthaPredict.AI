# Phase 1 — Problem Definition

## Project
**EthaPredict AI: Machine Learning-Based Ethanol Fuel Consumption, Range and Refueling-Time Prediction System**

## Problem Statement
Can machine learning predict the fuel consumption, remaining driving range, remaining
driving time, and refueling point of an ethanol-blended vehicle using vehicle telemetry,
fuel characteristics, and driving-condition data?

## Target Variable
`Fuel_Consumed_L` — liters of fuel consumed over a trip/segment.

## Derived Outputs (built on top of the model's prediction)
1. **Fuel efficiency** (km/L) — derived from predicted consumption and distance
2. **Remaining driving range** (km) — fuel remaining × predicted efficiency
3. **Remaining driving time** (hr) — remaining range / average speed
4. **Refueling recommendation** — distance-to-empty threshold alert

## Inputs Used by the Model
Vehicle data (speed, RPM, temperature, load, acceleration), fuel data (fuel remaining,
ethanol blend %), driving data (traffic level, road type, AC usage, driving time,
distance travelled).

## Success Criteria
- R² ≥ 0.85 on held-out test data for the best model
- A working prediction engine that converts model output into range/time/refuel alerts
- A dashboard that presents these predictions in a usable format
