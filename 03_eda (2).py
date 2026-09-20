"""
Phase 5 — Exploratory Data Analysis
=======================================
Generates the insight plots called for in the roadmap:
  - Ethanol % vs fuel consumption
  - Speed vs mileage (efficiency)
  - Traffic vs fuel consumption
  - AC vs fuel consumption
  - Load vs fuel consumption
  - RPM vs fuel consumption
  - Correlation heatmap (bonus, ties everything together)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "/home/claude/ethapredict/data/ethapredict_clean.csv"
FIG_DIR = "/home/claude/ethapredict/figures"

df = pd.read_csv(DATA_PATH)

# Derived helper column purely for EDA readability (not saved to clean CSV)
df["Efficiency_kmpl"] = (df["Distance_Travelled_km"] / df["Fuel_Consumed_L"]).replace([float("inf")], None)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {name}.png")

# 1. Ethanol % vs Fuel Consumption
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Ethanol_Blend", y="Fuel_Consumed_L", hue="Ethanol_Blend",
            palette="viridis", legend=False)
plt.title("Ethanol Blend % vs Fuel Consumption")
plt.xlabel("Ethanol Blend (%)")
plt.ylabel("Fuel Consumed (L)")
savefig("01_ethanol_vs_consumption")

# 2. Speed vs Efficiency
plt.figure(figsize=(7, 5))
sample = df.sample(min(3000, len(df)), random_state=1)
sns.scatterplot(data=sample, x="Speed_kmph", y="Efficiency_kmpl", alpha=0.35, s=15)
sns.regplot(data=sample, x="Speed_kmph", y="Efficiency_kmpl", scatter=False, color="red", order=2)
plt.title("Speed vs Fuel Efficiency")
plt.xlabel("Speed (km/h)")
plt.ylabel("Efficiency (km/L)")
plt.ylim(0, sample["Efficiency_kmpl"].quantile(0.98))
savefig("02_speed_vs_efficiency")

# 3. Traffic vs Fuel Consumption
plt.figure(figsize=(7, 5))
traffic_labels = {0: "Low", 1: "Medium", 2: "High"}
df["Traffic_Label"] = df["Traffic_Level"].map(traffic_labels)
sns.boxplot(data=df, x="Traffic_Label", y="Fuel_Consumed_L", order=["Low", "Medium", "High"],
            hue="Traffic_Label", palette="rocket", legend=False)
plt.title("Traffic Level vs Fuel Consumption")
plt.xlabel("Traffic Level")
plt.ylabel("Fuel Consumed (L)")
savefig("03_traffic_vs_consumption")

# 4. AC vs Fuel Consumption
plt.figure(figsize=(6, 5))
df["AC_Label"] = df["AC"].map({0: "OFF", 1: "ON"})
sns.violinplot(data=df, x="AC_Label", y="Fuel_Consumed_L", hue="AC_Label",
               palette="mako", legend=False)
plt.title("AC Usage vs Fuel Consumption")
plt.xlabel("AC")
plt.ylabel("Fuel Consumed (L)")
savefig("04_ac_vs_consumption")

# 5. Load vs Fuel Consumption
plt.figure(figsize=(7, 5))
sns.scatterplot(data=sample, x="Vehicle_Load_kg", y="Fuel_Consumed_L", alpha=0.35, s=15)
sns.regplot(data=sample, x="Vehicle_Load_kg", y="Fuel_Consumed_L", scatter=False, color="red")
plt.title("Vehicle Load vs Fuel Consumption")
plt.xlabel("Load (kg)")
plt.ylabel("Fuel Consumed (L)")
savefig("05_load_vs_consumption")

# 6. RPM vs Fuel Consumption
plt.figure(figsize=(7, 5))
sns.scatterplot(data=sample, x="RPM", y="Fuel_Consumed_L", alpha=0.35, s=15)
sns.regplot(data=sample, x="RPM", y="Fuel_Consumed_L", scatter=False, color="red")
plt.title("RPM vs Fuel Consumption")
plt.xlabel("RPM")
plt.ylabel("Fuel Consumed (L)")
savefig("06_rpm_vs_consumption")

# 7. Correlation heatmap
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include="number").drop(columns=["Efficiency_kmpl"], errors="ignore")
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            annot_kws={"size": 7})
plt.title("Correlation Heatmap - All Numeric Features")
savefig("07_correlation_heatmap")

print("\nKey correlations with Fuel_Consumed_L:")
print(corr["Fuel_Consumed_L"].sort_values(ascending=False))
