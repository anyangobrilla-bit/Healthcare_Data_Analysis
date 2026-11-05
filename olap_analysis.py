
# ============================================================
# HEALTHCARE DATA ANALYSIS PROJECT
# OLAP IMPLEMENTATION
# ============================================================

import pandas as pd
import os

# 1. Load the original healthcare dataset
data_path = "data/healthcare_data.csv"

df = pd.read_csv(data_path)

# 2. Create outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)

# 3. Verify the dataset
print("\n========================================")
print("HEALTHCARE DATASET LOADED SUCCESSFULLY")
print("========================================")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Dimensions:")
print(df.shape)

print("\nAvailable Columns:")
print(df.columns.tolist())

print("\nDataset loaded and ready for OLAP analysis!")

# ============================================================
# HEALTHCARE DATA ANALYSIS PROJECT
# OLAP OPERATIONS
# ============================================================

import pandas as pd
import os

# 1. Create output folder
os.makedirs("outputs", exist_ok=True)

# 2. Load healthcare dataset
df = pd.read_csv("data/healthcare_data.csv")

print("\n========================================")
print("HEALTHCARE OLAP ANALYSIS")
print("========================================")

print("\nOriginal Dataset Dimensions:")
print(df.shape)


# ============================================================
# OPERATION 1: SLICE
# Analyse healthcare data for one particular season
# ============================================================

print("\n========================================")
print("1. SLICE OPERATION")
print("========================================")

season = "Dry Season"

if season not in df["Season"].unique():
    season = df["Season"].iloc[0]

slice_df = df[df["Season"] == season]

print(f"\nHealthcare Records for {season}:")
print(slice_df)

print("\nSlice Dataset Dimensions:")
print(slice_df.shape)

slice_df.to_csv(
    "outputs/olap_slice.csv",
    index=False
)


# ============================================================
# OPERATION 2: DICE
# Analyse selected clinics and ailments
# ============================================================

print("\n========================================")
print("2. DICE OPERATION")
print("========================================")

clinics = sorted(df["Clinic"].unique())[:2]
ailments = sorted(df["Ailment"].unique())[:2]

dice_df = df[
    (df["Clinic"].isin(clinics)) &
    (df["Ailment"].isin(ailments))
]

print("\nSelected Clinics:")
print(clinics)

print("\nSelected Ailments:")
print(ailments)

print("\nDice Results:")
print(dice_df)

print("\nDice Dataset Dimensions:")
print(dice_df.shape)

dice_df.to_csv(
    "outputs/olap_dice.csv",
    index=False
)


# ============================================================
# OPERATION 3: DRILL-DOWN
# Examine monthly healthcare trends
# ============================================================

print("\n========================================")
print("3. DRILL-DOWN OPERATION")
print("========================================")

monthly_analysis = df.groupby(
    ["Month_Number", "Month"],
    as_index=False
).agg({
    "Patient_Attendance": "sum",
    "Medication_Demand": "sum",
    "Medication_Supplied": "sum",
    "Medication_Dispensed": "sum"
}).sort_values("Month_Number")

print("\nMonthly Healthcare Trends:")
print(monthly_analysis)

monthly_analysis.to_csv(
    "outputs/olap_drilldown.csv",
    index=False
)


# ============================================================
# OPERATION 4: ROLL-UP
# Summarise healthcare data by clinic and season
# ============================================================

print("\n========================================")
print("4. ROLL-UP OPERATION")
print("========================================")

rollup_analysis = df.groupby(
    ["Clinic", "Season"],
    as_index=False
).agg({
    "Patient_Attendance": "sum",
    "Medication_Demand": "sum",
    "Medication_Supplied": "sum",
    "Medication_Dispensed": "sum",
    "Closing_Stock": "sum"
})

print("\nHealthcare Summary by Clinic and Season:")
print(rollup_analysis)

rollup_analysis.to_csv(
    "outputs/olap_rollup.csv",
    index=False
)


# ============================================================
# SAVE OLAP RESULTS
# ============================================================

print("\n========================================")
print("OLAP ANALYSIS COMPLETED SUCCESSFULLY")
print("========================================")

print("\nGenerated Files:")

print("1. outputs/olap_slice.csv")
print("2. outputs/olap_dice.csv")
print("3. outputs/olap_drilldown.csv")
print("4. outputs/olap_rollup.csv")

print("\nAll OLAP operations completed successfully!")
