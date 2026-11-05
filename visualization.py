
# ============================================================
# HEALTHCARE DATA ANALYSIS PROJECT
# STAGE 3: DATA VISUALISATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Create output folder for charts
os.makedirs("outputs/figures", exist_ok=True)

# 2. Set visualisation style
sns.set_theme(style="whitegrid")

# 3. Load healthcare dataset
df = pd.read_csv("data/healthcare_data.csv")

print("\n========================================")
print("HEALTHCARE DATA VISUALISATION")
print("========================================")

# ============================================================
# CHART 1: PATIENT ATTENDANCE BY MONTH
# ============================================================

monthly_attendance = df.groupby(
    ["Month_Number", "Month"]
)["Patient_Attendance"].sum().reset_index()

monthly_attendance = monthly_attendance.sort_values("Month_Number")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=monthly_attendance,
    x="Month",
    y="Patient_Attendance"
)

plt.title("Monthly Patient Attendance", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Total Patient Attendance")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("outputs/figures/monthly_attendance.png", dpi=300)
plt.show()

# ============================================================
# CHART 2: MEDICATION DEMAND VERSUS SUPPLY
# ============================================================

monthly_medication = df.groupby(
    ["Month_Number", "Month"]
)[["Medication_Demand", "Medication_Supplied"]].sum().reset_index()

monthly_medication = monthly_medication.sort_values("Month_Number")

medication_long = monthly_medication.melt(
    id_vars=["Month_Number", "Month"],
    value_vars=["Medication_Demand", "Medication_Supplied"],
    var_name="Medication_Category",
    value_name="Quantity"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=medication_long,
    x="Month",
    y="Quantity",
    hue="Medication_Category"
)

plt.title("Monthly Medication Demand vs Supply", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Medication Quantity")
plt.xticks(rotation=30)
plt.legend(title="Medication Category")

plt.tight_layout()
plt.savefig("outputs/figures/medication_demand_supply.png", dpi=300)
plt.show()

# ============================================================
# CHART 3: PATIENT ATTENDANCE BY CLINIC
# ============================================================

clinic_attendance = df.groupby(
    "Clinic"
)["Patient_Attendance"].sum().reset_index()

plt.figure(figsize=(11, 6))

sns.barplot(
    data=clinic_attendance,
    x="Clinic",
    y="Patient_Attendance"
)

plt.title("Patient Attendance by Healthcare Facility", fontsize=14)
plt.xlabel("Healthcare Facility")
plt.ylabel("Total Patient Attendance")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig("outputs/figures/clinic_attendance.png", dpi=300)
plt.show()

# ============================================================
# CHART 4: DISTRIBUTION OF HEALTHCARE AILMENTS
# ============================================================

ailment_distribution = df["Ailment"].value_counts().reset_index()

ailment_distribution.columns = ["Ailment", "Frequency"]

plt.figure(figsize=(10, 6))

sns.barplot(
    data=ailment_distribution,
    x="Ailment",
    y="Frequency"
)

plt.title("Distribution of Recorded Healthcare Ailments", fontsize=14)
plt.xlabel("Ailment")
plt.ylabel("Number of Records")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig("outputs/figures/ailment_distribution.png", dpi=300)
plt.show()

# ============================================================
# CHART 5: PCA SCATTER PLOT
# ============================================================

pca_df = pd.read_csv("outputs/reduced_healthcare_data.csv")

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Ailment",
    s=70,
    alpha=0.8
)

plt.title("PCA: Healthcare Data Distribution", fontsize=14)
plt.xlabel("Principal Component 1 (77.81%)")
plt.ylabel("Principal Component 2 (18.85%)")
plt.legend(title="Ailment", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("outputs/figures/pca_scatter_plot.png", dpi=300)
plt.show()

# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("VISUALISATION COMPLETED SUCCESSFULLY")
print("========================================")

print("\nFive charts have been generated and saved in:")

print("outputs/figures/")

print("\n1. monthly_attendance.png")
print("2. medication_demand_supply.png")
print("3. clinic_attendance.png")
print("4. ailment_distribution.png")
print("5. pca_scatter_plot.png")

print("\nAll visualisation operations completed!")
