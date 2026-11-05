
# ============================================================
# HEALTHCARE DATA ANALYSIS PROJECT
# Simulated Healthcare Dataset - Six Months
# ============================================================

import pandas as pd
import numpy as np
import os

# 1. Set a random seed so results are reproducible
np.random.seed(42)

# 2. Define the clinics
clinics = [
    "Central Clinic",
    "Westlands Health Centre",
    "Kasarani Clinic",
    "Embakasi Medical Centre",
    "Ruiru Health Centre"
]

# 3. Define the six months
months = pd.date_range(
    start="2026-01-01",
    periods=6,
    freq="MS"
)

# 4. Define ailments and their corresponding medication
ailments = {
    "Malaria": "Antimalarial Drugs",
    "Respiratory Infection": "Antibiotics",
    "Diarrhoeal Disease": "ORS",
    "Typhoid": "Typhoid Medication"
}

# 5. Define seasonal patterns for the simulation
seasons = {
    1: "Dry Season",
    2: "Dry Season",
    3: "Long Rains",
    4: "Long Rains",
    5: "Long Rains",
    6: "Cool Season"
}

# Simulated seasonal multipliers
seasonal_effects = {
    "Malaria": {
        "Dry Season": 0.9,
        "Long Rains": 1.4,
        "Cool Season": 1.0
    },
    "Respiratory Infection": {
        "Dry Season": 0.9,
        "Long Rains": 1.1,
        "Cool Season": 1.3
    },
    "Diarrhoeal Disease": {
        "Dry Season": 0.9,
        "Long Rains": 1.3,
        "Cool Season": 1.0
    },
    "Typhoid": {
        "Dry Season": 0.9,
        "Long Rains": 1.2,
        "Cool Season": 1.0
    }
}

# 6. Generate the healthcare records
records = []

for clinic in clinics:
    for month in months:

        month_number = month.month
        season = seasons[month_number]

        for ailment, medication in ailments.items():

            # Generate simulated patient attendance
            base_attendance = np.random.randint(40, 100)

            seasonal_multiplier = seasonal_effects[ailment][season]

            attendance = int(
                base_attendance * seasonal_multiplier
            )

            # Medication supply and demand
            medication_demand = attendance * np.random.uniform(0.8, 1.2)

            medication_supplied = int(
                medication_demand * np.random.uniform(0.85, 1.25)
            )

            medication_dispensed = min(
                int(medication_demand),
                medication_supplied
            )

            closing_stock = max(
                medication_supplied - medication_dispensed, 0
            )

            # Record the results
            records.append({
                "Clinic": clinic,
                "Month": month.strftime("%B"),
                "Month_Number": month_number,
                "Season": season,
                "Ailment": ailment,
                "Medication": medication,
                "Patient_Attendance": attendance,
                "Medication_Demand": int(medication_demand),
                "Medication_Supplied": medication_supplied,
                "Medication_Dispensed": medication_dispensed,
                "Closing_Stock": closing_stock
            })

# 7. Convert records into a Pandas DataFrame
healthcare_df = pd.DataFrame(records)

# 8. Create the data folder if it does not exist
os.makedirs("data", exist_ok=True)

# 9. Export the dataset to CSV
healthcare_df.to_csv(
    "data/healthcare_data.csv",
    index=False
)

# 10. Display the dataset information
print("\n========================================")
print("HEALTHCARE DATASET GENERATED SUCCESSFULLY")
print("========================================")

print("\nFirst 10 Records:")
print(healthcare_df.head(10))

print("\nDataset Information:")
print(healthcare_df.info())

print("\nDataset Dimensions:")
print(healthcare_df.shape)

print("\nMissing Values:")
print(healthcare_df.isnull().sum())

print("\nDataset saved to data/healthcare_data.csv")

# ============================================================
# STEP 5: DIMENSIONALITY REDUCTION USING PCA
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("\n========================================")
print("DIMENSIONALITY REDUCTION USING PCA")
print("========================================")

# 1. Select numerical healthcare variables
features = [
    "Patient_Attendance",
    "Medication_Demand",
    "Medication_Supplied",
    "Medication_Dispensed",
    "Closing_Stock"
]

X = healthcare_df[features]

# 2. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA while retaining at least 90% variance
pca = PCA(n_components=0.90)

principal_components = pca.fit_transform(X_scaled)

# 4. Display the number of components selected
print("\nOriginal number of numerical features:", len(features))
print("Reduced number of principal components:",
      pca.n_components_)

# 5. Display explained variance
print("\nExplained Variance Ratio:")
for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {variance:.4f} ({variance*100:.2f}%)")

print(
    "\nTotal Variance Retained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

# 6. Preserve the healthcare dimensions for OLAP
dimension_columns = [
    "Clinic",
    "Month",
    "Month_Number",
    "Season",
    "Ailment",
    "Medication"
]

reduced_df = healthcare_df[dimension_columns].copy()

# 7. Add the new principal components
for i in range(pca.n_components_):
    reduced_df[f"PC{i+1}"] = principal_components[:, i]

# 8. Save the reduced dataset
reduced_df.to_csv(
    "outputs/reduced_healthcare_data.csv",
    index=False
)

# 9. Save PCA variance results
variance_df = pd.DataFrame({
    "Principal_Component": [
        f"PC{i+1}" for i in range(pca.n_components_)
    ],
    "Explained_Variance": pca.explained_variance_ratio_
})

variance_df.to_csv(
    "outputs/pca_variance.csv",
    index=False
)

# 10. Display the results
print("\nReduced Dataset Preview:")
print(reduced_df.head())

print("\nReduced Dataset Dimensions:")
print(reduced_df.shape)

print("\nPCA files saved successfully!")

