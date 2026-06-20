import pandas as pd
import numpy as np

print("Cleaning dataset...")

# Load original dataset, preventing 'None' string from being parsed as NaN automatically
df = pd.read_csv(r"D:\ML CCP\diabetic_data.csv", keep_default_na=False)

# Clean and Preprocess
df = df.replace("?", np.nan)
cols_to_drop = ["weight", "payer_code", "medical_specialty", "encounter_id", "patient_nbr"]
df = df.drop(cols_to_drop, axis=1)

# Drop rows with actual missing data (race, diag_1, etc.)
df = df.dropna()

print(f"Dataset shape after cleaning: {df.shape}")

# Save the completely cleaned dataset locally to the correct file
df.to_csv(r"D:\ML CCP\cleaned_datasetccp.csv", index=False)
print("Success! 'cleaned_datasetccp.csv' has been saved to your folder.")