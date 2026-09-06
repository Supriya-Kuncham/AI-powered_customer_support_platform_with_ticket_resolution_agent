import pandas as pd
import numpy as np

# ============================================================
# IT SUPPORT TICKET DATA - DATA CLEANING
# 7 STEPS OF DATA CLEANING USING PANDAS
# ============================================================

# ------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------------------------

input_file = "IT Support Ticket Data.csv"
output_file = "cleaned_IT_support_ticket_data.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# STEP 1: HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1: HANDLING MISSING VALUES")
print("=" * 60)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing text values with "Unknown"
text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = df[column].fillna("Unknown")

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 2: HANDLE DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: HANDLING DUPLICATES")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate rows found:", duplicates)

df.drop_duplicates(inplace=True)

print("Duplicate rows after cleaning:", df.duplicated().sum())


# ------------------------------------------------------------
# STEP 3: HANDLE DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: HANDLING DATA TYPES")
print("=" * 60)

print("\nData types before cleaning:")
print(df.dtypes)

# Convert text columns to string type
for column in ["Body", "Department", "Priority", "Tags"]:
    if column in df.columns:
        df[column] = df[column].astype("string")

# Convert index column to integer
if "Unnamed: 0" in df.columns:
    df["Unnamed: 0"] = pd.to_numeric(
        df["Unnamed: 0"],
        errors="coerce"
    )

print("\nData types after cleaning:")
print(df.dtypes)


# ------------------------------------------------------------
# STEP 4: HANDLE OUTLIERS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: HANDLING OUTLIERS")
print("=" * 60)

# This dataset mainly contains categorical and text data.
# There is no meaningful numerical measurement such as Age,
# Salary, Price, etc. for normal outlier removal.
#
# Therefore, we check the text length of each ticket.
# Extremely long/short records can be identified.

df["Body_Length"] = df["Body"].str.len()

Q1 = df["Body_Length"].quantile(0.25)
Q3 = df["Body_Length"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = (
    (df["Body_Length"] < lower_limit) |
    (df["Body_Length"] > upper_limit)
).sum()

print("Potential text-length outliers:", outliers)

# Keep all tickets because a long support ticket can still
# be a valid customer problem.
# We identify the outliers instead of deleting valid tickets.

print("Outliers identified but valid ticket records are retained.")


# ------------------------------------------------------------
# STEP 5: STRING OPERATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: STRING OPERATIONS")
print("=" * 60)

# Remove unnecessary spaces
df["Body"] = df["Body"].str.strip()
df["Department"] = df["Department"].str.strip()
df["Priority"] = df["Priority"].str.strip()
df["Tags"] = df["Tags"].str.strip()

# Standardize Priority values
df["Priority"] = df["Priority"].str.lower()

# Standardize Department values
df["Department"] = df["Department"].str.strip()

print("String cleaning completed.")


# ------------------------------------------------------------
# STEP 6: ENCODING CATEGORICAL VARIABLES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6: ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

# One-hot encoding for Department
department_encoded = pd.get_dummies(
    df["Department"],
    prefix="Department",
    dtype=int
)

# One-hot encoding for Priority
priority_encoded = pd.get_dummies(
    df["Priority"],
    prefix="Priority",
    dtype=int
)

# Add encoded columns to dataframe
df = pd.concat(
    [df, department_encoded, priority_encoded],
    axis=1
)

print("Categorical encoding completed.")

print("\nEncoded columns:")
print(
    department_encoded.columns.tolist()
    + priority_encoded.columns.tolist()
)


# ------------------------------------------------------------
# STEP 7: HANDLING LARGE DATASETS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7: HANDLING LARGE DATASETS")
print("=" * 60)

# Check memory usage
memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)

print("Current memory usage:",
      round(memory_usage, 2), "MB")

# Demonstration of chunk processing
print("\nDataset can be processed in chunks using Pandas.")

chunk_count = 0

for chunk in pd.read_csv(input_file, chunksize=5000):
    chunk_count += 1

print("Number of chunks of 5000 rows:",
      chunk_count)


# ------------------------------------------------------------
# FINAL CLEANING
# ------------------------------------------------------------

# Remove temporary column used for outlier analysis
df.drop(columns=["Body_Length"], inplace=True)


# ------------------------------------------------------------
# SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print("Final dataset shape:", df.shape)
print("Cleaned file saved as:", output_file)

print("\nFirst 5 rows of cleaned dataset:")
print(df.head())