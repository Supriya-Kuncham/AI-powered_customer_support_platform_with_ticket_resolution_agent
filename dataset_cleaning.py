import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the customer support ticket dataset
df = pd.read_csv("customer_support_tickets.csv")

# Check the first 5 rows
print(df.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)


# STEP 1: Handle Missing Values

# Check missing values in each column
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values with the most frequent value
for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# STEP 2: Handle Duplicates

# Check the number of duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# STEP 3: Handle Data Types

# Check the data types of all columns
print("\nData types:")
print(df.dtypes)

# Convert date/time columns to datetime format
for column in df.columns:
    if "date" in column.lower() or "time" in column.lower():
        df[column] = pd.to_datetime(df[column], errors="coerce")

print("\nDate/time columns converted successfully.")


# STEP 4: Handle Outliers

# Select numerical columns
numeric_columns = df.select_dtypes(include="number").columns

# Display statistical information about numerical columns
print("\nNumerical data statistics:")
print(df[numeric_columns].describe())

# Remove outliers using the IQR method
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[
        (df[column] >= lower_limit) &
        (df[column] <= upper_limit)
    ]


# STEP 5: String Operations

# Select text columns
text_columns = df.select_dtypes(include=["object", "str"]).columns

# Remove unnecessary spaces from text values
for column in text_columns:
    df[column] = df[column].str.strip()


# STEP 6: Encoding Categorical Variables

# Select categorical columns
categorical_columns = df.select_dtypes(include=["object", "str"]).columns

# Create a LabelEncoder
le = LabelEncoder()

# Convert categorical values into numerical values
for column in categorical_columns:

    # Encode columns having a small number of categories
    if df[column].nunique() <= 20:
        df[column + "_encoded"] = le.fit_transform(
            df[column].astype(str)
        )


# STEP 7: Handling Large Datasets

# Check the memory usage of the dataset
print("\nMemory usage:")
df.info(memory_usage="deep")

# Example of loading a large dataset in chunks
# for chunk in pd.read_csv("customer_support_tickets.csv", chunksize=5000):
#     print(chunk.shape)


# Display the final cleaned dataset
print("\nFinal cleaned dataset:")
print(df.head())

# Display the final number of rows and columns
print("\nFinal dataset shape:")
print(df.shape)

# Save the cleaned dataset as a new CSV file
df.to_csv("cleaned_customer_support_tickets.csv", index=False)

print("\nCleaned dataset saved successfully!")