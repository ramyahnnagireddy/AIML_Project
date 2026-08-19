
import pandas as pd
import os

# Load the Indian dataset into a Pandas DataFrame
df = pd.read_csv("W1D2_Pandas/data/indian_state_population.csv")

# Print the shape of the DataFrame
print("Shape:")
print(df.shape)

# Print the data types of each column
print("\nData Types:")
print(df.dtypes)

# Print the first 10 rows
print("\nFirst 10 Rows:")
print(df.head(10))
# Filter: select states with population in 2024 greater than 50 million
filtered_df = df[df["population(2024)"] > 50000000]

print("\nFiltered Data:")
print(filtered_df[["States/Uts", "population(2024)"]])

# Groupby: calculate the average 2024 population by majority religion
grouped_df = df.groupby("Majority")["population(2024)"].mean()

print("\nGrouped Data:")
print(grouped_df)
# Create a second DataFrame for demonstrating merge
state_info = pd.DataFrame({
    "States/Uts": ["Karnataka", "Maharashtra", "Tamil Nadu", "Kerala", "Gujarat"],
    "Region": ["South", "West", "South", "South", "West"]
})

# Merge the main DataFrame with the state information DataFrame
merged_df = pd.merge(df, state_info, on="States/Uts", how="inner")

print("\nMerged Data:")
print(merged_df[["States/Uts", "population(2024)", "Region"]])
# Create another DataFrame for demonstrating concat
additional_states = pd.DataFrame({
    "States/Uts": ["Odisha", "Punjab"],
    "population(2024)": [47000000, 31000000]
})

# Concatenate the two DataFrames
concatenated_df = pd.concat(
    [df[["States/Uts", "population(2024)"]], additional_states],
    ignore_index=True
)

print("\nConcatenated Data:")
print(concatenated_df.tail())
# Create a pivot table showing total population by Region and Majority
pivot_df = pd.pivot_table(
    merged_df,
    values="population(2024)",
    index="Region",
    columns="Majority",
    aggfunc="sum",
    fill_value=0
)

print("\nPivot Table:")
print(pivot_df)
# Export the merged DataFrame to a CSV file
output_csv = "W1D2_Pandas/data/merged_state_population.csv"
merged_df.to_csv(output_csv, index=False)

print("\nCSV Export:")
print(f"Data exported successfully to: {output_csv}")
# Export the merged DataFrame to a Parquet file
output_parquet = "W1D2_Pandas/data/merged_state_population.parquet"
merged_df.to_parquet(output_parquet, index=False)

print("\nParquet Export:")
print(f"Data exported successfully to: {output_parquet}")
# Compare the file sizes of CSV and Parquet files
csv_size = os.path.getsize(output_csv)
parquet_size = os.path.getsize(output_parquet)

print("\nFile Size Comparison:")
print(f"CSV size: {csv_size} bytes")
print(f"Parquet size: {parquet_size} bytes")
