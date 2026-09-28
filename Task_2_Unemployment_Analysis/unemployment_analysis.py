import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("Unemployment in India.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# 2. Clean column names
df.columns = [
    "Region",
    "Date",
    "Frequency",
    "Estimated Unemployment Rate",
    "Estimated Employed",
    "Estimated Labour Participation Rate",
    "Area"
]

# Remove spaces from text columns
df["Region"] = df["Region"].str.strip()
df["Area"] = df["Area"].str.strip()

# Convert date
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Convert numeric columns
numeric_columns = [
    "Estimated Unemployment Rate",
    "Estimated Employed",
    "Estimated Labour Participation Rate"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove missing values
df = df.dropna()

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 3. Overall unemployment rate
print("\nAverage Unemployment Rate:",
      round(df["Estimated Unemployment Rate"].mean(), 2), "%")

# 4. Average unemployment by region
region_data = df.groupby("Region")[
    "Estimated Unemployment Rate"
].mean().sort_values(ascending=False)

print("\nAverage Unemployment Rate by Region:")
print(region_data)

# 5. Plot regional unemployment
plt.figure(figsize=(10, 6))
region_data.plot(kind="bar")
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("regional_unemployment.png")
plt.close()

# 6. Monthly unemployment trend
monthly_data = df.groupby("Date")[
    "Estimated Unemployment Rate"
].mean()

plt.figure(figsize=(10, 6))
monthly_data.plot(marker="o")
plt.title("Unemployment Rate Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("unemployment_trend.png")
plt.close()

# 7. Monthly seasonal pattern
df["Month"] = df["Date"].dt.month

monthly_pattern = df.groupby("Month")[
    "Estimated Unemployment Rate"
].mean()

plt.figure(figsize=(8, 5))
monthly_pattern.plot(kind="bar")
plt.title("Monthly Unemployment Pattern")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("monthly_pattern.png")
plt.close()

# 8. COVID-19 period comparison
covid_period = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-12-31")
]

pre_covid_period = df[
    df["Date"] < "2020-03-01"
]

print("\nPre-COVID Average Unemployment Rate:",
      round(
          pre_covid_period["Estimated Unemployment Rate"].mean(), 2
      ),
      "%")

print("COVID-19 Period Average Unemployment Rate:",
      round(
          covid_period["Estimated Unemployment Rate"].mean(), 2
      ),
      "%")

# 9. COVID comparison graph
comparison = pd.Series({
    "Pre-COVID": pre_covid_period["Estimated Unemployment Rate"].mean(),
    "COVID Period": covid_period["Estimated Unemployment Rate"].mean()
})

plt.figure(figsize=(7, 5))
comparison.plot(kind="bar")
plt.title("Unemployment Rate: Pre-COVID vs COVID Period")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("covid_comparison.png")
plt.close()

print("\nTask 2 completed successfully!")
print("Graphs saved successfully.")