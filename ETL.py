import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# Splitting
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
#cleaning
for data in [train_df, test_df]:

    # Renaming columns
    data.columns = data.columns.str.lower()

    # Removing spaces
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()


# Handling Missing Values
train_df["product_category_2"] = train_df["product_category_2"].fillna(0)
train_df["product_category_3"] = train_df["product_category_3"].fillna(0)

test_df["product_category_2"] = test_df["product_category_2"].fillna(0)
test_df["product_category_3"] = test_df["product_category_3"].fillna(0)
# Convert product category columns from float to int after filling missing values
for data in [train_df, test_df]:

    data["product_category_2"] = data["product_category_2"].astype(int)
    data["product_category_3"] = data["product_category_3"].astype(int)
# Check Missing Values
print(train_df.isnull().sum())
print(test_df.isnull().sum())
# Check Duplicates
print(train_df.duplicated().sum())
print(test_df.duplicated().sum())
# Outlier Detection

q1 = train_df["purchase"].quantile(0.25)
q3 = train_df["purchase"].quantile(0.75)

iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

# Histogram before

plt.figure(figsize=(8,5))
plt.hist(train_df["purchase"], bins=50)
plt.title("Purchase Distribution Before Capping")
plt.show()
#outliers were 2677 out of 550068 rows, which is less than 1% of the data

# Apply Capping
train_df["purchase"] = train_df["purchase"].clip(
    lower=lower_limit,
    upper=upper_limit
)

test_df["purchase"] = test_df["purchase"].clip(
    lower=lower_limit,
    upper=upper_limit
)
# Visualization
plt.figure(figsize=(8,4))
plt.boxplot(train_df["purchase"])
plt.title("Purchase After Capping")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(train_df["purchase"], bins=50)
plt.title("Purchase Distribution After Capping")
plt.show()

# Checking categorical columns

print(train_df["gender"].value_counts())
print(train_df["city_category"].value_counts())
print(train_df["marital_status"].value_counts())
