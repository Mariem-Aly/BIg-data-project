import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("train.csv")
له
print(df.head())
df.info()
print(df.isnull().sum())
#filling missing values with 0
df["Product_Category_2"] = df["Product_Category_2"].fillna(0)
df["Product_Category_3"] = df["Product_Category_3"].fillna(0)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.dtypes)
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")
print(df.columns)
#checking spaces and odd characters 
for col in df.select_dtypes(include="object").columns:
    print(col, df[col].unique()[:10])
    df[col] = df[col].str.strip()
# Convert product category columns from float to int after filling missing values
df["product_category_2"] = df["product_category_2"].astype(int)
df["product_category_3"] = df["product_category_3"].astype(int)
print(df["purchase"].describe())
q1 = df["purchase"].quantile(0.25)
q3 = df["purchase"].quantile(0.75)

iqr = q3 - q1

outliers = df[(df["purchase"] < q1 - 1.5*iqr) | 
              (df["purchase"] > q3 + 1.5*iqr)]

print(f"Number of outliers: {len(outliers)}")
plt.figure(figsize=(8,5))
plt.hist(df["purchase"], bins=50)
plt.xlabel("Purchase Amount")
plt.ylabel("Frequency")
plt.title("Purchase Distribution")
plt.show()
#outliers were 2677 out of 550068 rows, which is less than 1% of the data
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
df["purchase"] = df["purchase"].clip(lower=lower_limit, upper=upper_limit)
print(df["purchase"].describe())
plt.figure(figsize=(8,4))
plt.boxplot(df["purchase"])
plt.title("Purchase After Capping Outliers")
plt.ylabel("Purchase")
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df["purchase"], bins=50, edgecolor="black")
plt.title("Purchase Distribution After Capping")
plt.xlabel("Purchase")
plt.ylabel("Frequency")
plt.show()
#checking categorical columns
print(df["gender"].value_counts())
print(df["city_category"].value_counts())
print(df["marital_status"].value_counts())