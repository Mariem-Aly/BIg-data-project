import os

os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, trim
from pyspark.sql.types import IntegerType

# Create Spark Session
spark = SparkSession.builder \
    .appName("BlackFriday_Preprocessing") \
    .master("local[*]") \
    .getOrCreate()


# Read CSV
df = spark.read.csv(
    "C:/Users/dubai phone/Desktop/vs python/big data project/train.csv",
    header=True,
    inferSchema=True
)

# Show data
df.show(5)

# Schema
df.printSchema()

# Statistics
df.describe().show()

# Columns
print(df.columns)


# Missing values
print("Missing Values:")
df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show()


# Duplicate rows
print("Duplicate Rows:")
print(df.count() - df.dropDuplicates().count())


# Drop unnecessary columns
df = df.drop(
    "Product_Category_2",
    "Product_Category_3"
)


# Change data types
df = df.withColumn(
    "User_ID",
    col("User_ID").cast("string")
)

df = df.withColumn(
    "Occupation",
    col("Occupation").cast("string")
)


# Replace 4+ with 4 and convert to integer
df = df.withColumn(
    "Stay_In_Current_City_Years",
    when(
        col("Stay_In_Current_City_Years") == "4+",
        4
    ).otherwise(
        col("Stay_In_Current_City_Years")
    ).cast(IntegerType())
)


# Age mapping
df = df.withColumn(
    "Age",
    when(col("Age")=="0-17",0)
    .when(col("Age")=="18-25",1)
    .when(col("Age")=="26-35",2)
    .when(col("Age")=="36-45",3)
    .when(col("Age")=="46-50",4)
    .when(col("Age")=="51-55",5)
    .when(col("Age")=="55+",6)
)


# Unique values
for c in [
    "Gender",
    "Age",
    "Occupation",
    "Marital_Status",
    "Stay_In_Current_City_Years"
]:
    print(c)
    df.select(c).distinct().show()


# Top Product IDs
df.groupBy("Product_ID") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(10)


# Train/Test Split
train_df, test_df = df.randomSplit(
    [0.8,0.2],
    seed=42
)


# Lowercase column names
for c in train_df.columns:
    train_df = train_df.withColumnRenamed(c, c.lower())
    test_df = test_df.withColumnRenamed(c, c.lower())


# Remove spaces from string columns
string_cols = [
    f.name for f in train_df.schema.fields
    if str(f.dataType) == "StringType()"
]


for c in string_cols:
    train_df = train_df.withColumn(
        c,
        trim(col(c))
    )

    test_df = test_df.withColumn(
        c,
        trim(col(c))
    )


# Missing values after cleaning
print("Train Missing:")
train_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in train_df.columns
]).show()


print("Test Missing:")
test_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in test_df.columns
]).show()


# Duplicate check
print("Train duplicates:",
      train_df.count() - train_df.dropDuplicates().count())

print("Test duplicates:",
      test_df.count() - test_df.dropDuplicates().count())


# Outlier Detection using IQR

q1 = train_df.approxQuantile(
    "purchase",
    [0.25],
    0.01
)[0]

q3 = train_df.approxQuantile(
    "purchase",
    [0.75],
    0.01
)[0]


iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr


print("Lower:", lower_limit)
print("Upper:", upper_limit)


# Capping outliers
train_df = train_df.withColumn(
    "purchase",
    when(col("purchase") < lower_limit, lower_limit)
    .when(col("purchase") > upper_limit, upper_limit)
    .otherwise(col("purchase"))
)


test_df = test_df.withColumn(
    "purchase",
    when(col("purchase") < lower_limit, lower_limit)
    .when(col("purchase") > upper_limit, upper_limit)
    .otherwise(col("purchase"))
)


train_df.show(5)






from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col, count


# -----------------------------
# Features and Target
# -----------------------------

# Drop ID columns and separate target
train_data = train_df.drop(
    "user_id",
    "product_id"
)

test_data = test_df.drop(
    "user_id",
    "product_id"
)







# -----------------------------
# Encoding categorical columns
# -----------------------------

categorical_cols = [
    "gender",
    "occupation",
    "city_category"
]


# Convert categorical columns to indexes
indexers = []

for c in categorical_cols:
    indexers.append(
        StringIndexer(
            inputCol=c,
            outputCol=c+"_index"
        )
    )


# Apply StringIndexer
for indexer in indexers:
    model = indexer.fit(train_data)

    train_data = model.transform(train_data)
    test_data = model.transform(test_data)

# One Hot Encoding

encoder = OneHotEncoder(
    inputCols=[
        c+"_index" for c in categorical_cols
    ],
    outputCols=[
        c+"_encoded" for c in categorical_cols
    ],
    dropLast=True
)


encoder_model = encoder.fit(train_data)

train_data = encoder_model.transform(train_data)
test_data = encoder_model.transform(test_data)

# -----------------------------
# Combine features
# -----------------------------

feature_cols = [
    c for c in train_data.columns
    if c not in categorical_cols
    and not c.endswith("_index")
    and c != "purchase"
]

feature_cols += [
    c+"_encoded"
    for c in categorical_cols
]


assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)


train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

print("Final Training Data:")
train_data.select(
    "features",
    "purchase"
).show(5, truncate=False)


# -----------------------------
# Check categorical columns
# -----------------------------

print("Gender:")
train_df.groupBy("gender") \
    .count() \
    .show()


print("City Category:")
train_df.groupBy("city_category") \
    .count() \
    .show()


print("Marital Status:")
train_df.groupBy("marital_status") \
    .count() \
    .show()





from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# =============================
# Random Forest
# =============================

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="purchase",
    numTrees=100,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(train_data)

rf_predictions = rf_model.transform(test_data)

evaluator_rmse = RegressionEvaluator(
    labelCol="purchase",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="purchase",
    predictionCol="prediction",
    metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="purchase",
    predictionCol="prediction",
    metricName="r2"
)

rf_rmse = evaluator_rmse.evaluate(rf_predictions)
rf_mae = evaluator_mae.evaluate(rf_predictions)
rf_r2 = evaluator_r2.evaluate(rf_predictions)

print("Random Forest Results")
print("RMSE:", rf_rmse)
print("MAE :", rf_mae)
print("R²  :", rf_r2)


# =============================
# Gradient Boosted Trees
# =============================

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="purchase",
    maxIter=200,
    maxDepth=5,
    stepSize=0.05,
    seed=42
)

gbt_model = gbt.fit(train_data)

gbt_predictions = gbt_model.transform(test_data)

gb_rmse = evaluator_rmse.evaluate(gbt_predictions)
gb_mae = evaluator_mae.evaluate(gbt_predictions)
gb_r2 = evaluator_r2.evaluate(gbt_predictions)

print("\nGradient Boosting Results")
print("RMSE:", gb_rmse)
print("MAE :", gb_mae)
print("R²  :", gb_r2)


comparison = spark.createDataFrame([
    ("Random Forest", rf_rmse, rf_mae, rf_r2),
    ("Gradient Boosting", gb_rmse, gb_mae, gb_r2)
], ["Model", "RMSE", "MAE", "R2"])

comparison.show()





# =====================================================
# Spark KMeans Clustering
# =====================================================

from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import avg, count, max


# =====================================================
# 1. Scale Features
# =====================================================

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(train_data)

train_cluster = scaler_model.transform(train_data)


# =====================================================
# 2. Elbow Method
# =====================================================

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        featuresCol="scaled_features",
        predictionCol="Cluster",
        k=k,
        seed=42
    )

    model = kmeans.fit(train_cluster)

    wcss.append(model.summary.trainingCost)


# Convert results for plotting

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.grid(True)
plt.show()



# =====================================================
# 3. Train Final KMeans Model
# =====================================================

kmeans = KMeans(
    featuresCol="scaled_features",
    predictionCol="Cluster",
    k=4,
    seed=42
)


kmeans_model = kmeans.fit(train_cluster)


clustered_df = kmeans_model.transform(train_cluster)



# Show clusters

clustered_df.select(
    "purchase",
    "age",
    "city_category",
    "occupation",
    "Cluster"
).show(10)



# =====================================================
# 4. Cluster Evaluation
# =====================================================

evaluator = ClusteringEvaluator(
    featuresCol="scaled_features",
    predictionCol="Cluster",
    metricName="silhouette"
)

silhouette = evaluator.evaluate(clustered_df)

print("Silhouette Score:", silhouette)



# =====================================================
# 5. Average Purchase By Cluster
# =====================================================

cluster_purchase = (
    clustered_df
    .groupBy("Cluster")
    .agg(
        avg("purchase")
        .alias("Avg_Purchase")
    )
    .orderBy("Cluster")
)


cluster_purchase.show()



# Convert to Pandas for visualization

cluster_purchase_pd = cluster_purchase.toPandas()


plt.figure(figsize=(8,5))

plt.bar(
    cluster_purchase_pd["Cluster"].astype(str),
    cluster_purchase_pd["Avg_Purchase"]
)

plt.title("Average Purchase by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Purchase")

plt.show()



# =====================================================
# 6. Number of Customers per Cluster
# =====================================================

cluster_count = (
    clustered_df
    .groupBy("Cluster")
    .agg(
        count("*")
        .alias("Customers")
    )
    .orderBy("Cluster")
)


cluster_count.show()


cluster_count_pd = cluster_count.toPandas()


plt.figure(figsize=(8,5))

plt.bar(
    cluster_count_pd["Cluster"].astype(str),
    cluster_count_pd["Customers"]
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Customers")

plt.show()



# =====================================================
# 7. Cluster Summary Table
# =====================================================

cluster_summary = (
    clustered_df
    .groupBy("Cluster")
    .agg(
        count("*").alias("Customers"),
        avg("purchase").alias("Avg_Purchase"),
        avg("age").alias("Avg_Age"),
        avg("stay_in_current_city_years")
        .alias("Avg_Stay"),
        avg("marital_status")
        .alias("Married_Rate")
    )
    .orderBy("Cluster")
)


cluster_summary.show()



# =====================================================
# 8. Most Common Characteristics
# =====================================================

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc


def top_value(df, column):

    temp = (
        df.groupBy(
            "Cluster",
            column
        )
        .count()
    )

    window = Window.partitionBy(
        "Cluster"
    ).orderBy(
        desc("count")
    )


    return (
        temp.withColumn(
            "rank",
            row_number().over(window)
        )
        .filter(
            col("rank")==1
        )
        .select(
            "Cluster",
            col(column).alias(
                "Top_"+column
            )
        )
    )



top_gender = top_value(
    clustered_df,
    "gender"
)

top_city = top_value(
    clustered_df,
    "city_category"
)

top_occupation = top_value(
    clustered_df,
    "occupation"
)

top_product = top_value(
    clustered_df,
    "product_category_1"
)



# Join all summaries

cluster_summary = (
    cluster_summary
    .join(top_gender,"Cluster")
    .join(top_city,"Cluster")
    .join(top_occupation,"Cluster")
    .join(top_product,"Cluster")
)


print("Final Cluster Summary")

cluster_summary.show(truncate=False)



# =====================================================
# 9. Convert Final Summary for Visualization
# =====================================================

cluster_vis = cluster_summary.toPandas()


# Average Purchase

plt.figure(figsize=(8,5))

plt.bar(
    cluster_vis["Cluster"].astype(str),
    cluster_vis["Avg_Purchase"]
)

plt.title("Average Purchase Value per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Purchase")

plt.show()



# Age and Stay Comparison

cluster_vis.set_index("Cluster")[
    [
        "Avg_Age",
        "Avg_Stay"
    ]
].plot(
    kind="bar",
    figsize=(8,5)
)


plt.title("Customer Profile: Age and City Stay")
plt.xlabel("Cluster")
plt.ylabel("Average Value")

plt.xticks(rotation=0)

plt.show()



# Married Rate

plt.figure(figsize=(8,5))

plt.bar(
    cluster_vis["Cluster"].astype(str),
    cluster_vis["Married_Rate"]
)

plt.title("Married Rate per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Marriage Rate")

plt.show()

spark.stop()
