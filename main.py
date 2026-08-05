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

#-----------------------------------------------------------------
df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("hdfs:///user/hadoop/cleaned_data")

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




"""
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





from pyspark.ml.regression import  GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator










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
# Save the trained GBT model
gbt_model.write().overwrite().save("gbt_saved_model")

gbt_predictions = gbt_model.transform(test_data)



gb_rmse = evaluator_rmse.evaluate(gbt_predictions)
gb_mae = evaluator_mae.evaluate(gbt_predictions)
gb_r2 = evaluator_r2.evaluate(gbt_predictions)

print("\nGradient Boosting Results")
print("RMSE:", gb_rmse)
print("MAE :", gb_mae)
print("R²  :", gb_r2)



"""




