from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit

# HEART DISEASE DATASET — CLASS BALANCING (SPARK)


# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("HeartDiseaseBalancing") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# 2. Read Raw Dataset from S3
print("... READING RAW DATA ...")

raw_path = "s3://heart-disease-ga2/raw/heart_2022_no_nans.csv"

df = spark.read.csv(
    raw_path,
    header=True,
    inferSchema=True
)

# 3. Calculate Class Balancing Ratio

# Based on dataset audit:
# Yes  = 13,435 (Heart Attack)
# No   = 232,587 (No Heart Attack)

print("... CALCULATING CLASS WEIGHTS ...")

yes_count = df.filter(col("HadHeartAttack") == "Yes").count()
no_count  = df.filter(col("HadHeartAttack") == "No").count()

# Ratio: how many "No" cases for every "Yes" case
balancing_ratio = no_count / yes_count

print(f"   Positives (Yes): {yes_count}")
print(f"   Negatives (No):  {no_count}")
print(f"   Balancing Ratio: {balancing_ratio:.2f}")
print(
    "   (This means 1 Heart Attack case is worth "
    "approx {:.2f} healthy cases)".format(balancing_ratio)
)


# 4. Add classWeight Column

# If HadHeartAttack == "Yes" → weight = balancing_ratio
# If HadHeartAttack == "No"  → weight = 1.0

df_weighted = df.withColumn(
    "classWeight",
    when(col("HadHeartAttack") == "Yes", balancing_ratio)
    .otherwise(1.0)
)

#
# 5. Save Weighted Dataset

# Saved separately to avoid mixing with raw data
output_path = "s3://heart-disease-ga2/heart_disease_data_weighted"

print(f"... SAVING WEIGHTED DATA TO: {output_path} ...")

df_weighted.write \
    .mode("overwrite") \
    .parquet(output_path)

print("SUCCESS! Data balanced via weighting.")


# 6. Stop Spark Session
spark.stop()
