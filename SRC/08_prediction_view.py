from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round
from pyspark.ml.functions import vector_to_array


# HEART DISEASE — DEPLOYMENT RESULTS (TABLE VIEW)


# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("HeartDiseaseDeploymentTableView") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n" + "=" * 60)
print("HEART DISEASE DEPLOYMENT PREDICTION RESULTS (TABLE)")
print("=" * 60)


# 2. Read Deployment Prediction Output
PRED_PATH = "s3://heart-disease-ga2-gold/deployment_output/gbt_predictions/"

df = spark.read.parquet(PRED_PATH)

print("SUCCESS: Prediction data loaded")


# 3. Convert probability vector → array
df = df.withColumn(
    "prob_array",
    vector_to_array(col("probability"))
)


# 4. Create Clean Tabular Output
df_table = df.select(
    round(col("prob_array")[1], 4).alias("Heart_Attack_Probability"),
    col("prediction").cast("int").alias("Predicted_Class")
)


# 5. Show Table
print("\n--- SAMPLE DEPLOYMENT PREDICTIONS ---")
df_table.show(20, truncate=False)


# 6. Prediction Distribution
print("\n--- PREDICTION DISTRIBUTION ---")
df_table.groupBy("Predicted_Class").count().show()

print("\nTable generated successfully.")
print("=" * 60)

# 7. Stop Spark Session
spark.stop()
