from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


# HEART DISEASE — MODEL DEPLOYMENT (BATCH INFERENCE)
# Model: Gradient Boosted Tree (Best Model)
# Platform: Amazon EMR + Spark


# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("HeartDiseaseGBTDeployment") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n" + "=" * 50)
print("HEART DISEASE MODEL DEPLOYMENT (BATCH)")
print("=" * 50)


# 2. Load Trained Spark ML Pipeline Model
print("\n[1] Loading trained model from S3...")

MODEL_PATH = "s3://heart-disease-ga2-gold/models/final_model_gbt"

model = PipelineModel.load(MODEL_PATH)

print("SUCCESS: Model loaded successfully")


# 3. Load Deployment Input Dat
print("\n[2] Loading deployment input data...")

INPUT_PATH = "s3://heart-disease-ga2/heart_disease_data_final"

df_input = spark.read.parquet(INPUT_PATH)

print(f"SUCCESS: Input data loaded ({df_input.count()} rows)")


# 4. Run Batch Inference
print("\n[3] Running batch inference...")

predictions = model.transform(df_input)

print("SUCCESS: Predictions generated")


# 5. Select Relevant Output Columns
# prediction   -> predicted class (0 / 1)
# probability  -> class probabilities

output_df = predictions.select(
    "prediction",
    "probability"
)


# 6. Save Prediction Results to S3
print("\n[4] Saving prediction results...")

OUTPUT_PATH = "s3://heart-disease-ga2-gold/deployment_output/gbt_predictions"

output_df.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)

print(f"SUCCESS: Predictions saved to {OUTPUT_PATH}")


# 7. Stop Spark Session
print("\nDeployment completed successfully.")
print("=" * 50)

spark.stop()
