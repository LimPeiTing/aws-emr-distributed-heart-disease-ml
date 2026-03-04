from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# HEART DISEASE — WEIGHTED FEATURE RANKING (SPARK)


# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("HeartWeightedRanking") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# 2. Read Weighted Dataset
print("... READING WEIGHTED DATA ...")

input_path = "s3://heart-disease-ga2/heart_disease_data_weighted"
df = spark.read.parquet(input_path)


# 3. Prepare Columns
target_col = "HadHeartAttack"

# Columns to exclude from features
ignore_cols = [target_col, "classWeight"]

# Identify string and numeric columns automatically
str_cols = [
    c.name for c in df.schema.fields
    if c.dataType.typeName() == "string" and c.name not in ignore_cols
]

num_cols = [
    c.name for c in df.schema.fields
    if c.dataType.typeName() != "string" and c.name not in ignore_cols
]

stages = []
inputs_for_model = num_cols[:]  # start with numeric columns


# 4. Index String Feature Columns
for col_name in str_cols:
    indexer = StringIndexer(
        inputCol=col_name,
        outputCol=f"{col_name}_idx",
        handleInvalid="skip"
    )
    stages.append(indexer)
    inputs_for_model.append(f"{col_name}_idx")


# 5. Index Target Column
label_indexer = StringIndexer(
    inputCol=target_col,
    outputCol="label"
)
stages.append(label_indexer)


# 6. Assemble Feature Vector
assembler = VectorAssembler(
    inputCols=inputs_for_model,
    outputCol="features"
)
stages.append(assembler)


# 7. Train Weighted Random Forest Model
print("... TRAINING MODEL (WITH WEIGHTS) ...")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=20,
    maxDepth=5,
    seed=42,
    maxBins=64
)

stages.append(rf)

pipeline = Pipeline(stages=stages)
model = pipeline.fit(df)


# 8. Extract Feature Importance Rankings
rf_model = model.stages[-1]
importances = rf_model.featureImportances

scores = []

for i, feature_name in enumerate(inputs_for_model):
    score = importances[i]
    clean_name = feature_name.replace("_idx", "")
    scores.append((clean_name, score))

# Sort features by importance score (descending)
scores.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 40)
print("   WEIGHTED FEATURE IMPORTANCE (The Truth)")
print("=" * 40)
print(f"{'RANK':<5} | {'FEATURE':<25} | {'SCORE'}")
print("-" * 45)

top_15_names = []

# Print top 20 features, save top 15
for i in range(min(20, len(scores))):
    name, score = scores[i]
    print(f"{i + 1:<5} | {name:<25} | {score:.4f}")
    if i < 15:
        top_15_names.append(name)

# 9. Save Final Dataset (Target + Top 15 + Weight)
print("\n... SAVING FINAL DATASET FOR TRAINING ...")

final_cols = [target_col, "classWeight"] + top_15_names
df_final = df.select(final_cols)

final_path = "s3://heart-disease-ga2-silver/heart_disease_data_final"

df_final.write \
    .mode("overwrite") \
    .parquet(final_path)

print(f"SUCCESS! Top 15 features saved to: {final_path}")
print("Your Top Features:", top_15_names)


# 10. Stop Spark Session
spark.stop()
