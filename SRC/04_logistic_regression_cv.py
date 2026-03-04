from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)


# HEART DISEASE — LOGISTIC REGRESSION WITH 10-FOLD CV


# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("HeartDiseaseLR_CV") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# 2. Read Final Prepared Dataset
print("... READING FINAL DATASET ...")

input_path = "s3://heart-disease-ga2-silver/heart_disease_data_final"
df = spark.read.parquet(input_path)


# 3. Setup Pipeline Columns
target_col = "HadHeartAttack"
weight_col = "classWeight"
ignore_cols = [target_col, weight_col]

# Identify string and numeric feature columns
str_cols = [
    c.name for c in df.schema.fields
    if c.dataType.typeName() == "string" and c.name not in ignore_cols
]

num_cols = [
    c.name for c in df.schema.fields
    if c.dataType.typeName() != "string" and c.name not in ignore_cols
]

stages = []
assembler_inputs = num_cols[:]


# 4. Index Categorical Features
for col_name in str_cols:
    indexer = StringIndexer(
        inputCol=col_name,
        outputCol=f"{col_name}_idx",
        handleInvalid="keep"
    )
    stages.append(indexer)
    assembler_inputs.append(f"{col_name}_idx")


# 5. Index Target Label
label_indexer = StringIndexer(
    inputCol=target_col,
    outputCol="label"
)
stages.append(label_indexer)


# 6. Assemble Feature Vector
assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features"
)
stages.append(assembler)


# 7. Configure Logistic Regression Classifier
print("... CONFIGURING LOGISTIC REGRESSION ...")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxIter=50,
    regParam=0.01,        # L2 regularization
    elasticNetParam=0.0  # 0 = Ridge, 1 = Lasso
)

stages.append(lr)

pipeline = Pipeline(stages=stages)


# 8. Manual 10-Fold Cross-Validation
print("\n" + "=" * 50)
print("STARTING 10-FOLD CV (Logistic Regression)")
print("=" * 50)

folds = df.randomSplit([0.1] * 10, seed=42)

scores = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "AUC": []
}

# Evaluators
eval_auc = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)
eval_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    metricName="accuracy"
)
eval_prec = MulticlassClassificationEvaluator(
    labelCol="label",
    metricName="weightedPrecision"
)
eval_rec = MulticlassClassificationEvaluator(
    labelCol="label",
    metricName="weightedRecall"
)
eval_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    metricName="f1"
)

# Cross-validation loop
for i in range(10):
    print(f"Running Fold {i + 1}/10 ...")

    test_data = folds[i]
    train_folds = [folds[j] for j in range(10) if j != i]

    train_data = train_folds[0]
    for k in range(1, 9):
        train_data = train_data.union(train_folds[k])

    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    scores["AUC"].append(eval_auc.evaluate(predictions))
    scores["Accuracy"].append(eval_acc.evaluate(predictions))
    scores["Precision"].append(eval_prec.evaluate(predictions))
    scores["Recall"].append(eval_rec.evaluate(predictions))
    scores["F1"].append(eval_f1.evaluate(predictions))


# 9. Print Average CV Results
print("\n" + "=" * 50)
print("   LOGISTIC REGRESSION 10-FOLD RESULTS")
print("=" * 50)

print(f"{'METRIC':<15} | {'PERCENTAGE':<15}")
print("-" * 35)
print(f"{'Accuracy':<15} | {sum(scores['Accuracy']) / 10 * 100:.2f}%")
print(f"{'Precision':<15} | {sum(scores['Precision']) / 10 * 100:.2f}%")
print(f"{'Recall':<15} | {sum(scores['Recall']) / 10 * 100:.2f}%")
print(f"{'F1 Score':<15} | {sum(scores['F1']) / 10 * 100:.2f}%")
print(f"{'ROC AUC':<15} | {sum(scores['AUC']) / 10 * 100:.2f}%")
print("=" * 50)


# 10. Train Final Model on Full Dataset & Save
print("\n... Saving Final Logistic Regression Model ...")

final_model = pipeline.fit(df)

output_path = "s3://heart-disease-ga2-gold/models/final_model_lr"
final_model.write().overwrite().save(output_path)

print(f"Saved to: {output_path}")


# Stop Spark Session
spark.stop()
