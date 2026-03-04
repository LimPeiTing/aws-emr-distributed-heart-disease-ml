from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count


# HEART DISEASE DATASET — DETAILED EDA (SPARK)


# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("HeartDiseaseDetailedEDA") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n" + "=" * 40)
print("       HEART DISEASE DATASET AUDIT")
print("=" * 40)

# 2. Read Raw Dataset from S3
raw_path = "s3://heart-disease-ga2/raw/heart_2022_no_nans.csv"

df = spark.read.csv(
    raw_path,
    header=True,
    inferSchema=True
)


# A. DATASET SHAPE
num_rows = df.count()
num_cols = len(df.columns)

print(f"\n[A] DATASET SHAPE")
print(f"Number of rows:    {num_rows}")
print(f"Number of columns: {num_cols}")

# B. ATTRIBUTE DATA FORMAT
print(f"\n[B] ATTRIBUTE DATA FORMAT")
df.printSchema()


# C. MISSING DATA CHECK
print(f"\n[C] MISSING DATA CHECK")
print("Checking for nulls or NaNs (this may take a moment)...")

select_exprs = [
    count(
        when(isnan(c) | col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
]

missing_df = df.select(select_exprs)

found_missing = False

for col_name in df.columns:
    missing_count = missing_df.select(col_name).first()[0]
    if missing_count > 0:
        found_missing = True
        print(f"  - {col_name}: {missing_count} MISSING")

if not found_missing:
    print("  SUCCESS: No missing values found in any column!")


# D. CLASS BALANCE CHECK (TARGET VARIABLE)
print(f"\n[D] CLASS BALANCE CHECK (HadHeartAttack)")

if "HadHeartAttack" in df.columns:

    class_counts = (
        df.groupBy("HadHeartAttack")
          .count()
          .orderBy("count")
    )

    rows = class_counts.collect()

    print(f"Total Rows: {num_rows}")
    print("Distribution:")

    for row in rows:
        label = row["HadHeartAttack"]
        count_val = row["count"]
        percentage = (count_val / num_rows) * 100
        print(
            f"  - Class '{label}': "
            f"{count_val} rows ({percentage:.2f}%)"
        )

    # Imbalance warning
    minority_pct = (rows[0]["count"] / num_rows) * 100

    if minority_pct < 15:
        print(
            f"\nWARNING: Severe imbalance detected "
            f"({minority_pct:.2f}%)."
        )
        print(
            "   Recommendation: Use class weighting "
            "during model training."
        )
    else:
        print("\nBalance looks acceptable.")

else:
    print(
        "ERROR: Column 'HadHeartAttack' not found. "
        "Cannot calculate class balance."
    )


# End of EDA
print("\n" + "=" * 40)

spark.stop()
