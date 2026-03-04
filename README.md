## Distributed Machine Learning Pipeline on AWS EMR
Heart Disease Prediction using Apache Spark

---

## Project Overview

This project implements a distributed machine learning pipeline for heart disease prediction using:

- Apache Spark

- Amazon EMR (Elastic MapReduce)

- Amazon S3

- Spark MLlib

The system demonstrates how distributed computing can be applied to scalable data preprocessing, feature engineering, and parallel machine learning training in a cloud environment.

This project was developed as part of a Parallel and Distributed Computing course and structured to reflect real-world cloud-native data engineering practices.

---


## Architecture Overview
Pipeline Flow

- Raw dataset uploaded to Amazon S3

- Distributed data processing using Apache Spark running on Amazon EMR

- Feature engineering and class imbalance handling

- Parallel model training using Spark MLlib

- Trained model stored back in Amazon S3

---


## Dataset

Source: Kaggle

Dataset: Indicators of Heart Disease (2022 Update)

Target Variable: HadHeartAttack

Approximately 920 records and 16 features

Class imbalance:

~85% No Heart Attack

~15% Had Heart Attack

Handled using weighted learning strategy in Spark.

---


## Distributed Processing Workflow
1. Exploratory Data Analysis

- Schema validation

- Missing value check

- Class distribution analysis

2. Class Imbalance Handling

- Calculated class ratio

- Added classWeight column

- Applied weighted learning during model training

3. Feature Selection

- Weighted Random Forest feature importance

- Selected top 15 most important features

- Saved final training dataset in Parquet format

4. Distributed Model Training (10-Fold Cross Validation)

Models implemented:

- Logistic Regression

- Random Forest

- Gradient Boosted Trees

Evaluation metrics:

- Accuracy

- Precision

- Recall

- F1 Score

- ROC-AUC

All training executed in distributed Spark environment on Amazon EMR.

---


## Final Model

- Best Performing Model: Gradient Boosted Trees

- Training Platform: Apache Spark on Amazon EMR

- Model stored as Spark PipelineModel in Amazon S3

- Batch inference supported using Spark job submission.

---


## Optional Extension (Not Implemented)

The architecture can be extended to support:

- Serverless inference using AWS Lambda

- REST API exposure via API Gateway

- Real-time streaming via Kinesis

These were proposed as future scalability enhancements.

---


## Technology Stack

- Python

- PySpark

- Apache Spark

- Spark MLlib

- Amazon S3

- Amazon EMR

- Parquet Storage Format

---


## Key Technical Highlights

- Distributed ETL processing

- Parallelized cross-validation

- Class imbalance strategy using weight column

- Feature importance ranking

- Cloud-based scalable ML architecture

- Separation of raw, processed, and model layers (Bronze/Silver/Gold structure)

---

```bash
aws-emr-distributed-heart-disease-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── 01_eda.py
│   ├── 02_class_balancing.py
│   ├── 03_feature_selection.py
│   ├── 04_logistic_regression_cv.py
│   ├── 05_random_forest_cv.py
│   ├── 06_gbt_cv.py
│   ├── 07_batch_deployment.py
│   └── 08_prediction_results.py
│
└── scripts/
    └── install_numpy.sh
