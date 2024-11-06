"""
==============================================================================
    Project Title   : Cardiovascular Disease Prediction using Spark
    Author          : Devon Midkiff
    Class           : SAT 5165
    Version         : 1.3
    Date Created    : 10/15/2024
    Date Modified   : 11/05/2024
    Description     : This project performs preprocessing and Naive Bayes analysis
                      to predict the likelihood of cardiovascular disease.
                      The project uses Spark for distributed processing.
==============================================================================
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F

# Start timer for performance comparison
start_time = time.time()

# Initializing Spark session
spark = SparkSession.builder \
    .appName("Cardiovascular Disease Analysis") \
    .config("spark.driver.bindAddress", "192.168.13.140") \
    .config("spark.driver.port", "7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")  # Setting error level to avoid cluttering the console

# Load dataset
df = spark.read.csv("/mnt/shared_folder/CVD_cleaned.csv", header=True, inferSchema=True)

# Convert categorical features to ordinal values for machine learning
df = df.withColumn('Checkup', 
    when(col('Checkup') == 'Within the past year', 1)
    .when(col('Checkup') == 'Within the past 2 years', 2)
    .when(col('Checkup') == 'Within the past 5 years', 3)
    .when(col('Checkup') == '5 or more years ago', 4)
    .otherwise(5))

df = df.withColumn('General_Health', 
    when(col('General_Health') == 'Excellent', 1)
    .when(col('General_Health') == 'Very Good', 2)
    .when(col('General_Health') == 'Good', 3)
    .when(col('General_Health') == 'Fair', 4)
    .otherwise(5))

df = df.withColumn('Age_Category', 
    when(col('Age_Category') == '18-24', 1)
    .when(col('Age_Category') == '25-29', 2)
    .when(col('Age_Category') == '30-34', 3)
    .when(col('Age_Category') == '35-39', 4)
    .when(col('Age_Category') == '40-44', 5)
    .when(col('Age_Category') == '45-49', 6)
    .when(col('Age_Category') == '50-54', 7)
    .when(col('Age_Category') == '55-59', 8)
    .when(col('Age_Category') == '60-64', 9)
    .when(col('Age_Category') == '65-69', 10)
    .when(col('Age_Category') == '70-74', 11)
    .when(col('Age_Category') == '75-79', 12)
    .when(col('Age_Category') == '80+', 13)
    .otherwise(None))

# Convert boolean features to integers
boolean_cols = [
    'Exercise', 'Heart_Disease', 'Skin_Cancer', 
    'Other_Cancer', 'Depression', 'Diabetes', 
    'Arthritis', 'Smoking_History', 'Sex'
]

for col_name in boolean_cols:
    if col_name == 'Sex':
        df = df.withColumn(col_name, when(col(col_name) == 'Male', 1).otherwise(0))
    else:
        df = df.withColumn(col_name, when(col(col_name) == 'Yes', 1).otherwise(0))

# Define columns to be scaled
scale_cols = [
    'Alcohol_Consumption', 'Fruit_Consumption', 
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption',
    'Height_(cm)', 'Weight_(kg)', 'BMI'
]

# Scaling numerical values to help make data less biased
assembler = VectorAssembler(inputCols=scale_cols, outputCol="features_to_scale")
df_scaled = assembler.transform(df)

# Create MinMaxScaler
scaler = MinMaxScaler(inputCol="features_to_scale", outputCol="scaled_features")

# Fit and transform the data
scaler_model = scaler.fit(df_scaled)
df = scaler_model.transform(df_scaled)
df = df.repartition(4)

# Correlation Analysis between features that are normally considered connected to CVD
features_to_correlate = ['BMI', 'Alcohol_Consumption', 'Smoking_History', 'Age_Category']
correlation_matrix = {}
for feature in features_to_correlate:
    correlation = df.stat.corr(feature, "Heart_Disease")
    correlation_matrix[feature] = correlation
    print(f"Correlation between {feature} and Heart_Disease: {correlation}")

# Feature assembly
assembler = VectorAssembler(inputCols=features_to_correlate, outputCol="features")
df = assembler.transform(df)

# Train/test split
train, test = df.randomSplit([0.8, 0.2], seed=1234)

# Initialize Naive Bayes model
nb = NaiveBayes(labelCol="Heart_Disease", featuresCol="features", modelType="multinomial")

# Train the model
nb_model = nb.fit(train)

# Make predictions
predictions = nb_model.transform(test)

# Ensure predictions and labels are DoubleType for MulticlassMetrics
predictions = predictions.withColumn("label", col("Heart_Disease").cast("double"))
predictions = predictions.withColumn("prediction", col("prediction").cast("double"))

# Evaluate the model for AUC
evaluator = BinaryClassificationEvaluator(labelCol="Heart_Disease", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Naive Bayes Model AUC (Area Under ROC): {auc}")

# Calculate additional metrics: recall, precision, specificity, and accuracy
prediction_and_labels = predictions.select("prediction", "label").rdd
metrics = MulticlassMetrics(prediction_and_labels)

recall = metrics.recall(1.0)  # Sensitivity / True Positive Rate
precision = metrics.precision(1.0)  # Precision
specificity = metrics.recall(0.0)  # True Negative Rate
accuracy = metrics.accuracy  # Overall accuracy

print(f"Naive Bayes Model Recall (Sensitivity): {recall}")
print(f"Naive Bayes Model Precision: {precision}")
print(f"Naive Bayes Model Specificity: {specificity}")
print(f"Naive Bayes Model Accuracy: {accuracy}")

# End timer and calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time} seconds")

# Stop the Spark session
spark.stop()

