from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
import os
from dotenv import load_dotenv
# Assuming you have already loaded your dataset into a Spark DataFrame called 'df'
# and that it has columns like 'date', 'open_price', 'high_price', 'low_price', 'Close'


# spark = SparkSession.builder.appName("Linear Regression").getOrCreate()
# Load environment variables from .env file
load_dotenv()

# Fetch configurations from .env
SPARK_MASTER_HOST = os.getenv("SPARK_MASTER_HOST", "localhost")
SPARK_MASTER_PORT = os.getenv("SPARK_MASTER_PORT", "7077")
APP_NAME = os.getenv("SPARK_APP_NAME", "PySpark Training App")

# Initialize Spark session
spark = SparkSession.builder \
    .appName(APP_NAME) \
    .master(f"spark://{SPARK_MASTER_HOST}:{SPARK_MASTER_PORT}") \
    .getOrCreate()
df = spark.read.csv("/opt/spark/code/FINAL_USO.csv", header=True,
inferSchema=True)


columns=df.columns
del columns[-1]
assembler = VectorAssembler(inputCols=columns,

outputCol="features")
# Step 1: Prepare data for linear regression
inputCols=['Open', 'High', 'Low']

assembler = VectorAssembler(inputCols=['Open', 'High', 'Low'], outputCol='features')

# Step 2: Create Linear Regression model
linearReg = LinearRegression(featuresCol="features",
                            labelCol="Close", 
                            standardization=True,
                            solver='auto',
                            maxIter=100,
                            regParam=0.3,
                            elasticNetParam=0.8)

# Step 3: Create Pipeline with assembler and linear regression model
pipeline = Pipeline(stages=[assembler, linearReg])

# Step 4: Split data into training and test sets (if not already done)
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Step 5: Fit the pipeline to the training data
LinearR_model = pipeline.fit(train_data)

# Step 6: Make predictions on the test set
predictions = LinearR_model.transform(test_data)

# Step 7: Print out the first few predictions
print(predictions.select("prediction", "Close").show(20))

# Step 8: Evaluate the model






evaluator1 = RegressionEvaluator(labelCol="Close",
predictionCol="prediction", metricName="rmse")
rmse = evaluator1.evaluate(predictions)
evaluator_r2 = RegressionEvaluator(labelCol="Close",
predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
evaluator_r3 = RegressionEvaluator(labelCol="Close",
predictionCol="prediction", metricName="mae")
mae = evaluator_r3.evaluate(predictions)





lr=LinearR_model.stages[-1] #you must access the last stage in the trained pipleine model
coefficients = lr.coefficients
intercept = lr.intercept
print("Coefficients: ", coefficients)
print("Intercept: {:.3f}".format(intercept))
print("RMSE: ", rmse)
print("R2: ", r2)
print("MAE: ", mae)




feature_importance = sorted(list(zip(df.columns, map(abs,coefficients))), key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance:

    print(" {}: {:.3f}".format(feature, importance))



