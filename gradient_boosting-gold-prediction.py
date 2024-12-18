from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time

start_time = time.time()

spark = SparkSession.builder.appName("GradientBoosting").getOrCreate()

df = spark.read.csv("FINAL_USO.csv", header=True,inferSchema=True)


inputCols =['Low','Open','High','EG_low','EG_close','GDX_High','SP_open','GDX_Low','USDI_Price']

assembler = VectorAssembler(inputCols=inputCols, outputCol="features")  

df = assembler.transform(df)


gbt = GBTRegressor(labelCol="Close", featuresCol="features", maxIter=10)

# Split data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")


param_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [2,5,10]).addGrid(gbt.maxIter, [10]).addGrid(gbt.stepSize, [0.5, 0.1]).build()
# Define the cross-validator
crossval = CrossValidator(estimator=gbt,estimatorParamMaps=param_grid,
evaluator=evaluator, numFolds=5, seed=42)

# Train the model using cross-validation
cv_model = crossval.fit(train_data)
# Make predictions on the testing data
cv_predictions = cv_model.transform(test_data)
# Print RMSE
rmse = evaluator.evaluate(cv_predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="r2")
r_squared = evaluator.evaluate(cv_predictions)
print("R-squared:", r_squared)


bestModel=cv_model.bestModel
bestparams=bestModel.extractParamMap()
for param, value in bestparams.items():

    print(f"{param.name}:{value}")


# After training and evaluating your model
end_time = time.time()
total_running_time = end_time - start_time

print(f"Total running time: {total_running_time} seconds")

# You can also print out the elapsed time per major step if desired
print(f"Time taken for data loading and preprocessing: {(time.time() - start_time) / 60:.2f} minutes")
print(f"Time taken for model training and evaluation: {(time.time() - end_time) / 60:.2f} minutes")
