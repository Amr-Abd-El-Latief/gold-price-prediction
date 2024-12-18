from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
import time

start_time = time.time()

spark = SparkSession.builder.appName("Linear Regression").getOrCreate()

df = spark.read.csv("FINAL_USO.csv", header=True,
inferSchema=True)


inputCols=['Low','Open','High','EG_low','EG_close','GDX_High','SP_open','GDX_Low','USDI_Price']

#'Low','Open','High','EG_low','EG_close','GDX_High','SP_open','GDX_Low','USDI_Price']

assembler = VectorAssembler(inputCols=inputCols,

outputCol="features")

assembler = VectorAssembler(inputCols=['Low','Open','High','EG_low','EG_close','GDX_High','SP_open','GDX_Low','USDI_Price'], outputCol='features')

linearReg = LinearRegression(featuresCol="features",
                            labelCol="Close", 
                            standardization=True,
                            solver='auto',
                            maxIter=100,
                            regParam=0.3,
                            elasticNetParam=0.8)

pipeline = Pipeline(stages=[assembler, linearReg])

train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

LinearR_model = pipeline.fit(train_data)

predictions = LinearR_model.transform(test_data)

print(predictions.select("prediction", "Close").show(20))

evaluator1 = RegressionEvaluator(labelCol="Close",
predictionCol="prediction", metricName="rmse")
rmse = evaluator1.evaluate(predictions)
evaluator_r2 = RegressionEvaluator(labelCol="Close",
predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
evaluator_r3 = RegressionEvaluator(labelCol="Close",
predictionCol="prediction", metricName="mae")
mae = evaluator_r3.evaluate(predictions)

# Print the evaluation metrics
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)
print("Mean Absolute Error (MAE):", mae)



lr=LinearR_model.stages[-1] #you must access the last stage in the trained pipleine model
coefficients = lr.coefficients
intercept = lr.intercept
print("Coefficients: ", coefficients)
print("Intercept: {:.3f}".format(intercept))



print("feature_importance :")

feature_importance = sorted(list(zip(inputCols, map(abs,coefficients))), key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance:

    print(" {}: {:.3f}".format(feature, importance))

end_time = time.time()
total_running_time = end_time - start_time

print(f"Total running time: {total_running_time} seconds")

print(f"Time taken for data loading and preprocessing: {(time.time() - start_time) / 60:.2f} minutes")
print(f"Time taken for model training and evaluation: {(time.time() - end_time) / 60:.2f} minutes")
