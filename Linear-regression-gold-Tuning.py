from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import time

start_time = time.time()

spark = SparkSession.builder.appName("Linear Regression").getOrCreate()

df = spark.read.csv("FINAL_USO.csv", header=True, inferSchema=True)

#inputCols =['Low','Open','High','EG_low','EG_close','GDX_High','SP_open','GDX_Low','USDI_Price']
inputCols = [ 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume', 'DJ_open', 'DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume', 'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume', 'EU_Price', 'EU_open', 'EU_high', 'EU_low', 'EU_Trend', 'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend', 'OS_Price', 'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend', 'SF_Price', 'SF_Open', 'SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Open', 'USB_High', 'USB_Low', 'USB_Trend', 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low', 'PLT_Trend', 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low', 'PLD_Trend', 'RHO_PRICE', 'USDI_Price', 'USDI_Open', 'USDI_High', 'USDI_Low', 'USDI_Volume', 'USDI_Trend', 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume', 'USO_Open', 'USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume']
assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
  
   
assembler = VectorAssembler(inputCols=inputCols, outputCol='features')

linearReg = LinearRegression(featuresCol="features",
                            labelCol="Close", 
                            standardization=True,
                            solver='auto',
                            maxIter=100)

pipeline = Pipeline(stages=[assembler, linearReg])

param_grid = ParamGridBuilder() \
    .addGrid(linearReg.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(linearReg.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(linearReg.maxIter, [10, 50, 100]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=RegressionEvaluator(labelCol="Close"),
    numFolds=5
)

cv_model = crossval.fit(df)

best_model = cv_model.bestModel

predictions = best_model.transform(df)

evaluator1 = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = evaluator1.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

evaluator_r3 = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="mae")
mae = evaluator_r3.evaluate(predictions)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)
print("Mean Absolute Error (MAE):", mae)

lr = best_model.stages[-1]
coefficients = lr.coefficients
intercept = lr.intercept

print("Coefficients:", coefficients)
print("Intercept: {:.3f}".format(intercept))
   
feature_importance = sorted(list(zip(df.columns, map(abs, coefficients))), key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance:
    print(f"{feature}: {importance:.3f}")


accuracy = evaluator1.evaluate(predictions)

best_model = cv_model.bestModel

print(f"\nbest model parameters are:\n")

bestparams=best_model.stages[-1].extractParamMap()

for param, value in bestparams.items():

    print(f"{param.name}:{value}")

end_time = time.time()
total_running_time = end_time - start_time

print(f"Total running time: {total_running_time} seconds")

print(f"Time taken for data loading and preprocessing: {(time.time() - start_time) / 60:.2f} minutes")
print(f"Time taken for model training and evaluation: {(time.time() - end_time) / 60:.2f} minutes")
