from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time

start_time = time.time()

spark = SparkSession.builder.appName("Random forest Model").getOrCreate()

df = spark.read.csv("FINAL_USO.csv", header=True,inferSchema=True)

#inputCols =['Low','Open','High','EG_low','EG_close','GDX_High','SP_open','GDX_Low','USDI_Price']

inputCols = [ 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume', 'DJ_open', 'DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume', 'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume', 'EU_Price', 'EU_open', 'EU_high', 'EU_low', 'EU_Trend', 'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend', 'OS_Price', 'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend', 'SF_Price', 'SF_Open', 'SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Open', 'USB_High', 'USB_Low', 'USB_Trend', 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low', 'PLT_Trend', 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low', 'PLD_Trend', 'RHO_PRICE', 'USDI_Price', 'USDI_Open', 'USDI_High', 'USDI_Low', 'USDI_Volume', 'USDI_Trend', 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume', 'USO_Open', 'USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume']


assembler = VectorAssembler(inputCols=inputCols, outputCol="features")

df = assembler.transform(df)
print('df after assempling: ')

print(df.show(10))

rf = RandomForestRegressor(featuresCol="features", labelCol="Close", numTrees=100) # Split data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
pipeline = Pipeline(stages=[assembler, rf])

paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [45]).addGrid(rf.maxDepth, [15]).build()

evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")

crossval = CrossValidator(estimator=rf, 
                         estimatorParamMaps=paramGrid, 
                         evaluator=evaluator, 
                         numFolds=5) 


cv_model = crossval.fit(train_data)
predictions = cv_model.transform(test_data)


bestModel = cv_model.bestModel

predictions = bestModel.transform(test_data)


rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE): %g" % rmse)

evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="r2")
r_squared = evaluator.evaluate(predictions)
print("R-squared:", r_squared)


feature_importances = bestModel.featureImportances
print("Feature Importances:", feature_importances)


end_time = time.time()
total_running_time = end_time - start_time

print(f"Total running time: {total_running_time} seconds")

# You can also print out the elapsed time per major step if desired
print(f"Time taken for data loading and preprocessing: {(time.time() - start_time) / 60:.2f} minutes")
print(f"Time taken for model training and evaluation: {(time.time() - end_time) / 60:.2f} minutes")
