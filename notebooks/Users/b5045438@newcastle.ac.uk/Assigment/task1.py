# Databricks notebook source
# MAGIC %md # Task 1
# MAGIC ## Data Loading
# MAGIC 
# MAGIC Load parquet as an RDD using spark.read.parquet():

# COMMAND ----------

# file locations
RATINGS_SMALL_PARQUET = "/FileStore/tables/ratings-small.parquet"

# COMMAND ----------

# load file as a RDD
ratings = spark.read.parquet(RATINGS_SMALL_PARQUET)
ratings.count()

# COMMAND ----------

# MAGIC %md ## Data Exploration and Sanity Checks

# COMMAND ----------

# MAGIC %md ### Graphical and Numerical Summaries
# MAGIC 
# MAGIC #### Ratings Histogram

# COMMAND ----------

display(ratings)

# COMMAND ----------

# MAGIC %md Most ratings range around 3 and 4 and the most common rating seems to be around 4. There seems to be nothing out of ordinary

# COMMAND ----------

# MAGIC %md #### Fields Numerical Summaries

# COMMAND ----------

ratings.describe().show()

# COMMAND ----------

# MAGIC %md Unsurpsingly, ratings average around 3.5 and have a tight standard deviation of 1, again confirming that most ratings are clustered around 3-4 as stated before. When it comes to the other fields, the userids seem to range from 1 to 610 and the movie ids from 1 to 193609, lastly the timestamps start at 828124615 and stop at 1537799250. Nothing seems out of place.

# COMMAND ----------

# MAGIC %md ## Construction of ALS Reccomendation Model
# MAGIC 
# MAGIC To construct the reccomendation model, an explicit collaborative filter method which uses ratings derived from the movie lens dataset is needed. However, a problem is that the movie ratings tend to be sparse. Hence, missing ratings need to be filled with a method like Alternating Least Squares which uses matrix factorisation to approximate ratings via optimisation.
# MAGIC 
# MAGIC ### Imports

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %md ### Data Cleaning
# MAGIC 
# MAGIC Drop timestamps as it will not be useful for the rest of the analysis.

# COMMAND ----------

ratings = ratings.drop('timestamp')

# COMMAND ----------

# MAGIC %md ### Test Train Split
# MAGIC 
# MAGIC The data will be split into a trainning for the fit and a testing set for the evaulation later. A 50% 50% train and test split is used here as with the rest of the models used for the assignment. 

# COMMAND ----------

(train, test) = ratings.randomSplit([0.50, 0.50], seed = 1234) # seed for reproducability 

# COMMAND ----------

# MAGIC %md ### ALS Construction
# MAGIC 
# MAGIC The ALS is constructed by using the userIds to represent users, the movieIds to represent the items and the ratings column to represent the rating to fill.  Since the rating is not negative it, the non negative is True and preferences are not implicit. Also, a cold start strategy of drop is used to resolve the cold start problem (users with no reviews).

# COMMAND ----------

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
          nonnegative = True, implicitPrefs = False,
          coldStartStrategy = 'drop')

# COMMAND ----------

# MAGIC %md ### Tuning Hyperparameters using Cross Validation 
# MAGIC 
# MAGIC By using k fold cross validation, we can compare ALS models using different hyper parameters to pick the ideal hyperparameters. For ALS, the parameters we will tune will be the rank which refers to the number of features to discover throughout the run.

# COMMAND ----------

# MAGIC %md #### Create Grid

# COMMAND ----------

paramGrid = ParamGridBuilder() \
           .addGrid(als.rank, [10, 5, 1]) \
           .build()

# COMMAND ----------

# MAGIC %md #### Create RMSE Evaluator

# COMMAND ----------

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# COMMAND ----------

# MAGIC %md #### Cross Validation and Best Model

# COMMAND ----------

cv = CrossValidator(
  estimator= als, estimatorParamMaps=paramGrid,
  evaluator=evaluator, numFolds=3)

# Run cross-validation, and choose the best set of parameters.
model = cv.fit(train)
bestModel = model.bestModel

# COMMAND ----------

bestModel.rank

# COMMAND ----------

# MAGIC %md #### Best Model Results

# COMMAND ----------

predictions = bestModel.transform(test)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md The model seems to quite well here considering the sparsity of the matrix, at least according to the root mean square error. However, perhaps the algorithm can be improved by feeding it communities of users instead of the whole users datasets, where users of said communities are likely to have similar preferneces and hence predicting ratings would be easier. This will be done using Girwan Newman in the remaining tasks.