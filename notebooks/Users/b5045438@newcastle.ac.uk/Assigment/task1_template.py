# Databricks notebook source
# MAGIC %md # task: train ALS models on the entire `ratings` dartaset
# MAGIC 
# MAGIC * load ratings.csv file (provided)
# MAGIC * use ratings.csv to train a ALS model to provide recommendations
# MAGIC * evaluate model performance using RSME. Possibly suggest oter metrics, please justifiy if you use other metrics
# MAGIC * use GridSearch or other methods to adjust model Hyperparameters 
# MAGIC * comment on computational cost of optimisation, and the improvements achieved over the baseline model

# COMMAND ----------

# MAGIC %md # Task 1
# MAGIC ## Data Loading
# MAGIC 
# MAGIC Load parquet as an RDD using spark.read.parquet():

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f

# COMMAND ----------

# file location
RATINGS_SMALL_PARQUET = "/FileStore/tables/ratings-small.parquet"

# COMMAND ----------

# load file as a RDD
ratings = spark.read.parquet(RATINGS_SMALL_PARQUET)
ratings.count()

# COMMAND ----------

# MAGIC %md ## Data Exploration 

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
# MAGIC To construct the reccomendation model, an explicit collaborative filter methood which uses ratings directly from the movie lens dataset is needed. However, a problem is that the movie ratings tend to be sparse. Hence, missing ratings need to be filled with a method like Alternating Least Squares which uses matrix factorisation to approximate ratings via optimisation.
# MAGIC 
# MAGIC ### Imports

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %md ### Test Train Split
# MAGIC 
# MAGIC The data will be split into a trainning for the fit and a testing set for the evaulation later. A 80% 20% train and test split is used here as with the rest of the models used for the assignment. 

# COMMAND ----------

(train, test) = ratings.randomSplit([0.80, 0.20], seed = 1) # seed for reproducability 

# COMMAND ----------

# MAGIC %md ### ALS Construction
# MAGIC 
# MAGIC The ALS is constructed by using the userIds to represent users, the movieIds to represent the items and the ratings column to represent the rating to fill.  Since the rating is not negative it, the non negative is True and preferences are not implicit

# COMMAND ----------

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
          nonnegative = True, implicitPrefs = False)

# COMMAND ----------

# Helps prevent overflows
sc.setCheckpointDir('/checkpoint')
ALS.checkpointInterval = 2

# COMMAND ----------

# MAGIC %md ### Tuning Hyperparameters using Cross Validation 
# MAGIC 
# MAGIC By using k fold cross validation, we can compare ALS models using different hyper parameters to pick the ideal hyperparameters. For ALS, the parameters we will tune will be:
# MAGIC * rank: Number of features to discover
# MAGIC * maxIter: the maximum number of iterations the algorithm is allowed to run
# MAGIC * regParam: regularization parameter to limit overfitting

# COMMAND ----------

# MAGIC %md #### Create Grid

# COMMAND ----------

param_grid = ParamGridBuilder() \
           .addGrid(als.rank, [10, 85]) \
           .addGrid(als.maxIter, [5, 55]) \
           .addGrid(als.regParam, [.01, .125]) \
           .build()

# COMMAND ----------

# MAGIC %md #### Create RMSE Evaluator

# COMMAND ----------

# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse",
labelCol="rating", predictionCol="prediction")

# COMMAND ----------

# MAGIC %md #### Cross Validation and Best Model

# COMMAND ----------

# create 5 fold CV using ALS and created grid
cv = CrossValidator(
  estimator= als, estimatorParamMaps=param_grid,
  evaluator=evaluator, numFolds=5)

model = cv.fit(train)
best_model = model.bestModel

# COMMAND ----------

best_model.getRank()

# COMMAND ----------

best_model.getMaxIter()

# COMMAND ----------

best_model.regParam()