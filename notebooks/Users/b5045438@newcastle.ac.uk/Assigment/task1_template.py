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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ### Graphical and Numerical Summaries
# MAGIC 
# MAGIC #### Ratings Histogram

# COMMAND ----------

display(ratings)

# COMMAND ----------

# MAGIC %md Most ratings range around 3 and 4 and the most common rating seems to be around 4. There seems to be nothing out of ordinary

# COMMAND ----------

# MAGIC %md ### Fields Numerical Summaries

# COMMAND ----------

ratings.describe().show()

# COMMAND ----------

# MAGIC %md Unsurpsingly, ratings average around 3.5 and have a tight standard deviation of 1, again confirming that most ratings are clustered around 3-4 as stated before. When it comes to the other fields, the userids seem to range from 1 to 610 and the movie ids from 1 to 193609, lastly the timestamps start at 828124615 and stop at 1537799250. Nothing seems out of place.

# COMMAND ----------

# MAGIC %md ## Construction of ALS Reccomendation Model

# COMMAND ----------

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# COMMAND ----------

ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "target/tmp/myCollaborativeFilter")
sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

# COMMAND ----------

# Build the recommendation model using Alternating Least Squares based on implicit ratings
model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)