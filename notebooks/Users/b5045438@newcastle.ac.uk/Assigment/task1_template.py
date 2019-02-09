# Databricks notebook source
# MAGIC %md # task: train ALS models on the entire `ratings` dartaset
# MAGIC 
# MAGIC * load ratings.csv file (provided)
# MAGIC * use ratings.csv to train a ALS model to provide recommendations
# MAGIC * evaluate model performance using RSME. Possibly suggest oter metrics, please justifiy if you use other metrics
# MAGIC * use GridSearch or other methods to adjust model Hyperparameters 
# MAGIC * comment on computational cost of optimisation, and the improvements achieved over the baseline model

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f

# COMMAND ----------

#IN 
RATINGS_SMALL_PARQUET = "/FileStore/tables/ratings-small.parquet"

# RATINGS_100K_PARQUET = "/FileStore/tables/ratings-small.parquet"  # DO NOT USE

# COMMAND ----------

ratings = spark.read.parquet(RATINGS_SMALL_PARQUET)
ratings.count()

# COMMAND ----------

# MAGIC %md ### here you write your ALS model training and evalutation code

# COMMAND ----------

