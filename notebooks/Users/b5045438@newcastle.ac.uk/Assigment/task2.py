# Databricks notebook source
# MAGIC %md ## Library and Data Loading
# MAGIC 
# MAGIC Load appropriate libraries and ratings file. Also create constant for edges file location.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import Row
from operator import add
from collections import Counter

# COMMAND ----------

# IN File locations
RATINGS_SMALL_PARQUET = "/FileStore/tables/ratings-small.parquet"

# OUT file locations
EDGES_SMALL_PARQUET = "/FileStore/tables/150454388/edges-small.parquet"

# COMMAND ----------

ratings = spark.read.parquet(RATINGS_SMALL_PARQUET)
display(ratings)

# COMMAND ----------

ratings.count()

# COMMAND ----------

# MAGIC %md ### count number of distinct users  

# COMMAND ----------

ratings.agg(f.countDistinct('userId')).show()

# COMMAND ----------

# MAGIC %md ## Finding Edges

# COMMAND ----------

display(ratings)

# COMMAND ----------

# MAGIC %md ### Dropping Ratings and Timestamps
# MAGIC 
# MAGIC These two fields are not required since the edges will just record the nodes between the edges and weighting.

# COMMAND ----------

ratings = ratings.drop('timestamp')
ratings = ratings.drop('rating')

# COMMAND ----------

# MAGIC %md ### Create Dataframes
# MAGIC 
# MAGIC Two dataframes will be used to help divide the task up, each representing one of the user ids in the edge

# COMMAND ----------

df1 = ratings.select('userId', 'movieId')
df2 = df1

# COMMAND ----------

df2.count()

# COMMAND ----------

df1.count()

# COMMAND ----------

# MAGIC %md ### Renaming Rows 
# MAGIC 
# MAGIC Rename userId rows before joining the two dataframes to represent the first user id (userId1) and the second user id (userId2) in the edges.

# COMMAND ----------

df1 = df1.withColumnRenamed("userId", "userId1")
df2 = df2.withColumnRenamed("userId", "userId2")

# COMMAND ----------

display(df1)

# COMMAND ----------

display(df2)

# COMMAND ----------

# MAGIC %md ### Joining Second Dataframe with RDD
# MAGIC 
# MAGIC Using an outer join, connect the two dataframes based on movieId to connect users that rated the same movie. Make sure to restrict direction to avoid duplicates.

# COMMAND ----------

jDf = df1.join(df2, on = ["movieId"], how = "outer")

# COMMAND ----------

jDf.count()

# COMMAND ----------

# restrict direction
jDf = jDf[(jDf['userId1'] < jDf['userId2'])]

# COMMAND ----------

display(jDf)

# COMMAND ----------

jDf.count()

# COMMAND ----------

# MAGIC %md ### Finding Weights 
# MAGIC 
# MAGIC Now that we have all pairs of users that watch the same movie, we can group by the ids and count the occurances to get the weights (number of shared movie ratings).

# COMMAND ----------

weightedEdges= jDf.groupby(df1['userId1'], df2['userId2']).count()

# COMMAND ----------

weightedEdges.count()

# COMMAND ----------

# MAGIC %md ### Save Weights

# COMMAND ----------

weightedEdges.withColumnRenamed('count','weight').write.parquet(EDGES_SMALL_PARQUET, mode="overwrite")