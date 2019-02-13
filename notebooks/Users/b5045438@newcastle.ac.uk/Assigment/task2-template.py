# Databricks notebook source
# MAGIC %md # task: create a user-user network from the raw `ratings` dataset
# MAGIC 
# MAGIC * load ratings file (code provided)
# MAGIC * generate a set of edges of the form `[(u,v,w)]` where each edge represents the fact that users `u`,`v` have both rated the same `w` movies (ie if they have both rated movie1, movie2, the edge weigth will be 2)
# MAGIC  * you should have 164054 edges
# MAGIC * save the set of edges to a file as indicated below. **Make sure you name the file using your `<id>` as indicated below, to avoid conflicts with other students within the file workspace**

# COMMAND ----------

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

# MAGIC %md ### count number of distinct users  -- this is the number of unique userId that you will have in your adjacency list:

# COMMAND ----------

ratings.agg(f.countDistinct('userId')).show()

# COMMAND ----------

# MAGIC %md ## Finding Edges

# COMMAND ----------

# MAGIC %md #add your code here. 
# MAGIC 
# MAGIC you need to construct a RDD consisting of edges of the form:
# MAGIC 
# MAGIC `    [(source_node, target_node, weight)]`
# MAGIC     
# MAGIC     for example: `[(1,6,1), (1,8,1),(2,3,1), (2,4,1)]`
# MAGIC     
# MAGIC please name the RDD with edges `weightedEdges`

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

j_df = df1.join(df2, on = ["movieId"], how = "outer")

# COMMAND ----------

j_df.count()

# COMMAND ----------

# restrict direction
j_df = j_df[(j_df['userId1'] < j_df['userId2'])]

# COMMAND ----------

display(j_df)

# COMMAND ----------

j_df.count()

# COMMAND ----------

# MAGIC %md ### Finding Weights 
# MAGIC 
# MAGIC Now that we have all pairs of users that watch the same movie, we can group by the ids and count the occurances to get the weights (number of shared movie ratings).

# COMMAND ----------

weightedEdges= j_df.groupby(df1['userId1'], df2['userId2']).count()

# COMMAND ----------

weightedEdges.count()

# COMMAND ----------

# MAGIC %md ### Save Weights

# COMMAND ----------

weightedEdges.withColumnRenamed('count','weight').write.parquet(EDGES_SMALL_PARQUET, mode="overwrite")