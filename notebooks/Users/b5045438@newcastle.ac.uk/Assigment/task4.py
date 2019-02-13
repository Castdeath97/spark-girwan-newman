# Databricks notebook source
# MAGIC %md #Task 4
# MAGIC ## Data Loading

# COMMAND ----------

ratings = spark.read.parquet(RATINGS_SMALL_PARQUET)
nodes_rdd = sc.pickleFile(ALL_NODES_SMALL_TEXT)
nodes = nodes_rdd.collect()
adjLists = sc.pickleFile(ALL_ADJLIST_SMALL_TEXT).collectAsMap()
edges = spark.read.parquet(EDGES_SMALL_PARQUET)
communities = sc.pickleFile(CC_SMALL).collectAsMap()