# Databricks notebook source
# MAGIC %md # Task 3

# COMMAND ----------

# MAGIC %md ## Library and Method Loading
# MAGIC 
# MAGIC Load appropriate libraries, methods and file location constants.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f
import os,sys

# COMMAND ----------

# IN:
EDGES_SMALL_PARQUET = "/FileStore/tables/edges-small.parquet"

# Checkpoint files (to facilitate partial rerun during debugging)
ALL_NODES_SMALL_TEXT = "/FileStore/tables/150454388/nodes-small.txt"
ALL_ADJLIST_SMALL_TEXT =  "/FileStore/tables/150454388/adjlist-small.txt"
SHORTEST_PATH_SMALL =   "/FileStore/tables/150454388/shortest-paths-small.txt"

# OUT . the output you should produce containing all discovered communities (list of lists of graph nodes)
CC_SMALL = "/FileStore/tables/150454388/ConnectedComponents-small.txt"

# COMMAND ----------

# Import modules
from comscan.model.graph import Graph
from comscan.algorithms.connected import connected_components
from comscan.algorithms.connected import number_connected_components
from comscan.distributed_algorithms.betweenness import compute_edge_betweenness
from comscan.distributed_algorithms.shortest_paths import compute_shortest_paths

# COMMAND ----------

def _remove_self_loops(graph):
    for n, neighbors in graph.adj_list.items():
        if n in neighbors:
            graph.remove_edge(n, n)

# COMMAND ----------

def get_top_k_edges_with_highest_betweenness(sc, graph, k):
  shortest_paths_rdd = compute_shortest_paths(sc, graph)
  edge_betweenness_rdd = compute_edge_betweenness(shortest_paths_rdd)
  top_k_betweenness = edge_betweenness_rdd.sortBy(lambda x: x[1], ascending=False).take(k)
  top_k_edges = list(map(lambda x: x[0],top_k_betweenness))
  return top_k_edges

# COMMAND ----------

edges = spark.read.parquet(EDGES_SMALL_PARQUET)
edges.count()

# COMMAND ----------

# MAGIC %md ## Creating Data Structures
# MAGIC ### Nodes
# MAGIC Find nodes by selecting user Ids from edges and creating a distinct union of both ids.

# COMMAND ----------

edges = edges.rdd

# COMMAND ----------

distinctUserIds1 = edges.map(lambda x: x["userId1"])
distinctUserIds2 = edges.map(lambda x: x["userId2"])

# COMMAND ----------

nodes = distinctUserIds1.union(distinctUserIds2).distinct()

# COMMAND ----------

display(edges.collect())

# COMMAND ----------

# MAGIC %md ### Adjacency Lists
# MAGIC 
# MAGIC Create Adjacency list by creating lists for edges from first id direction and the second id, joining the two lists and then aggregating them.

# COMMAND ----------

# Find lists for id1 to id 2 direction and vice versa
id1List = edges.map(lambda x: (x[0], (x[1], x[2])))
id2List = edges.map(lambda x: (x[1], (x[0], x[2])))

# aggregate results
unionList = id1List.union(id2List)
aggList = unionList.aggregateByKey(list(), lambda x, y: x + [y], lambda x, y: x+y) 
adjLists = aggList.map(lambda x: (x[0], dict(x[1])))

# COMMAND ----------

# MAGIC %md ### checkpoint save nodes and lists

# COMMAND ----------

# MAGIC %fs rm -r "dbfs:/FileStore/tables/150454388/nodes-small.txt"

# COMMAND ----------

# MAGIC %fs rm -r "dbfs:/FileStore/tables/150454388/adjlist-small.txt"

# COMMAND ----------

# note this will fail if the files already exist

nodes.saveAsPickleFile(ALL_NODES_SMALL_TEXT)
adjLists.saveAsPickleFile(ALL_ADJLIST_SMALL_TEXT) 

# COMMAND ----------

# MAGIC %md ## checkpoint. load nodes and adjlist from file

# COMMAND ----------

## load up all nodes from file, as a RDD. However the Graph object expects to see a list, thus we use collect() here
nodes_rdd = sc.pickleFile(ALL_NODES_SMALL_TEXT)
nodes = nodes_rdd.collect()

## adjLists is an RDD representing a dictionary of dictionaries (the adjlist). the collectAsMap() method reconstructs the original dictionary
adjLists = sc.pickleFile(ALL_ADJLIST_SMALL_TEXT).collectAsMap()

# COMMAND ----------

# MAGIC %md ### Graph Object

# COMMAND ----------

# MAGIC %md create a `Graph()`   object and add all nodes and adjacency lists

# COMMAND ----------

graph = Graph('user-user-network', nodes, adjLists)

# COMMAND ----------

# MAGIC %md ## Girwan-Newman Implementation

# COMMAND ----------

# MAGIC %md ### Algorithm

# COMMAND ----------

k = 10000
noOfCom = 6

# COMMAND ----------

ncc = number_connected_components(graph)

while ncc < noOfCom:
  edges = get_top_k_edges_with_highest_betweenness(sc, graph, k)
  for edge in edges:
    graph.remove_edge(edge[0], edge[1])
  ncc = number_connected_components(graph)

# COMMAND ----------

# MAGIC %md ### Output

# COMMAND ----------

# MAGIC %md #### Number of Connected Components

# COMMAND ----------

ncc

# COMMAND ----------

# MAGIC %md #### Communities

# COMMAND ----------

communities = [com for com in connected_components(graph)]

# COMMAND ----------

print(communities)

# COMMAND ----------

print([len(l) for l in communities])

# COMMAND ----------

# MAGIC %md ##### Saving the Communities

# COMMAND ----------

# MAGIC %fs rm -r "dbfs:/FileStore/tables/150454388/ConnectedComponents-small.txt"

# COMMAND ----------

sc.parallelize(communities).saveAsPickleFile(SHORTEST_PATH_SMALL)