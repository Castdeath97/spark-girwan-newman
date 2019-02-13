# Databricks notebook source
# MAGIC %md # Task: discover user communities in the user-user network
# MAGIC 
# MAGIC in task 2 you produced the list of edges.
# MAGIC 
# MAGIC now you will need to transform those into two additional data structures:
# MAGIC * a set of graph nodes  
# MAGIC * a set of adjacency lists
# MAGIC  
# MAGIC  To understand their formats, consider this small example:
# MAGIC edges RDD:  `([('u1','u2','w1'),('u2','u3','w2'),('u2','u4','w3'),('u1','u5','w4')])`
# MAGIC 
# MAGIC * the nodes set contains the unique userIds that are found in the edges RDD:
# MAGIC   `['u3', 'u4', 'u2', 'u5', 'u1']`
# MAGIC * the adjacency lists are in the form of a python dictionary of dictionaries, as in this example:
# MAGIC ```
# MAGIC [('u3', {'u2': 'w2'}),
# MAGIC  ('u2', {'u3': 'w2', 'u1': 'w1', 'u4': 'w3'}),
# MAGIC  ('u4', {'u2': 'w3'}),
# MAGIC  ('u5', {'u1': 'w4'}),
# MAGIC  ('u1', {'u5': 'w4', 'u2': 'w1'})]
# MAGIC  ```
# MAGIC 
# MAGIC once you have produced these data structures, you add them to a `Graph` object. This Graph is then passed on to a library that computes Single-source-Shortest-Paths (SSSP):
# MAGIC `graph = Graph('user-user-network', allNodes.collect(), allAdjLists)`
# MAGIC 
# MAGIC ## available methods
# MAGIC The following library methods are provided to you to implement the Girwan-Newman algorithm, using this Graph as input:
# MAGIC 
# MAGIC ### SSSP
# MAGIC 
# MAGIC which is available in module
# MAGIC `from comscan.algorithms.shortest_path.single_source_shortest_paths_dijkstra`
# MAGIC 
# MAGIC and is defined as:
# MAGIC 
# MAGIC ```
# MAGIC def single_source_shortest_paths_dijkstra(graph, source, cutoff=None):
# MAGIC     """
# MAGIC     Provides Single Source Shortest Path implementation for a weighted graph
# MAGIC     :param graph: Weighted graph implemented with adjacency list.
# MAGIC     Edges in the graph are in the form of (u,v,w) where u,v are long and w is float
# MAGIC     :param source: source node
# MAGIC     :param cutoff: threshold value of largest possible distance in shortest paths.
# MAGIC     :return: all possible shortest paths from source to every other node in the graph
# MAGIC     """
# MAGIC ```
# MAGIC this will return the set of all shortest paths in the graph, starting from a **specific source node**, e.g.
# MAGIC 
# MAGIC `paths, distances = single_source_shortest_paths_dijkstra(graph, source=<source node>)`
# MAGIC 
# MAGIC ### computing the betweeness of each edge in the graph, given the shortest paths:
# MAGIC 
# MAGIC `def compute_edge_betweenness(shortest_paths_rdd: RDD) -> RDD:
# MAGIC     """Compute edge betweenness centrality of a graph in a distributed fashion given the shortest paths `
# MAGIC  
# MAGIC Using these methods and your knowledge of the GW algorithm, you should try and implement a Spark version of it where RDDs are parallelised where possible so that as much of the computation as possible occurs in parallel.
# MAGIC 
# MAGIC ### removing edges from the graph:
# MAGIC `graph.remove_edge(u,v)`  where `(u,v)` is an edge (works by side-effect. no need to reassign the graph variable)
# MAGIC 
# MAGIC ### calculating the number of connected components in the graph, once edges are removed (removing edges may produce multiple CC)
# MAGIC `ncc = number_connected_components(graph)`
# MAGIC 
# MAGIC ### other utility functions
# MAGIC `get_top_k_edges_with_highest_betweenness(sc, graph, k)` 
# MAGIC 
# MAGIC will give you the top-k edges with highest betweeness, as you will look for connected components (communities) by repeatedly removing high-betweeness edges from the original graph
# MAGIC 
# MAGIC `_remove_self_loops(graph)` updates `graph` by removing all loops from a node to itself
# MAGIC 
# MAGIC please note:
# MAGIC * you may ask workers to call the shortst-path method. If you do so, be careful as the method requires access to the entire graph. So you must **broadcast** the graph data structure to the workers prior to letting them work in it.
# MAGIC * the total betweenees of an edge can be computed *piecemeal* because it can be obtained by adding up partial betweeness values. 
# MAGIC 
# MAGIC You will produce a set of communities. Each of them is represented by a set of node IDs, so you end up with a list of lists:
# MAGIC 
# MAGIC ```[ [u,v,...], [u2, v2, ...] ]```
# MAGIC 
# MAGIC save the result to file as indicated below. don't forget to use your `<id>` to make your file name unique.

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
ALL_NODES_SMALL_TEXT = "/FileStore/tables/150454388/nodes-small2.txt"
ALL_ADJLIST_SMALL_TEXT =  "/FileStore/tables/150454388/adjlist-small2.txt"
SHORTEST_PATH_SMALL =   "/FileStore/tables/150454388/shortest-paths-small2.txt"

# OUT . the output you should produce containing all discovered communities (list of lists of graph nodes)
CC_SMALL = "/FileStore/tables/150454388/ConnectedComponents-small2.txt"

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

# MAGIC %md 
# MAGIC # your code here.
# MAGIC once you have created:
# MAGIC * nodes
# MAGIC * adjLists 
# MAGIC as explained abovew, you *may* save them here for checkpointing.
# MAGIC however please note that there is no overwrite mode for pickle files. please ask a demonstrator on how to delete files from Databricks' DBFS

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

k = 800

# COMMAND ----------

ncc = number_connected_components(graph)

while ncc == 1:
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