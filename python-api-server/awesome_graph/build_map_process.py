
from py2neo import Graph, Node, Relationship, NodeMatcher
from py2neo.bulk import create_relationships, merge_relationships
from py2neo.bulk import merge_nodes
import numpy as np
from collections import Counter
from tqdm.notebook import tqdm
import json
import os
import re
import ast
import pandas as pd
from termcolor import colored
import datetime
import time
import nltk
from nltk.cluster import KMeansClusterer
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML

from sklearn.preprocessing import MinMaxScaler
from awesome_graph import get_configure ,get_logger

# Here is a function to do the rescaling
def rescale_number(x, x_min, x_max, y_min, y_max):
    if x < x_min or x > x_max:
        print(f"rescale_number: x is out of range, x: {x}, x_min: {x_min}, x_max: {x_max}")
        if x < x_min:
            x = x_min
        elif x > x_max:
            x = x_max

    # Scale x from its original range to [0, 1]
    x_scaled = (x - x_min) / (x_max - x_min)

    # Rescale x_scaled to y range
    y = y_min + (x_scaled * (y_max - y_min))

    return int(y)



class BuildMapProcess:
    def __init__(self, global_config):
        self.config = global_config["build_map"]
        self.project_name = global_config["project_name"]

        for k, v in self.config.items():
            if isinstance(v, str):
                self.config[k] = v.replace("{project_name}", self.project_name)

    def connect_db(self, uri, user, password):
        # Create Neo4j graph
        self.graph = Graph(uri, auth=(user, password))  # self.node_matcher = NodeMatcher(self.graph)

    def create_cluster_assignments(self, clusters, vocab):
        cluster_assignments = {}
        # print(clusters)
        for vocab_index, i in enumerate(clusters):
            cluster_assignments[vocab[vocab_index]] = i

        return cluster_assignments

    def upsert_node(self, name, cluster, graph, new_properties=None):
        # Define the node label and properties
        node_label = cluster
        node_properties = {"name": name}

        # Create a NodeMatcher object
        matcher = NodeMatcher(graph)

        # Find the node with the given properties
        node = matcher.match(node_label, **node_properties).first()
        print(node)
        print(type(node))
        # If the node exists, update it; otherwise, create a new node

        if node is None:
            # Create a new node with the given properties and the new properties
            node_properties.update(new_properties)
            node = Node(node_label, **node_properties)
            graph.create(node)
        else:
            # Update the node with new properties
            for key, value in new_properties.items():
                node[key] = value
            graph.push(node)
        return node

    def bulk_build_relationship(self, filted_cooccur, graph, datarun_id):
        relationship_count = {"very low": 0, "low": 0, "medium": 0, "high": 0, "very high": 0, }
        print(f"bulk_build_relationship: {len(filted_cooccur)}")
        def partition_list(input_list, partition_size=1000):
            for i in range(0, len(input_list), partition_size):
                yield input_list[i:i + partition_size]

        def get_nodes_as_dict(graph):
            query = "MATCH (n) WHERE NOT n:title RETURN n.name AS name, id(n) AS id"
            result = graph.run(query)
            node_dict = {record["name"]: int(record["id"]) for record in result}
            return node_dict

        node_dict = get_nodes_as_dict(graph)

        bulk_data = {}
        for source_node, target_nodes_info in filted_cooccur.items():
            for target_node, concurrency_weight in target_nodes_info.items():
                concurrency_weight_config = self.config["concurrency_weight"]
                if concurrency_weight in concurrency_weight_config["very_low"]:
                    relationship = "very low"
                    relationship_count["very low"] += 1
                elif concurrency_weight in concurrency_weight_config["low"]:
                    relationship = "low"
                    relationship_count["low"] += 1
                elif concurrency_weight in concurrency_weight_config["medium"]:
                    relationship = "medium"
                    relationship_count["medium"] += 1
                elif concurrency_weight in concurrency_weight_config["high"]:
                    relationship = "high"
                    relationship_count["high"] += 1
                elif concurrency_weight >= concurrency_weight_config["very_high"]:
                    relationship = "very high"
                    relationship_count["very high"] += 1
                else:
                    relationship = "other"

                # relationship = relationship.replace(" ", "_")
                # relationship = "cooccur_" + relationship
                relationship = relationship

                if relationship not in bulk_data:
                    bulk_data[relationship] = []
                try:
                    if concurrency_weight > 100:
                        concurrency_weight = 100
                        print(f"concurrency_weight > 100: {source_node} {target_node}, update to 100")
                    if source_node == target_node:
                        pass
                        # print(f"source_node == target_node: {source_node} {target_node}, skip")
                    elif relationship == "other":
                        pass
                    else:
                        bulk_data[relationship].append((node_dict[source_node], {"concurrency_weight": int(concurrency_weight), 'datarun_id': datarun_id}, node_dict[target_node],))
                except KeyError:
                    print(f"Error: {source_node} or {target_node} not in node_dict")



        for relationship, data in bulk_data.items():
            for chunk in partition_list(data):
                error_count = 0
                while 1:
                    try:
                        merge_relationships(graph.auto(), chunk, relationship)
                        break
                    except Exception as e:
                        from awesome_graph import get_configure
                        global_config = get_configure()
                        graph = Graph(global_config["build_map"]["uri"], name=global_config["build_map"]["database_name"], auth=(global_config["build_map"]["user"], global_config["build_map"]["password"]), )

                        error_count += 1
                        time.sleep(5)
                        print('merge_relationships error_count', error_count)

                        if error_count > 10:
                            print("error_count > 10, break")
                            print(e)

                            break


        # graph.commit()
        print(f"relationship_count: {relationship_count}")

    def build_relationship(self, filted_cooccur, f):
        relationship_count = {"very low": 0, "low": 0, "medium": 0, "high": 0, "very high": 0, }
        for source_node, target_nodes_info in filted_cooccur.items():
            for target_node, concurrency_weight in target_nodes_info.items():
                concurrency_weight_config = self.config["concurrency_weight"]
                if concurrency_weight in concurrency_weight_config["very_low"]:
                    relationship = "very low"
                    relationship_count["very low"] += 1
                elif concurrency_weight in concurrency_weight_config["low"]:
                    relationship = "low"
                    relationship_count["low"] += 1
                elif concurrency_weight in concurrency_weight_config["medium"]:
                    relationship = "medium"
                    relationship_count["medium"] += 1
                elif concurrency_weight in concurrency_weight_config["high"]:
                    relationship = "high"
                    relationship_count["high"] += 1
                elif concurrency_weight >= concurrency_weight_config["very_high"]:
                    relationship = "very high"
                    relationship_count["very high"] += 1

                relationship = relationship.replace(" ", "_")
                relationship = "cooccur_" + relationship

                query = f"""MATCH (source), (target)
                WHERE source.name = "{source_node}" AND target.name = "{target_node}"
                MERGE (source)-[r:{relationship}]-(target)
                RETURN type(r);\n""".replace("                ", "")
                f.write(query)

                # node_rel = Relationship(source_node, relationship, target_node, weight=str(concurrency_weight))  # self.graph.create(node_rel)
        print(f"relationship_count: {relationship_count}")

    def group_keywords_by_year(self, publications, vocab_set):
        def convert_to_ad(year):
            try:
                year = int(year)
            except:
                print(f"Invalid publication date: {year}")
                return "0000"

            if not isinstance(year, int) or year <= 0:
                return "0000"
            else:
                return str(year + 1911)

        # Create a dictionary to store keywords grouped by publication year
        keyword_years = {}

        print("group_keywords_by_year")
        # Iterate through the publications and extract keywords
        for publication in publications:
            keywords = publication["keywords"]
            if publication["publication_date"]:
                publication_date = publication["publication_date"].strip()[:4]  # Extract the year
            else:
                publication_date = "0000"

            if '-' in publication_date:
                publication_date = publication_date.split('-')[0]
                publication_date = convert_to_ad(publication_date)  # print(f'convert to ad citation year: {publication_date}')

            if publication_date.isdigit():
                for keyword in keywords:
                    if keyword not in keyword_years:
                        keyword_years[keyword] = [int(publication_date)]

                    else:
                        keyword_years[keyword].append(int(publication_date))


            else:
                print(f"Invalid publication date: {publication_date}")


        valid_keyword = set(vocab_set).intersection(set(keyword_years.keys()))

        result = {}
        for keyword, years in keyword_years.items():
            if keyword in valid_keyword:
                years = list(set(years))
                years.sort()
                result[keyword] = years
        return result

    def bulk_build_node(self, cluster_assignments, g, corpus, vocab_year_assignments, datarun_id,vocab_set):
        print("bulk_build_node")
        set_vocab_set = set(vocab_set)
        vocab = []
        for words in tqdm(corpus):
            for word in words:
                if word in set_vocab_set:
                    vocab.append(word)

        vocab_counter = Counter(vocab)

        def swap_key_value(dictionary):
            swapped_dict = {}
            for key, value in dictionary.items():
                if value not in swapped_dict:
                    swapped_dict[value] = [key]
                else:
                    swapped_dict[value].append(key)
            return swapped_dict

        def get_min_max_freq(swapped_dict):
            print("get_min_max_freq")
            min_freq = 100000
            max_freq = 0
            for cluster_number, data in tqdm(swapped_dict.items()):
                for d in data:
                    frequency = vocab_counter[d]
                    # update min and max frequencies
                    if frequency < min_freq:
                        min_freq = frequency
                    if frequency > max_freq:
                        max_freq = frequency

                    if min_freq == max_freq:
                        min_freq = max_freq - 1
            return min_freq, max_freq

        def partition_list(input_list, partition_size=1000):
            for i in range(0, len(input_list), partition_size):
                yield input_list[i:i + partition_size]

        print(f'Create a new node with a "name" property and a "cluster" property')
        swapped_dict = swap_key_value(cluster_assignments)



        min_freq, max_freq = get_min_max_freq(swapped_dict)
        for cluster_number, input_list in tqdm(swapped_dict.items()):
            node_data = []

            for data in partition_list(input_list):
                for d in data:
                    if d in vocab_year_assignments:
                        years = vocab_year_assignments[d]
                    else:
                        years = []
                        print(f"{d} not in vocab_year_assignments")

                    log_frequency = np.log(vocab_counter[d])
                    log_min_freq = np.log(min_freq)
                    log_max_freq = np.log(max_freq)
                    frequency = vocab_counter[d]

                    d3_size = {'d3_r': rescale_number(log_frequency, log_min_freq, log_max_freq,1, 42),
                               'd3_text_y': rescale_number(log_frequency,log_min_freq, log_max_freq, 1, 42),
                               'd3_collide': rescale_number(log_frequency,log_min_freq, log_max_freq, 1, 8, ),  # calcuate the d3 properties based on frequency
                               'd3_node_font_size': rescale_number(log_frequency,log_min_freq, log_max_freq, 1, 4, )}  # calcuate the d3 properties based on frequency

                    _node_data = {"name": d, "cluster_number": cluster_number, "frequency": int(frequency), 'years': years, 'datarun_id': datarun_id}
                    _node_data.update(d3_size)

                    node_data.append(_node_data)

                # data = [{"name": d, "cluster_number": cluster_number, "concurrency": int(vocab_counter[d]),'years':vocab_year_assignments[d] } for d in data]
                """
                :var data: list of dict
                key: cluster_number
                value: {"name":d,"cluster_number":cluster_number}
                """
            errror_count = 0
            while 1:
                try:
                    merge_nodes(g.auto(), node_data, (f"cluster_{cluster_number}", "name"))  # upsert data where name = name  and  node type : cluster_{cluster_number},
                    break
                except Exception as e:
                    print(e)
                    errror_count += 1
                    print(f'DB connection error, retry {errror_count}')
                    merge_nodes(g.auto(), node_data, (f"cluster_{cluster_number}", "name"))  # upsert data where name = name  and  node type : cluster_{cluster_number},
                    if errror_count > 10:
                        print("error_count > 10, break")
                        break

    def build_node(self, cluster_assignments, f):
        print('Create a new node with a "name" property and a "cluster" property')
        print(f"Node count: {len(cluster_assignments)}")
        for node_name, cluster_number in cluster_assignments.items():
            query = f"""CREATE (:cluster_{cluster_number} {{name: '{node_name}',cluster_number:{cluster_number} }});"""
            f.write(query)
            f.write("\n")

    def execute_query_file(self, query_file_path, database_name="neo4j", is_print=False, offset=0):
        print(f"execute_query_file: {query_file_path}")
        graph = Graph(self.config["uri"], auth=(self.config["user"], self.config["password"],))

        with open(query_file_path, "r") as f:
            query_list = f.read().split(";")
            for q in tqdm(query_list[offset:], desc="execute_query_file"):
                if len(q) >= 2:
                    try:
                        q = f"USE {database_name} " + q
                        result = graph.run(q)
                        if is_print:
                            print(result)
                    except:
                        print(f"Error: {q}")

    def bulk_build_title_node(self, g, datarun_id):
        data = json.load(open(self.config["source_data"], "r"))
        corpus = json.load(open(self.config["source_corpus"], "r"))

        bulk_data = []
        for i in range(len(data["papers"])):
            if data["papers"][i]["title"]:
                title = data["papers"][i]["title"].strip()
            else:
                title = "None"
            keywords = corpus[i]

            bulk_data.append({"name": title, "keywords": keywords, 'datarun_id': datarun_id})

        merge_nodes(g.auto(), bulk_data, (f"title", "name"))  # upsert data where name = name  and  node type : title},

    def build_title_node(self, f):
        data = json.load(open(self.config["source_data"], "r"))
        corpus = json.load(open(self.config["source_corpus"], "r"))

        for i in tqdm(range(len(data["papers"]))):
            if data["papers"][i]["title"]:
                title = data["papers"][i]["title"].strip()
            else:
                title = "None"
            keywords = corpus[i]

            query = f"""CREATE (:title {{name: '{title}',keywords:{keywords} }});"""
            f.write(query)
            f.write("\n")

    def bulk_build_title_relationship(self, graph, datarun_id):
        data = json.load(open(self.config["source_data"], "r"))
        corpus = json.load(open(self.config["source_corpus"], "r"))

        def get_title_nodes_as_dict(graph):
            # 查詢語句，選擇所有類型為 "title" 的節點，並返回節點 ID 和名稱
            query = "MATCH (n) RETURN n.name AS name, id(n) AS id"
            result = graph.run(query)
            node_dict = {record["name"]: record["id"] for record in result}

            return node_dict

        node_dict = get_title_nodes_as_dict(graph)

        bulk_data = []

        for i in range(len(data["papers"])):
            if data["papers"][i]["title"]:
                title = data["papers"][i]["title"].strip()
            else:
                title = "None"
            keywords = corpus[i]
            for keyword in keywords:
                if keyword in node_dict and title in node_dict:
                    bulk_data.append((node_dict[title], {'datarun_id': datarun_id}, node_dict[keyword]))  # else:  #     print(f'Error: {title} or {keyword} not in node_dict')

        merge_relationships(graph.auto(), bulk_data, "have_title")  # graph.commit()

    def build_title_relationship(self, f):
        data = json.load(open(self.config["source_data"], "r"))
        corpus = json.load(open(self.config["source_corpus"], "r"))

        for i in tqdm(range(len(data["papers"]))):
            if data["papers"][i]["title"]:
                title = data["papers"][i]["title"].strip()
            else:
                title = "None"

            keywords = corpus[i]
            for keyword in keywords:
                relationship = "have_title"

                query = f"""MATCH (source), (target)
                WHERE source.name = "{keyword}" AND target.name = "{title}"
                MERGE (source)-[r:{relationship}]-(target)
                RETURN type(r); \n""".replace("                ", "")
                f.write(query)

    def build_title_relationship(self, f):
        data = json.load(open(self.config["source_data"], "r"))
        corpus = json.load(open(self.config["source_corpus"], "r"))

        for i in tqdm(range(len(data["papers"]))):
            if data["papers"][i]["title"]:
                title = data["papers"][i]["title"].strip()
            else:
                title = "None"

            keywords = corpus[i]
            for keyword in keywords:
                relationship = "have_title"

                query = f"""MATCH (source), (target)
                WHERE source.name = "{keyword}" AND target.name = "{title}"
                MERGE (source)-[r:{relationship}]-(target)
                RETURN type(r); \n""".replace("                ", "")
                f.write(query)
