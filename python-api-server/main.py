import os
from awesome_graph.word_pre_process import WordPreProcess
from awesome_graph.report_process import ReportProcess
from awesome_graph import get_configure, get_logger
from awesome_graph.cluster_process import ClusterProcess
import json
import numpy as np

global_config = get_configure('AH_title_taiwanhistory_chatgpt')
logger = get_logger()
project_name = global_config['project_name']
logger.info('project_name:' + global_config['project_name'])


def create_project_folders(project_name, paths):
    for path in paths:
        full_path = path.format(project_name=project_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

            logger.info(f"Created folder: {full_path}")
        else:
            logger.info(f"Folder already exists: {full_path}")


paths = ["./datarun/{project_name}/organize_categories/", "./datarun/{project_name}/bert_paraphrases_mining", "./datarun/{project_name}/cluster", "./datarun/{project_name}/keyword_extraction", "./datarun/{project_name}/pre_process",
         "./datarun/{project_name}/report", "./datarun/{project_name}/graph_commons/"]

create_project_folders(project_name, paths)



# # # # Step 3 Preprocess paper
data_process = WordPreProcess(global_config)
data_process.run()

rp = ReportProcess(global_config)
rp.tending(f'trending_report_keyword.xlsx', f"datarun/{project_name}/pre_process/data_output.json")  # ###Tending report

from awesome_graph.build_map_process import BuildMapProcess
from py2neo import Graph, Node, Relationship, NodeMatcher
from awesome_graph import get_configure, get_logger
from awesome_graph.cluster_process import ClusterProcess
import numpy as np

print('upload node without cluster')
import json

logger = get_logger()
project_name = global_config["project_name"]
logger.info("project_name:" + global_config["project_name"])

dp = ClusterProcess(global_config)
num_clusters = dp.config['num_clusters']
repeats = dp.config['repeats']
vocab, corpus = dp.drop_low_frequency_words(3)
vocab_set = list(set(vocab))
bm = BuildMapProcess(global_config)
dp = ClusterProcess(global_config)
cooccur_matrix = dp.process_cooccur_matrix(vocab_set, corpus)

clusters = [10000 for i in range(len(vocab_set))]
cluster_assignments = bm.create_cluster_assignments(clusters, vocab_set)
graph = Graph(global_config["build_map"]["uri"], name=global_config["build_map"]["database_name"], auth=(global_config["build_map"]["user"], global_config["build_map"]["password"]), )

# open /home/arden/Documents/gdrive/awesome_graph_2/datarun/AH_title_taiwanhistory_chatgpt/pre_process/data_output.json
with open(f"./datarun/{project_name}/pre_process/data_output.json", "r") as file:
    data = json.load(file)

vocab_year_assignments = bm.group_keywords_by_year(data["papers"], vocab_set)

filted_cooccur = dp.filter_cooccur(cooccur_matrix, vocab_set)

bm.bulk_build_node(cluster_assignments, graph, corpus, vocab_year_assignments, global_config["build_map"]["datarun_id"], vocab_set)
bm.bulk_build_relationship(filted_cooccur, graph, global_config["build_map"]["datarun_id"])

logger.info("Done")

#
#
# def save_cluster_to_file(cluster_list, filename="cluster.json"):
#     try:
#         with open(filename, "w") as file:
#             json.dump(cluster_list, file, indent=4, ensure_ascii=False)
#         logger.info("Cluster data saved to" + filename)
#     except Exception as e:
#         logger.info("Error:", str(e))
#
#
# def read_cluster_from_file(filename="cluster.json"):
#     try:
#         with open(filename, "r") as file:
#             cluster_list = json.load(file)
#         return cluster_list
#     except Exception as e:
#         logger.info("Error:", str(e))
#         return []
#
#
# from awesome_graph import get_configure, get_logger
#
# logger = get_logger()
# project_name = global_config['project_name']
# logger.info('project_name:' + global_config['project_name'])
#
# ## Step 5 ClusterProcess
# dp = ClusterProcess(global_config)
# num_clusters = dp.config['num_clusters']
# repeats = dp.config['repeats']
#
# vocab, corpus = dp.drop_low_frequency_words(global_config['cluster']["minimal_word_frequency"])
# vocab_set = list(set(vocab))
# cooccur_matrix = dp.process_cooccur_matrix(vocab_set, corpus)
# norm_matrix = dp.process_norm_matrix(cooccur_matrix, vocab_set)
#
# # read similarity_matrix from numpy array
# logger.info(f"Load ./datarun/{project_name}/cluster/similarity_matrix.npy")
# # similarity_matrix = np.load(f"./datarun/{project_name}/cluster/similarity_matrix.npy")
# similarity_matrix = dp.process_similarity_matrix(norm_matrix, vocab_set)
#
# # save similarity_matrix is a numpy array
# np.save(f"./datarun/{project_name}/cluster/similarity_matrix.npy", similarity_matrix)
#
# ## Step 5 ClusterProcess
# dp = ClusterProcess(global_config)
#
# clusters = dp.cluster_data(similarity_matrix, num_clusters=num_clusters, repeats=repeats)
#
# # %%
# filted_cooccur = dp.filter_cooccur(cooccur_matrix, vocab_set)
# # Save the cluster data to a JSON file
# save_cluster_to_file(clusters, filename=global_config['cluster']['cluster_path'].format(project_name=project_name))
# save_cluster_to_file(filted_cooccur, filename=global_config['cluster']['filted_cooccur_path'].format(project_name=project_name))
#
# from awesome_graph import get_configure, get_logger
#
# logger = get_logger()
# project_name = global_config['project_name']
# logger.info('project_name:' + global_config['project_name'])
#
# filted_cooccur = dp.filter_cooccur(cooccur_matrix, vocab_set)
# save_cluster_to_file(clusters, filename=global_config['cluster']['cluster_path'].format(project_name=project_name))
# save_cluster_to_file(filted_cooccur, filename=global_config['cluster']['filted_cooccur_path'].format(project_name=project_name))
#
# filted_cooccur = dp.filter_cooccur(cooccur_matrix, vocab_set)
# save_cluster_to_file(clusters, filename=global_config['cluster']['cluster_path'].format(project_name=project_name))
# save_cluster_to_file(filted_cooccur, filename=global_config['cluster']['filted_cooccur_path'].format(project_name=project_name))
#
# from awesome_graph import get_configure, get_logger
#
# from awesome_graph.cluster_process import ClusterProcess
#
# dp = ClusterProcess(global_config)
# vocab, corpus = dp.drop_low_frequency_words(global_config['cluster']["minimal_word_frequency"])
# vocab_set = list(set(vocab))
#
# from awesome_graph.build_map_process import BuildMapProcess
# from py2neo import Graph, Node, Relationship, NodeMatcher
# from awesome_graph import get_configure, get_logger
# import json
#
# logger = get_logger()
# project_name = global_config["project_name"]
# logger.info("project_name:" + global_config["project_name"])
#
#
# def save_cluster_to_file(cluster_list, filename="cluster.json"):
#     with open(filename, "w") as file:
#         json.dump(cluster_list, file, indent=4, ensure_ascii=False)
#     logger.info("Cluster data saved to" + filename)
#
#
# def read_cluster_from_file(filename="cluster.json"):
#     with open(filename, "r") as file:
#         cluster_list = json.load(file)
#     return cluster_list
#
#
# def load_json_file(file_path):
#     with open(file_path, "r") as file:
#         data = json.load(file)
#     return data


# # Step 6 BuildMapProcess
# bm = BuildMapProcess(global_config)
# filted_cooccur = read_cluster_from_file(filename=global_config["cluster"]["filted_cooccur_path"].format(project_name=project_name))
# clusters = read_cluster_from_file(filename=global_config["cluster"]["cluster_path"].format(project_name=project_name))

# vocab_set = list(set(vocab))

# cluster_assignments = bm.create_cluster_assignments(clusters, vocab_set)
# graph = Graph(global_config["build_map"]["uri"], name=global_config["build_map"]["database_name"], auth=(global_config["build_map"]["user"], global_config["build_map"]["password"]), )
#
# # open /home/arden/Documents/gdrive/awesome_graph_2/datarun/AH_title_taiwanhistory_chatgpt/pre_process/data_output.json
# with open(f"./datarun/{project_name}/pre_process/data_output.json", "r") as file:
#     data = json.load(file)
#
# papers = data["papers"]
# vocab_year_assignments = bm.group_keywords_by_year(papers, vocab_set)
#
# with open(global_config['cluster']["source_corpus"].format(project_name=project_name), "r") as f:
#     corpus = json.load(f)
#
# bm.bulk_build_node(cluster_assignments, graph, corpus, vocab_year_assignments, global_config["build_map"]["datarun_id"], vocab_set)
# bm.bulk_build_relationship(filted_cooccur, graph, global_config["build_map"]["datarun_id"])
# # bm.bulk_build_title_node(graph,global_config["build_map"]["datarun_id"])
# # bm.bulk_build_title_relationship(graph,global_config["build_map"]["datarun_id"])
#
# logger.info("Done")

# query_file_path = f"{bm.config['sql_query_path']}/{bm.config['database_name']}.txt"

