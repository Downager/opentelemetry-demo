import findpapers
from pprint import pprint
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from py2neo import Graph, Node, Relationship, NodeMatcher
from py2neo.bulk import create_relationships, merge_relationships
from py2neo.bulk import merge_nodes
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import Counter
# from tqdm.notebook import tqdm
from tqdm import tqdm
import json
import os
import openai
import re
import ast
import pandas as pd
from termcolor import colored
import time
import nltk
from nltk.cluster import KMeansClusterer
from collections import Counter


from awesome_graph import logger
class ClusterProcess:
    def __init__(self, global_config):
        self.config = global_config["cluster"]
        self.project_name = global_config["project_name"]
        if not os.path.exists(f"./datarun/{self.project_name}/cluster"):
            os.mkdir(f"./datarun/{self.project_name}/cluster")

        for k, v in self.config.items():
            if isinstance(v, str):
                self.config[k] = v.replace("{project_name}", self.project_name)


    def drop_low_frequency_words_in_corpus(self, corpus, vocab,frequency_words):
        def group_elements_by_count(counter):
            grouped_by_count = {}
            for element, count in counter.items():
                if count not in grouped_by_count:
                    grouped_by_count[count] = [element]
                else:
                    grouped_by_count[count].append(element)
            return grouped_by_count

        def print_grouped_elements_sorted(grouped_elements):
            print("Elements grouped by count:")
            for count in sorted(grouped_elements.keys()):
                elements = grouped_elements[count]
                co_occurrence = f"Co-occurrence:{count} : 關鍵字數:{len(elements)}"
                print(co_occurrence)

        print("start to drop low frequency words in corpus")
        word_counter = Counter(vocab)

        # Grouping the elements by their count number using the function
        grouped_elements = group_elements_by_count(word_counter)
        # Using the function to print the grouped elements
        print_grouped_elements_sorted(grouped_elements)

        for i in tqdm(range(len(corpus))):
            new_word_list = []
            for word in corpus[i]:
                if word_counter[word] >= frequency_words:
                    new_word_list.append(word)

            corpus[i] = new_word_list

        return corpus

    def process_cooccur_matrix(self, vocab_set, corpus):
        from scipy import sparse

        print("start to process cooccur matrix")
        # cooccur_matrix = np.zeros((len(vocab_set), len(vocab_set)))
        cooccur_matrix = np.zeros((len(vocab_set), len(vocab_set)), dtype='f')
        # cooccur_matrix = sparse.lil_matrix((len(vocab_set), len(vocab_set)))
        # cooccur_matrix = cp.zeros((len(vocab), len(vocab)),dtype=np.int8)
        for doc in tqdm(corpus):
            for i, word1 in enumerate(doc):
                for j, word2 in enumerate(doc):
                    if i != j:
                        cooccur_matrix[vocab_set.index(word1), vocab_set.index(word2)] += 1

        # save cooccur_matrix
        if self.config["save_cooccur_matrix"] is True:
            if not os.path.exists(os.path.dirname(self.config["cooccur_matrix_path"])):
                os.makedirs(os.path.dirname(self.config["cooccur_matrix_path"]))
            pd.DataFrame(cooccur_matrix, columns=vocab_set, index=vocab_set).to_excel(
                self.config["cooccur_matrix_path"], header=True, index=True
            )

        return cooccur_matrix

    def process_norm_matrix(self, cooccur_matrix, vocab_set):
        print("start to process norm matrix")

        # 計算行總和和列總和
        row_sums = np.sum(cooccur_matrix, axis=1)
        col_sums = np.sum(cooccur_matrix, axis=0)

        # 處理分母為零的情況
        row_sums[row_sums == 0] = np.finfo(float).eps  # 使用最小的浮点数
        col_sums[col_sums == 0] = np.finfo(float).eps

        # 計算norm matrix
        norm_matrix = cooccur_matrix / np.sqrt(np.outer(row_sums, col_sums))

        if self.config["save_norm_matrix"] is True:
            if not os.path.exists(os.path.dirname(self.config["norm_matrix_path"])):
                os.makedirs(os.path.dirname(self.config["norm_matrix_path"]))
            norm_matrix = pd.DataFrame(norm_matrix, columns=vocab_set, index=vocab_set)
            norm_matrix.to_excel(
                self.config["norm_matrix_path"], header=True, index=True
            )

        self.cooccur_matrix = cooccur_matrix
        return norm_matrix

    # Apply K-means clustering to similarity matrix
    def cluster_data(self, similarity_matrix, num_clusters, repeats=25):
        start_time = time.time()
        print(
            f"start to cluster data with {num_clusters} clusters and {repeats} repeats"
        )
        clusterer = KMeansClusterer(
            num_clusters,
            distance=nltk.cluster.util.cosine_distance,
            repeats=repeats,
            avoid_empty_clusters=True,
        )
        clusters = clusterer.cluster(similarity_matrix, assign_clusters=True)

        # End the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"Time taken to cluster data: {elapsed_time} seconds.")

        return clusters

    # Calculate similarity matrix
    def process_similarity_matrix(self, norm_matrix, vocab_set):
        if np.isnan(norm_matrix).any():
            print(colored("norm_matrix has nan value", "red"))
            raise ValueError("norm_matrix has nan value")
            # print('start to replace nan value with 0')
            # norm_matrix = np.nan_to_num(norm_matrix, nan=0.0)
            # print('finish to replace nan value with 0')

        print("start to process similarity matrix")

        similarity_matrix = np.dot(norm_matrix, norm_matrix.T)
        # similarity_matrix = cp.dot(norm_matrix, norm_matrix.T)

        if self.config["save_similarity_matrix"] is True:
            print("start to save similarity matrix as excel")
            if not os.path.exists(self.config["similarity_matrix_path"]):
                os.makedirs(os.path.dirname(self.config["similarity_matrix_path"]))
            np.save(
                self.config["similarity_matrix_path"].replace(".xlsx", ".npy"),
                similarity_matrix,
            )
            pd.DataFrame(similarity_matrix, columns=vocab_set, index=vocab_set).to_excel(
                self.config["similarity_matrix_path"], header=True, index=True
            )

        print("finish to process similarity matrix")
        return similarity_matrix

    def get_vocab(self, corpus):
        vocab = []
        for words in corpus:
            for word in words:
                vocab.append(word)
        return vocab

    def post_process(self, cluster, filted_cooccur_matrix, is_print=True):
        print("start to save cluster result as excel")

        df_cluster = pd.DataFrame({"vocab": vocab, "cluster": cluster})
        df_cluster["cooccur_weight"] = df_cluster["vocab"].map(filted_cooccur_matrix)
        df_cluster.sort_values(by=["cluster"], inplace=True)

        if self.config["save_cluster_result"] is True:
            if not os.path.exists(os.path.dirname(self.config["cluster_result_path"])):
                os.makedirs(os.path.dirname(self.config["cluster_result_path"]))
            df_cluster.to_csv(
                self.config["cluster_result_path"], index=True, header=True
            )

            df_cluster_group = df_cluster.groupby("cluster").count()

            if self.config["cluster_result_text_path"]:
                if not os.path.exists(
                    os.path.dirname(self.config["cluster_result_text_path"])
                ):
                    os.makedirs(
                        os.path.dirname(self.config["cluster_result_text_path"])
                    )
            with open(self.config["cluster_result_text_path"], "w") as f:
                for i in range(0, len(df_cluster_group)):
                    f.write("Cluster {}:".format(i))
                    f.write("\n")
                    f.write(
                        str(df_cluster[df_cluster["cluster"] == i]["vocab"].tolist())
                    )
                    f.write("\n")
                    f.write("\n")
                    if is_print:
                        print(colored("Cluster {}:".format(i), "red"))
                        print(df_cluster[df_cluster["cluster"] == i]["vocab"].tolist())

        return df_cluster

    def drop_low_frequency_words(self,minimal_word_frequency):
        print("start to preprocess corpus")
        corpus_path = self.config["source_corpus"]
        print("start to load corpus from {}".format(corpus_path))

        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        for i, doc in enumerate(corpus):
            corpus[i] = [x for x in doc if isinstance(x, str) and not x.isdigit() and len(x) > 1]


        vocab  =  [word for sublist in corpus for word in sublist]
        print('unique vocab before : ',len(set(vocab)))
        corpus = self.drop_low_frequency_words_in_corpus(corpus, vocab,minimal_word_frequency)



        vocab  =  [word for sublist in corpus for word in sublist]
        print('unique vocab after : ',len(set(vocab)))

        data = json.load(open(self.config["source_data"], "r"))
        papers = data["papers"]
        for i in range(len(papers)):
            papers[i]["keywords"] = corpus[i]
        with open(f"./datarun/{self.project_name}/cluster/data_output.json", "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        self.write_corpus(
            corpus, f"./datarun/{self.project_name}/cluster/corpus_output.json"
        )

        # dump vocab
        with open(f"./datarun/{self.project_name}/cluster/vocab_output.json", "w") as f:
            json.dump(vocab, f, indent=4, ensure_ascii=False)

        print("finish to preprocess corpus")
        return vocab, corpus

    def write_corpus(self, corpus, file_name):
        print("write corpus to ", file_name)
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_name, "w") as f:
            json.dump(corpus, f, indent=4, ensure_ascii=False)

    def filter_cooccur(self, cooccur_matrix, vocab_set) -> dict:
        df = pd.DataFrame(cooccur_matrix, columns=vocab_set, index=vocab_set)
        result = {}
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            new_dict = {
                k: v for k, v in row_dict.items() if v
            }  # remove the row which all value is 0
            result[str(index)] = new_dict
        return result
