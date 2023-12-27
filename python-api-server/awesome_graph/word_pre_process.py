
from sentence_transformers import SentenceTransformer, util

from tqdm.notebook import tqdm
import json
import os
import re

from termcolor import colored

import nltk
from collections import Counter








class WordPreProcess:
    def __init__(self, global_config):
        self.config = global_config["word_preprocess"]
        self.project_name = global_config["project_name"]
        self.corpus_path = (
            f"./datarun/{self.project_name}/pre_process/corpus_output.json"
        )

        for k, v in self.config.items():
            if isinstance(v, str):
                self.config[k] = v.replace("{project_name}", self.project_name)



    def write_corpus(self, corpus, file_name):
        print("write corpus to ", file_name)
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_name, "w") as f:
            json.dump(corpus, f, indent=4, ensure_ascii=False)

    def process_augmentation(self, corpus):
        print(colored("start augmentation process", "green"))
        data = self.read_data(self.config["source_data"])
        papers = data["papers"]

        vocab = self.get_vocab(corpus)
        vocab = [v for v in vocab if " " in v or "-" in v]

        vocab_before_count = len(vocab)
        vocab = list(set(vocab))
        print("unique vocab: ", len(vocab))

        for i in range(len(papers)):
            result = self.search_words(papers[i]["abstract"], vocab)
            if len(result) > 0:
                # print(colored(result,'blue'))
                # break
                corpus[i].extend(result)
                corpus[i] = list(set(corpus[i]))

        vocab_after_count = len(self.get_vocab(corpus))
        print("before augmentation: ", vocab_before_count)
        print("after augmentation", vocab_after_count)

        # display(corpus)
        self.write_corpus(
            corpus, f"./datarun/{self.project_name}/pre_process/corpus_output.json"
        )
        return corpus

    def read_data(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        return data

    def get_gpt_unique_corpus(self, papers):
        corpus = []
        for index, paper in enumerate(papers):
            corpus.append([])
            # if 'gpt_response' not in paper:
            #     print('index: {} has no gpt_response'.format(index))
            #     continue
            # gpt_response = paper['gpt_response']
            # for response in gpt_response:
            #     breakpoint()
            corpus[index].extend(paper["keywords"])
            corpus[index] = list(set(corpus[index]))
        return corpus

    def get_bert_unique_corpus(self, papers):
        corpus = []
        for paper in papers:
            corpus.append(paper["bert_extract_keywords"])
        return corpus

    def get_papers_invalid_index(self, papers):
        invalid_index = [i for i in range(len(papers))]
        for index, paper in enumerate(papers):
            if "gpt_response" not in paper:
                # print(paper)
                print("index: {} has no gpt_response".format(index))
                continue
            gpt_response = paper["gpt_response"]
            for response in gpt_response:
                if len(response["keywords"]) > 1 and index in invalid_index:
                    invalid_index.remove(index)
        return invalid_index

    def run(self):
        def remove_keywords(corpus, remove_list):
            for words in corpus:
                for i in range(len(words)):
                    for remove_this_str in remove_list:
                        words[i] = words[i].replace(remove_this_str, "")

            return corpus

        def lower_nested_list(nested_list):
            return [[item.lower() for item in sublist] for sublist in nested_list]

        def remove_corpus_parentheses(corpus):
            def remove_parentheses(input_list):
                updated_list = []
                for string in input_list:
                    updated_string = ""
                    within_parentheses = False
                    for char in string:
                        if char == "(":
                            within_parentheses = True
                        elif char == ")":
                            within_parentheses = False
                        elif not within_parentheses:
                            updated_string += char
                    updated_list.append(updated_string)
                return updated_list

            for index, words in enumerate(corpus):
                corpus[index] = remove_parentheses(words)
            return corpus

        def filter_sublists_max_len(superset_list, max_len):
            # iterate over all sublists
            for sublist in superset_list:
                # iterate over all strings in the sublist in reverse order
                for i in range(len(sublist) - 1, -1, -1):
                    # if the string length is greater than the 'max_len'
                    if len(sublist[i]) > max_len:
                        # remove the string
                        sublist.pop(i)

            return superset_list


        def remove_numbers_from_sublists(sublists):
            # Define a function to check if an element is a number
            def is_number(element):
                try:
                    float(element)
                    return True
                except ValueError:
                    return False

            # Iterate through each sub-list and filter out the number elements
            filtered_sublists = []
            for sublist in sublists:
                filtered_sublist = [
                    element for element in sublist if not is_number(element)
                ]
                filtered_sublists.append(filtered_sublist)

            return filtered_sublists

        def remove_hyphen(corpus):
            for index, words in enumerate(corpus):
                corpus[index] = [i.replace("-", " ").replace("_", " ") for i in words]
            return corpus

        def remove_short_words(sublists, min_len=1):
            # Define a function to check if an element is a number

            # Iterate through each sub-list and filter out the number elements
            filtered_sublists = []
            for sublist in sublists:
                filtered_sublist = [
                    element for element in sublist if len(element) > min_len
                ]
                filtered_sublists.append(filtered_sublist)

            return filtered_sublists

        def remove_punctuation_except_spaces(corpus):
            def preprocess_string(input_string):
                # Task 1: Remove punctuation except space
                cleaned_string = re.sub(r"[^\w\s]", "", input_string)

                # Task 2: Remove leading and trailing spaces, normalize multiple spaces
                cleaned_string = " ".join(cleaned_string.split())

                return cleaned_string

            for index, words in enumerate(corpus):
                corpus[index] = [preprocess_string(i) for i in words]
            return corpus


        def add_subject_subject_areas(corpus, papers):
            for index, words in enumerate(corpus):
                if "publication" in papers[index] and papers[index]["publication"]:
                    if (
                        "subject_areas" in papers[index]["publication"]
                        and papers[index]["publication"]["subject_areas"]
                    ):
                        subject_areas = papers[index]["publication"]["subject_areas"]
                        if isinstance(subject_areas, list):
                            corpus[index].extend(subject_areas)
                        else:
                            subject_areas = subject_areas.split(";")
                            subject_areas = [_.strip() for _ in subject_areas ]
                        corpus[index].extend(subject_areas)
            return corpus

        def remove_word(corpus, remove_list):
            remove_list = [i.lower() for i in remove_list]
            for index, words in enumerate(corpus):
                for remove_word in remove_list:
                    corpus[index] = [i for i in words if i.lower() != remove_word]
            return corpus

        def strip_strings_in_nested_list(nested_list):
            if isinstance(nested_list, list):
                return [
                    strip_strings_in_nested_list(item)
                    if isinstance(item, list)
                    else str(item).strip()
                    for item in nested_list
                ]
            else:
                return str(nested_list).strip()

        def drop_low_frequency_words_in_corpus( corpus, vocab, frequency_words):
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

        def english_strings_only(input_list):
            english_list = []
            for string in input_list:
                # This checks if all characters are either English alphabets or symbols
                if all((ord(char) < 128) for char in string):
                    english_list.append(string)
            return english_list

        def lemmatize_with_bert(corpus, vocab_set,language,retry=1):
            def process_paraphrases(unique_vocab_list, model_name):
                model = SentenceTransformer(model_name)
                paraphrases = util.paraphrase_mining(model, unique_vocab_list)
                return paraphrases

            def word_in_exclude_list(word):
                for exclude_word in self.config["lemmatized_exclude"]:
                    if exclude_word in word:
                        return True
                return False

            # Define a function to calculate the edit distance
            def edit_distance(str1, str2):
                return nltk.edit_distance(str1, str2)

            valid_lemmatized_words = []



            model_name = self.config["english_model_name"] if language == 'en' else self.config["model_name"]
            paraphrases_threshold = self.config["english_paraphrases_threshold"] if language == 'en' else self.config["paraphrases_threshold"]
            edit_distance_threshold = self.config["english_edit_distance_threshold"] if language == 'en' else self.config["edit_distance_threshold"]
            print(f'model_name: {model_name},language: {language}, paraphrases_threshold: {paraphrases_threshold}')
            unique_vocab_list = list(set(vocab_set))

            if language == 'en':
                unique_vocab_list = english_strings_only(unique_vocab_list)
                print(f"English unique_vocab_list: {len(unique_vocab_list)}")

                paraphrases = process_paraphrases(unique_vocab_list, model_name)

            else:
                unique_vocab_list_copy = unique_vocab_list.copy()
                _ = english_strings_only(unique_vocab_list_copy)
                unique_vocab_list = [i for i in unique_vocab_list if i not in _]
                print(f"unique_vocab_list: {len(unique_vocab_list)}")

                paraphrases = process_paraphrases(unique_vocab_list, model_name)

            assert len(vocab_set) > 0




            # Iterate through the paraphrases and filter out the valid lemmatized words
            for i in range(retry):
                for paraphrase in tqdm(paraphrases):
                    score, i, j = paraphrase
                    i = unique_vocab_list[i]
                    j = unique_vocab_list[j]


                    if (score > paraphrases_threshold) and (
                        edit_distance(i, j) < edit_distance_threshold
                    ):

                        if word_in_exclude_list(i) or word_in_exclude_list(
                            j
                        ):  # if any word in exclude list, skip
                            continue
                        else:
                            valid_lemmatized_words.append((i, j))

            valid_lemmatized_words = list(set(valid_lemmatized_words))
            word_counts = Counter(vocab_set)

            # Function to swap the values of two keys in the dictionary
            def swap(dictionary, key1, key2):
                dictionary[key1], dictionary[key2] = dictionary[key2], dictionary[key1]

            # Iterate through the valid_lemmatized_words list
            for vocab_pairs in valid_lemmatized_words:
                vocab1, vocab2 = vocab_pairs

                # Check if the count of vocab2 is greater than vocab1
                if word_counts[vocab2] > word_counts[vocab1]:
                    # Swap the counts of vocab1 and vocab2
                    swap(word_counts, vocab1, vocab2)

            valid_lemmatized_words = list(set(valid_lemmatized_words))


            lemmatized_words_path = f"./datarun/{self.project_name}/pre_process/lemmatized_words_{language}.txt"
            with open(
                lemmatized_words_path, "w"
            ) as f:
                for i, j in valid_lemmatized_words:
                    f.write(f"{i},{j}\n")

            # Replace the words in the corpus
            for sub_list in corpus:
                for index, word in enumerate(sub_list):
                    for vocab_pairs in valid_lemmatized_words:
                        vocab1, vocab2 = vocab_pairs
                        if word == vocab2:
                            sub_list[index] = vocab1

            print(f"Lemmatied  {len(valid_lemmatized_words)} words! in {language}")
            return corpus

        papers = self.read_data(self.config["source_data"])["papers"]


        def get_vocab_set(corpus):
            vocab_set = set()
            for words in corpus:
                for word in words:
                    vocab_set.add(word)
            return vocab_set


        remove_list = [
            "keywords:",
            "index_keywords:",
            "author_keywords:",
            "subject_keywords:",
            "application_keywords:",
            "this title:",
            "",
            "title",
        ]

        if self.config["keyword_extraction"].lower() == "gpt":
            # papers_invalid_index = self.get_papers_invalid_index(papers)
            # print('len(papers_valid_index): ', len(papers_invalid_index))
            corpus = self.get_gpt_unique_corpus(papers)
        elif self.config["keyword_extraction"].lower() == "bert":
            corpus = self.get_bert_unique_corpus(papers)
        else:
            raise ValueError("corpus_source must be either gpt or bert")

        if self.config["process_augmentation"]:
            corpus = self.process_augmentation(corpus)

        if self.config["add_subject_areas"]:
            corpus = add_subject_subject_areas(corpus, papers)


        self.write_corpus(
            corpus,
            f"./datarun/{self.project_name}/pre_process/before_pre_process.txt",
        )

        print(f"start pre-processing")
        corpus = remove_short_words(corpus)
        corpus = remove_numbers_from_sublists(corpus)
        corpus = strip_strings_in_nested_list(corpus)
        corpus = lower_nested_list(corpus)
        corpus = remove_hyphen(corpus)
        corpus = remove_keywords(corpus, remove_list)
        corpus = remove_corpus_parentheses(corpus)
        corpus = filter_sublists_max_len(corpus, max_len=self.config["word_max_len"])
        corpus = remove_punctuation_except_spaces(corpus)
        corpus = remove_word(corpus, self.config["exclude_words"])
        corpus = lower_nested_list(corpus)



        if self.config["paraphrases_threshold"] != 1 and self.config["paraphrases_threshold"] :


            print(f"start lemmatization in Chinese")
            _corpus = corpus.copy()
            _vocab = self.get_vocab(corpus)
            _corpus = drop_low_frequency_words_in_corpus(_corpus, _vocab, 2)
            _vocab_set = get_vocab_set(_corpus)
            corpus = lemmatize_with_bert(corpus, _vocab_set,language='zh')


        if self.config["english_paraphrases_threshold"] != 1 and self.config["english_paraphrases_threshold"] :
            print(f"start lemmatization in English")

            _corpus = corpus.copy()
            _corpus = drop_low_frequency_words_in_corpus(_corpus, self.get_vocab(corpus), 2)
            _vocab_set = get_vocab_set(_corpus)

            corpus = lemmatize_with_bert(corpus, _vocab_set,language='en')


        self.write_corpus(corpus, f"./datarun/{self.project_name}/pre_process/after_lemmatization.txt")

        # strip nest list
        self.write_corpus(corpus, self.corpus_path)

        vocab_set = get_vocab_set(corpus)

        print("len set(vocab) ", len(vocab_set))

        data = json.load(open(self.config["source_data"], "r"))
        papers = data["papers"]
        for i in range(len(papers)):
            papers[i]["keywords"] = corpus[i]
        with open(
            f"./datarun/{self.project_name}/pre_process/data_output.json", "w"
        ) as f:
            json.dump(data, f, ensure_ascii=False)

    def get_vocab(self, corpus):
        vocab = []
        for words in corpus:
            vocab.extend(words)
        return vocab

    def search_words(self, article, words):
        # Create a set of the words for faster lookup
        word_set = set(words)

        # print(colored(word_set,'green'))

        # Split the article into individual words
        article_words = article.split()

        # Use set intersection to find the common words
        common_words = word_set.intersection(article_words)

        # Return the common words as a list
        return list(common_words)
