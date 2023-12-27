
import json
import os

import pandas as pd



class ReportProcess:
    def __init__(self, global_config):
        self.config = global_config["report"]
        self.project_name = global_config["project_name"]

        for k, v in self.config.items():
            if isinstance(v, str):
                self.config[k] = v.replace("{project_name}", self.project_name)

    def tending(self, file_name, source_data):
        def explode_corpus():

            data = json.load(open(source_data, "r"))
            papers = data["papers"]
            corpus = []
            for paper in papers:
                corpus.append(paper["keywords"])

            df_paper = pd.DataFrame(papers)
            df_paper["corpus"] = corpus
            df_paper["publication_date"] = pd.to_datetime(
                df_paper["publication_date"], errors="coerce"
            )
            df_paper["publication_year"] = df_paper["publication_date"].dt.year
            # df_paper['publication_quarter'] = df_paper['publication_date'].dt.quarter
            # df_paper['publication_label'] = df_paper['publication_year'].astype(str) + 'Q' + df_paper[ 'publication_quarter'].astype(str)

            df_paper["publication_year"] = df_paper["publication_year"].astype(str)
            # df_paper = df_paper[['publication_label','title','publication_date', 'abstract', 'corpus', 'publication_year', 'publication_quarter']]
            df_paper = df_paper[["corpus", "publication_year"]]

            # publication_year astype string
            df_paper["publication_year"] = df_paper["publication_year"].astype(str)
            df_exploded = df_paper.explode("corpus")
            return df_exploded

        # mkdir datarun/{project_name}/report
        if not os.path.exists(f"./datarun/{self.project_name}/report/"):
            os.makedirs(f"./datarun/{self.project_name}/report/")

        df_exploded = explode_corpus()

        # display(df_exploded)
        # 使用pivot_table函數進行透視操作
        df_transformed = df_exploded.pivot_table(
            index="corpus", columns="publication_year", aggfunc="size", fill_value=""
        )

        # 重設列索引
        df_transformed = df_transformed.reset_index()

        # 重新命名列名
        df_transformed.columns.name = None

        # sum df_transformed other column values
        df_transformed.fillna(0, inplace=True)

        def apply_sum(row):
            row = dict(row)
            sum = 0
            for k, v in row.items():
                if k != "corpus" and v != "":
                    sum += int(v)
            return sum

        df_transformed["sum"] = df_transformed.apply(apply_sum, axis=1)
        df_transformed = df_transformed.sort_values(by="sum", ascending=False)
        if not os.path.exists(f"./datarun/{self.project_name}/report/"):
            os.makedirs(f"./datarun/{self.project_name}/report/")

        # replace column name to year
        columns = [
            name.replace(".0", "").replace("nan", "unknown")
            for name in df_transformed.columns
        ]

        df_transformed.rename(
            columns=dict(zip(df_transformed.columns, columns)), inplace=True
        )
        df_transformed.to_excel(
            f"./datarun/{self.project_name}/report/{file_name}", index=False
        )
        # display(df_transformed)
