from os import getenv
from flask import Flask
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from flask import Flask, render_template
from flask_restful import Resource, Api
from flask import send_from_directory
import json

from flask import request, jsonify
from neo4j import GraphDatabase, basic_auth
import opentelemetry.sdk.resources

# Initialize TracerProvider
opentelemetry_resource = opentelemetry.sdk.resources.Resource(attributes={"service.name": "python-api-server"})
trace.set_tracer_provider(TracerProvider(resource=opentelemetry_resource))

# Initialize OTLP exporter and BatchSpanProcessor
oltp_endpoint = getenv("OTLP_ENDPOINT", default="localhost:4317")
otlp_exporter = OTLPSpanExporter(endpoint=oltp_endpoint, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)

trace.get_tracer_provider().add_span_processor(span_processor)
app = Flask(__name__)
# Instrument Flask
FlaskInstrumentor().instrument_app(app)
api = Api(app)

# Get a tracer
tracer = trace.get_tracer(__name__)

# Initialize Neo4j driver
driver = GraphDatabase.driver("neo4j+s://a597625d.databases.neo4j.io", auth=basic_auth("neo4j", "ak590OpsHi1xZ_CUa_UkjM4zriAaBmpVazsrlCOx-a0"))

# Initialize Flask app
app = Flask(__name__)
api = Api(app)


import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import Flask, render_template,redirect, url_for
import re
import pandas as pd
from datetime import datetime
import uuid
import numpy as np

# Data processing and visualization imports
from awesome_graph.build_map_process import BuildMapProcess
from awesome_graph.cluster_process import ClusterProcess
from awesome_graph import  logger



# Flask app config for file upload
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'xlsx', 'xls','csv','txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1000 * 1000 # 200MB max file size




# Upload Data processing
def data_process(project_name):
    from awesome_graph.word_pre_process import WordPreProcess
    from awesome_graph.report_process import ReportProcess
    from awesome_graph import get_configure

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
    global_config = get_configure(project_name)
    logger.info('project_name:' + global_config['project_name'])


    # # # # Step 3 Preprocess paper
    data_process = WordPreProcess(global_config)
    data_process.run()

    rp = ReportProcess(global_config)
    rp.tending(f'trending_report_keyword.xlsx', f"datarun/{project_name}/pre_process/data_output.json")  # ###Tending report




    project_name = global_config["project_name"]
    logger.info("project_name:" + global_config["project_name"])

    dp = ClusterProcess(global_config)

    # generate corpus
    data = json.load(open(f'datarun/{project_name}/keyword_extraction/data_output.json', "r"))
    papers = data["papers"]
    corpus = []
    for i in range(len(papers)):
        corpus.append(papers[i]["keywords"])

    with open(f"./datarun/{project_name}/keyword_extraction/corpus_output.json", "w") as file:
        json.dump(corpus, file, ensure_ascii=False, indent=4)



    vocab, corpus = dp.drop_low_frequency_words(3)
    vocab_set = list(set(vocab))
    bm = BuildMapProcess(global_config)
    dp = ClusterProcess(global_config)
    cooccur_matrix = dp.process_cooccur_matrix(vocab_set, corpus)
    logger.info('upload node without cluster')

    clusters = ['無分類' for i in range(len(vocab_set))]
    cluster_assignments = bm.create_cluster_assignments(clusters, vocab_set)
    graph = Graph(global_config["build_map"]["uri"], name=global_config["build_map"]["database_name"], auth=(global_config["build_map"]["user"], global_config["build_map"]["password"]), )

    # open /home/arden/Documents/gdrive/awesome_graph_2/datarun/AH_title_taiwanhistory_chatgpt/pre_process/data_output.json
    with open(f"./datarun/{project_name}/pre_process/data_output.json", "r") as file:
        data = json.load(file)

    vocab_year_assignments = bm.group_keywords_by_year(data["papers"], vocab_set)

    filted_cooccur = dp.filter_cooccur(cooccur_matrix, vocab_set)

    bm.bulk_build_node(cluster_assignments, graph, corpus, vocab_year_assignments, global_config["build_map"]["datarun_id"], vocab_set)
    bm.bulk_build_relationship(filted_cooccur, graph, global_config["build_map"]["datarun_id"])

    logger.info('Done')

# Transfer Scopus csv file to json file and normalize the schema
def transfer_and_save_json(file_path):
    '''
    transfer_and_save_json Scopus csv file to json file and normalize the schema
    :param file_path:
    :return:
    '''
    df = pd.read_csv(file_path, header=0, usecols=[ 'Authors', 'Year', 'Index Keywords', 'Title', ], sep=',', dtype=str)

    def clean_df(df):
        df['Title'].replace('', np.nan, inplace=True)
        df.dropna(subset=['Title'], inplace=True)
        return df

    def remove_brackets(df):
        df['Title'] = df['Title'].apply(lambda x: re.sub(r'\(.*?\)', '', x)) # removes half-width brackets
        df['Title'] = df['Title'].apply(lambda x: re.sub(r'（.*?）', '', x)) # removes full-width brackets
        return df
    def remove_duplicate_title(df):
        df['Title'] = df['Title'].str.strip()
        df = df.drop_duplicates(subset='Title', keep='first')
        return df

    def remove_empty_title(df):
        # Strip whitespace from 'Title' column
        df['Title'] = df['Title'].str.strip()

        # Remove rows where 'Title' after strip is empty
        df = df[df['Title'] != '']
        return df

    def count_chars(df):
        total_chars = df['Title'].str.len().sum()
        logger.info('Total number of characters in Title: {total_chars}'.format(total_chars=total_chars))


    before = len(df)
    logger.info(f'Number of rows before remove_brackets: {before}')
    df = clean_df(df)
    df = remove_brackets(df)
    df = remove_duplicate_title(df)
    df = remove_empty_title(df)
    after = len(df)
    logger.info(f'Number of rows after remove_brackets: {after}')
    count_chars(df)



    # Initialize the JSON data structure
    json_data = {
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "publication_types": None,
        "query": "",
        "since": "",
        "until": "",
        "databases": ["Scoups"],
        "limit": None,
        "limit_per_database": None,
        "number_of_papers": len(df),
        "number_of_papers_by_database": {"AH": len(df)},
        "papers": []
    }

    # Iterate through each row in the DataFrame and create JSON entries
    for index, row in df.iterrows():
        authors = [author.strip() for author in str(row["Authors"]).split(";")] if not pd.isna(row["Authors"]) else []


        # keywords = dict(row)
        # del keywords['Authors']
        keywords = list(dict(row).values())
        keywords = [str(x) for x in keywords]
        keywords = [x for x in keywords if x != 'nan']
        keywords = ' '.join(keywords).replace(',',';').split(';')
        keywords = [x.strip() for x in keywords]
        keywords = [x.lower() for x in keywords if x != '']
        keywords = list(set(keywords))
        keywords.extend(authors)
        paper = {
            "abstract":  '',
            "authors": authors,
            "categories": None,
            "citations": None,
            "comments": None,
            "databases": [],
            "doi": None,  # You can add DOI if available in the CSV
            "keywords": keywords,
            "number_of_pages": None,
            "pages": None,
            "publication": {
                "category": None,
                "cite_score": None,
                "is_potentially_predatory": False,
                "isbn": None,
                "issn": None,
                "publisher": None,
                "sjr": None,
                "snip": None,
                "subject_areas": str(row["Index Keywords"]).strip() if not pd.isna(row["Index Keywords"]) else None,
                "title": str(row["Title"]).strip() if not pd.isna(row["Title"]) else '',
            },
            "publication_date": f'{row["Year"].strip()[0:4]}-01-01' if not pd.isna(row["Year"]) else None,
            "selected": None,
            "title": str(row["Title"]).strip() if not pd.isna(row["Title"]) else None,
            "urls": '',
        }

        # Append the paper entry to the JSON data
        json_data["papers"].append(paper)

    return json_data




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        # project_name = request.form['project_name']
        # add uuid to project name
        # project_name = project_uuid

        project_uuid = str(uuid.uuid4())
        project_name = project_uuid

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the uploaded file to a temporary path
            project_name = secure_filename(project_name)
            temp_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], project_name.replace(' ','_')+'.csv')
            file.save(temp_csv_path)
            logger.info('file saved to ' + temp_csv_path)

            # Transfer the CSV data to JSON
            json_data = transfer_and_save_json(temp_csv_path)

            # Save the JSON data to a file with specified formatting
            file_save = './datarun/{project_name}/keyword_extraction/data_output.json'.format(project_name=project_name)
            os.makedirs(os.path.dirname(file_save), exist_ok=True)
            with open(file_save, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)
            logger.info(f"JSON data saved to '{file_save} with {len(json_data['papers'])} papers")



            # Copy config file to project folder
            with open('./datarun_config_template.json') as f:
                config = json.load(f)
                config['project_name'] = project_name
                config['data_path'] = file_save
                config['build_map']['datarun_id'] = project_uuid
                json.dump(config, open('./datarun/{project_name}/config.json'.format(project_name=project_name), 'w'), indent=4)

            # Run the data processing
            data_process(project_name)
            logger.info(f"Data processing completed")
            return render_template('upload/successfull.html', project_url=f'http://127.0.0.1:5000/graph/{project_name}'.format(project_name=project_name))

    return render_template('upload/index.html')



#=======================================================================================================================
from py2neo import Graph, Node, Relationship, NodeMatcher


# Neo4j data converat to graph data (d3.js)
def convert_to_graph(results, selected):
    import uuid
    nodes = {}
    relationships = []

    def get_node(node):
        if 'frequency' not in node:
            node['frequency'] = 1
        if node['name'] in selected:
            node['d3_node_font_size'] = 6
            node['d3_r'] = 50
            node['d3_collide'] = 4
            node['d3_text_y'] = 42

        # node['frequency'] = rescale_number(node['frequency'], 6, 25)
        results = {"id": node['name'], "labels": [node['cluster_number']], "properties": node}
        return results

    def get_releation(p):
        results = {"id": str(uuid.uuid4())[:5], "type": p[1], "startNode": p[0]['name'], "endNode": p[2]['name'], "properties": {}}
        return results

    for p in results:
        p = p['p']
        nodes[get_node(p[0])['id']] = (get_node(p[0]))
        nodes[get_node(p[2])['id']] = (get_node(p[2]))
        relationships.append(get_releation(p))

    nodes = list(nodes.values())

    return_data = {"results": [{"columns": ["user", "entity"], "data": [{"graph": {"nodes": nodes, "relationships": relationships}}]}], "errors": []}
    return return_data

# API for graph data (d3.js)
class Graph_Data(Resource):
    def post(self):
        with tracer.start_as_current_span("get graph data"):
            data = request.get_json()
            selected = data['selected']
            limit = data.get('limit', 300)  # defaults to 300 if not provided
            skip = data.get('skip', 0)  # defaults to 0 if not provided
            years = data.get('years', [])  # defaults to 0 if not provided
            cooccurrence = data.get('cooccurrence', 10)  # defaults to 4 if not provided

            cond = ""
            if selected:
                formatted_selected = ', '.join([f"'{s}'" for s in selected])
                cond += f" n.name IN [{formatted_selected}] "
            else:
                # AND    not type(r)= "very low"
                cond += f'r.concurrency_weight >= {cooccurrence}  '
            if years:
                formatted_years = ', '.join([str(y) for y in years])
                cond += f"AND ANY(year IN n.years WHERE year IN [{formatted_years}])"

            # datarun_id = 'ec8fcd8a-8123-11ee-b962-0242ac120002'
            datarun_id = None

            if datarun_id:
                cond += f"AND n.datarun_id='{datarun_id}'"

            # ORDER BY n.frequency DESC
            cypher_query = f'''
                MATCH p=(m)-[r]-(n) WHERE   {cond}    RETURN p   SKIP {skip} LIMIT {limit}
            '''

            print(cypher_query)

            with driver.session(database="neo4j") as session:
                results = session.execute_read(lambda tx: tx.run(cypher_query).data())

            print('COUNT', len(results))
            return convert_to_graph(results, selected)

# API for bootstrap select
class Graph_Data_Select(Resource):
    def get(self):
        def transform_result_to_optgroups(input_result):
            # first, group our entries by cluster_number
            grouped_dict = {}
            for res in input_result:
                if res['n.cluster_number'] not in grouped_dict:
                    grouped_dict[res['n.cluster_number']] = []
                grouped_dict[res['n.cluster_number']].append({'label': res['n.name'] + f' ({res["n.frequency"]})', "value": res['n.name']})

            # then, transform it to the desired format
            optgroups = []
            for cluster_key in sorted(grouped_dict.keys()):
                optgroups.append({'label': 'cluster_number_' + str(cluster_key), 'children': grouped_dict[cluster_key]})
            return optgroups

        # datarun_id = 'ec8fcd8a-8123-11ee-b962-0242ac120002'
        datarun_id = None
        cond = ""
        if datarun_id:
            cond += f"AND n.datarun_id='{datarun_id}'"

        cypher_query = f'''
        MATCH (n) WHERE NOT 'title' IN labels(n) AND n.frequency > 15 {cond} RETURN n.name, n.cluster_number, n.frequency 
        '''
        with driver.session(database="neo4j") as session:
            results = session.execute_read(lambda tx: tx.run(cypher_query).data())

        return transform_result_to_optgroups(results)



# Main endpoint
@app.route('/')
def main_endpoint():
    with tracer.start_as_current_span("main_endpoint"):
        file_path = "application.json"

        # Read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)
        print('Is production environment?', data["is_production_environment"])

        if data["is_production_environment"] == True:
            return render_template('d3/index.html', api_server_url='https://kg-api-sigma.vercel.app')
        else:
            return render_template('d3/index.html', api_server_url='http://127.0.0.1:5000')


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

api.add_resource(Graph_Data, '/graph_data')
api.add_resource(Graph_Data_Select, '/graph_data_select')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
