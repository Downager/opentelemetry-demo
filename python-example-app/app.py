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
opentelemetry_resource = opentelemetry.sdk.resources.Resource(
    attributes={"service.name": "python-example-app"}
)
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

# Auto-instrument flask
FlaskInstrumentor().instrument_app(app)

import json

driver = GraphDatabase.driver(
    "neo4j+s://1e02c965.databases.neo4j.io",
    auth=basic_auth("neo4j", "zbvV0gWwxbcXrnlIgoAy_LpzZTvc2PatWuKiA0fDsSc"),
)


# Initialize Flask app
app = Flask(__name__)
api = Api(app)


def convert_to_graph(results, selected):
    import uuid

    nodes = {}
    relationships = []

    def get_node(node):
        if "frequency" not in node:
            node["frequency"] = 1
        if node["name"] in selected:
            node["d3_node_font_size"] = 6
            node["d3_r"] = 50
            node["d3_collide"] = 4
            node["d3_text_y"] = 42

        # node['frequency'] = rescale_number(node['frequency'], 6, 25)
        results = {
            "id": node["name"],
            "labels": [node["cluster_number"]],
            "properties": node,
        }
        return results

    def get_releation(p):
        results = {
            "id": str(uuid.uuid4())[:5],
            "type": p[1],
            "startNode": p[0]["name"],
            "endNode": p[2]["name"],
            "properties": {},
        }
        return results

    for p in results:
        p = p["p"]
        nodes[get_node(p[0])["id"]] = get_node(p[0])
        nodes[get_node(p[2])["id"]] = get_node(p[2])
        relationships.append(get_releation(p))

    nodes = list(nodes.values())

    return_data = {
        "results": [
            {
                "columns": ["user", "entity"],
                "data": [{"graph": {"nodes": nodes, "relationships": relationships}}],
            }
        ],
        "errors": [],
    }
    return return_data


class Graph_Data(Resource):
    def post(self):
        with tracer.start_as_current_span("get graph data"):
            data = request.get_json()
            selected = data["selected"]
            limit = data.get("limit", 300)  # defaults to 300 if not provided
            skip = data.get("skip", 0)  # defaults to 0 if not provided
            years = data.get("years", [])  # defaults to 0 if not provided
            cooccurrence = data.get("cooccurrence", 10)  # defaults to 4 if not provided

            cond = ""
            if selected:
                formatted_selected = ", ".join([f"'{s}'" for s in selected])
                cond += f" n.name IN [{formatted_selected}] "
            else:
                # AND    not type(r)= "very low"
                cond += f"r.concurrency_weight >= {cooccurrence}  "
            if years:
                formatted_years = ", ".join([str(y) for y in years])
                cond += f"AND ANY(year IN n.years WHERE year IN [{formatted_years}])"

            # datarun_id = 'ec8fcd8a-8123-11ee-b962-0242ac120002'
            datarun_id = None

            if datarun_id:
                cond += f"AND n.datarun_id='{datarun_id}'"

            # ORDER BY n.frequency DESC
            cypher_query = f"""
                MATCH p=(m)-[r]-(n) WHERE   {cond}    RETURN p   SKIP {skip} LIMIT {limit}
            """

            print(cypher_query)

            with driver.session(database="neo4j") as session:
                results = session.execute_read(lambda tx: tx.run(cypher_query).data())

            print("COUNT", len(results))
            return convert_to_graph(results, selected)


class Graph_Data_Select(Resource):
    def get(self):
        def transform_result_to_optgroups(input_result):
            # first, group our entries by cluster_number
            grouped_dict = {}
            for res in input_result:
                if res["n.cluster_number"] not in grouped_dict:
                    grouped_dict[res["n.cluster_number"]] = []
                grouped_dict[res["n.cluster_number"]].append(
                    {
                        "label": res["n.name"] + f' ({res["n.frequency"]})',
                        "value": res["n.name"],
                    }
                )

            # then, transform it to the desired format
            optgroups = []
            for cluster_key in sorted(grouped_dict.keys()):
                optgroups.append(
                    {
                        "label": "cluster_number_" + str(cluster_key),
                        "children": grouped_dict[cluster_key],
                    }
                )
            return optgroups

        # datarun_id = 'ec8fcd8a-8123-11ee-b962-0242ac120002'
        datarun_id = None
        cond = ""
        if datarun_id:
            cond += f"AND n.datarun_id='{datarun_id}'"

        cypher_query = f"""
        MATCH (n) WHERE NOT 'title' IN labels(n) AND n.frequency > 15 {cond} RETURN n.name, n.cluster_number, n.frequency 
        """
        with driver.session(database="neo4j") as session:
            results = session.execute_read(lambda tx: tx.run(cypher_query).data())

        return transform_result_to_optgroups(results)


api.add_resource(Graph_Data, "/graph_data")
api.add_resource(Graph_Data_Select, "/graph_data_select")


@app.route("/")
def main_endpoint():
    with tracer.start_as_current_span("main_endpoint"):
        file_path = "application.json"

        # Read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)
        print("Is production environment?", data["is_production_environment"])

        if data["is_production_environment"] == True:
            return render_template(
                "d3/index.html", api_server_url="https://kg-api-sigma.vercel.app"
            )
        else:
            return render_template(
                "d3/index.html", api_server_url="http://127.0.0.1:5000"
            )


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
