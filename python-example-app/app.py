# Standard library imports
from os import getenv
import json
import socket

# Third-party imports
from flask import Flask, render_template, request
from flask_restful import Resource, Api
from neo4j import GraphDatabase, basic_auth

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
import opentelemetry.sdk.resources


# Getting the hostname
hostname = socket.gethostname()

# Getting the IP address
try:
    ip_address = socket.gethostbyname(hostname)
except socket.gaierror:
    # Couldn't get the IP address
    ip_address = "Unavailable"

# Initialize TracerProvider
opentelemetry_resource = opentelemetry.sdk.resources.Resource(
    attributes={
        "service.name": "python-example-app",
        "host.name": hostname,
        "host.ip": ip_address,
    }
)
trace.set_tracer_provider(TracerProvider(resource=opentelemetry_resource))

# Initialize OTLP exporter and BatchSpanProcessor
oltp_endpoint = getenv("OTLP_ENDPOINT", default="localhost:4317")
otlp_exporter = OTLPSpanExporter(endpoint=oltp_endpoint, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)

trace.get_tracer_provider().add_span_processor(span_processor)

# Get a tracer
tracer = trace.get_tracer(__name__)

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Auto-instrument flask
FlaskInstrumentor().instrument_app(app)

import json

driver = GraphDatabase.driver(
    "neo4j+s://1e02c965.databases.neo4j.io",
    auth=basic_auth("neo4j", "zbvV0gWwxbcXrnlIgoAy_LpzZTvc2PatWuKiA0fDsSc"),
)


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
        with tracer.start_as_current_span("post graph data") as span:
            data = request.get_json()
            selected = data["selected"]
            limit = data.get("limit", 300)
            skip = data.get("skip", 0)
            years = data.get("years", [])
            cooccurrence = data.get("cooccurrence", 10)

            # Add attributes to the span
            span.set_attribute("selected_items", len(selected) if selected else 0)
            span.set_attribute("limit", limit)
            span.set_attribute("skip", skip)
            span.set_attribute("years", years)
            span.set_attribute("cooccurrence", cooccurrence)

            with tracer.start_as_current_span("construct condition") as condition_span:
                cond = ""
                if selected:
                    formatted_selected = ", ".join([f"'{s}'" for s in selected])
                    cond += f" n.name IN [{formatted_selected}] "
                else:
                    cond += f"r.concurrency_weight >= {cooccurrence}  "
                if years:
                    formatted_years = ", ".join([str(y) for y in years])
                    cond += (
                        f"AND ANY(year IN n.years WHERE year IN [{formatted_years}])"
                    )

                # Add an attribute for the condition
                condition_span.set_attribute("condition", cond)

            datarun_id = None
            if datarun_id:
                cond += f"AND n.datarun_id='{datarun_id}'"

            with tracer.start_as_current_span("build cypher query") as query_span:
                cypher_query = f"""
                    MATCH p=(m)-[r]-(n) WHERE   {cond}    RETURN p   SKIP {skip} LIMIT {limit}
                """
                query_span.set_attribute("cypher_query", cypher_query)

            print(cypher_query)

            with tracer.start_as_current_span("execute query") as execute_span:
                with driver.session(database="neo4j") as session:
                    results = session.execute_read(
                        lambda tx: tx.run(cypher_query).data()
                    )

                # Add an attribute for the count of results
                execute_span.set_attribute("result_count", len(results))

            print("COUNT", len(results))
            return convert_to_graph(results, selected)


class Graph_Data_Select(Resource):
    def get(self):
        with tracer.start_as_current_span("get graph data select") as select_span:
            datarun_id = None
            cond = ""
            if datarun_id:
                cond += f"AND n.datarun_id='{datarun_id}'"
                select_span.set_attribute("datarun_id", datarun_id)
            else:
                select_span.set_attribute("datarun_id", "none")

            cypher_query = f"""
            MATCH (n) WHERE NOT 'title' IN labels(n) AND n.frequency > 15 {cond} RETURN n.name, n.cluster_number, n.frequency 
            """
            select_span.set_attribute("cypher_query", cypher_query)

            with tracer.start_as_current_span("execute cypher query") as execute_span:
                with driver.session(database="neo4j") as session:
                    results = session.execute_read(
                        lambda tx: tx.run(cypher_query).data()
                    )
                execute_span.set_attribute("result_count", len(results))

            def transform_result_to_optgroups(input_result):
                with tracer.start_as_current_span("transform result") as transform_span:
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

                    optgroups = []
                    for cluster_key in sorted(grouped_dict.keys()):
                        optgroups.append(
                            {
                                "label": "cluster_number_" + str(cluster_key),
                                "children": grouped_dict[cluster_key],
                            }
                        )

                    transform_span.set_attribute(
                        "number_of_clusters", len(grouped_dict)
                    )
                    return optgroups

            transformed_results = transform_result_to_optgroups(results)
            select_span.set_attribute(
                "transformed_results_count", len(transformed_results)
            )
            return transformed_results


class Graph_Data_Select(Resource):
    def get(self):
        with tracer.start_as_current_span("get graph data select") as select_span:
            datarun_id = None
            cond = ""
            if datarun_id:
                cond += f"AND n.datarun_id='{datarun_id}'"
                select_span.set_attribute("datarun_id", datarun_id)
            else:
                select_span.set_attribute("datarun_id", "none")

            cypher_query = f"""
            MATCH (n) WHERE NOT 'title' IN labels(n) AND n.frequency > 15 {cond} RETURN n.name, n.cluster_number, n.frequency 
            """
            select_span.set_attribute("cypher_query", cypher_query)

            with tracer.start_as_current_span("execute cypher query") as execute_span:
                with driver.session(database="neo4j") as session:
                    results = session.execute_read(
                        lambda tx: tx.run(cypher_query).data()
                    )
                execute_span.set_attribute("result_count", len(results))

            def transform_result_to_optgroups(input_result):
                with tracer.start_as_current_span("transform result") as transform_span:
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

                    optgroups = []
                    for cluster_key in sorted(grouped_dict.keys()):
                        optgroups.append(
                            {
                                "label": "cluster_number_" + str(cluster_key),
                                "children": grouped_dict[cluster_key],
                            }
                        )

                    transform_span.set_attribute(
                        "number_of_clusters", len(grouped_dict)
                    )
                    return optgroups

            transformed_results = transform_result_to_optgroups(results)
            select_span.set_attribute(
                "transformed_results_count", len(transformed_results)
            )
            return transformed_results


api.add_resource(Graph_Data, "/graph_data")
api.add_resource(Graph_Data_Select, "/graph_data_select")


@app.route("/")
def main_endpoint():
    with tracer.start_as_current_span("main_endpoint") as main_span:
        file_path = "application.json"
        main_span.set_attribute("file_path", file_path)

        # Read the JSON file
        with tracer.start_as_current_span("read_json_file"):
            with open(file_path, "r") as file:
                data = json.load(file)
            print("Is production environment?", data["is_production_environment"])

        main_span.set_attribute(
            "is_production_environment", data["is_production_environment"]
        )

        if data["is_production_environment"] == True:
            api_server_url = "https://kg-api-sigma.vercel.app"
        else:
            api_server_url = "http://127.0.0.1:5000"

        main_span.set_attribute("api_server_url", api_server_url)
        return render_template("d3/index.html", api_server_url=api_server_url)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
