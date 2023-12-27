# Standard library imports
from os import getenv
import json
import socket

# Third-party imports
from flask import Flask, render_template, request
from flask_restful import Resource, Api
import requests

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
import opentelemetry.sdk.resources
from opentelemetry import context
from opentelemetry.propagate import inject


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
        "service.name": "flask-web-server",
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


# New function to call Database Service
def call_database_service(endpoint, method="get", data=None):
    db_service_url = getenv("DB_SERVICE_URL", default="http://localhost:5001")
    url = f"{db_service_url}/{endpoint}"

    # Create a new context and inject it into the headers
    headers = {}
    current_ctx = context.get_current()
    inject(headers, context=current_ctx)

    if method == "get":
        response = requests.get(url, headers=headers)
    else:  # Assuming 'post'
        response = requests.post(url, json=data, headers=headers)

    return response.json()


class Graph_Data(Resource):
    def post(self):
        data = request.get_json()
        # call Database Service for graph data
        return call_database_service("graph_data", method="post", data=data)


class Graph_Data_Select(Resource):
    def get(self):
        # call Database Service for graph data select
        return call_database_service("graph_data_select")


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
