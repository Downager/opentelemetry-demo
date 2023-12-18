from os import getenv
from flask import Flask
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Initialize Flask app
app = Flask(__name__)

# Initialize TracerProvider
resource = Resource(attributes={"service.name": "python-example-app"})
trace.set_tracer_provider(TracerProvider(resource=resource))

# Initialize OTLP exporter and BatchSpanProcessor
oltp_endpoint = getenv("OTLP_ENDPOINT", default="localhost:4317")
otlp_exporter = OTLPSpanExporter(endpoint=oltp_endpoint, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)

trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument Flask
FlaskInstrumentor().instrument_app(app)

# Get a tracer
tracer = trace.get_tracer(__name__)

# Sample functions to trace
def some_work():
    with tracer.start_as_current_span("doing-some-work"):
        print("Doing busy work")

@app.route("/")
def hello():
    with tracer.start_as_current_span("hello-span"):
        some_work()
        return "Hello, world!"

@app.route("/another")
def another_endpoint():
    with tracer.start_as_current_span("another-span"):
        some_work()
        return "This is another endpoint!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
