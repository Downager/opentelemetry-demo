# Python example app
## Setup
```bash
python3 -m venv venv # Optional
source ./venv/bin/activate # Optional
pip3 install -r requirements.txt
```

## Start the app
```bash
python3 app.py
```

## Example requests
```
❯ curl localhost:8080/
Hello, world!
❯ curl localhost:8080/another
This is another endpoint!
```