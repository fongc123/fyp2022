import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/', methods = [ 'GET' ])
def hello():
    return "hello world"

@app.route('/api/predict', methods = [ 'POST' ])
def predict():
    pass

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)