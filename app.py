A. Commit message:
Replace risky shell execution with subprocess for improved security

B. Change summary:
Replaced `os.system` calls with Python's `subprocess.run` to avoid shell injection vulnerabilities and enhance security of process execution.

C. Compatibility Risk:
Medium 

D. Fixed Code:
from flask import Flask, request, jsonify, render_template
import os
import subprocess
from flask_cors import CORS, cross_origin
from cnnClassifer.utils.Common import decodeImage
from cnnClassifer.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    #subprocess.run(["python", "main.py"]) #use either one
    subprocess.run(["dvc", "repro"])    #use either one
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #for AWS