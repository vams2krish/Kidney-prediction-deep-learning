from flask import Flask, request, jsonify, render_template
import os
from pathlib import Path
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
        try:
            self.classifier = PredictionPipeline(self.filename)
        except Exception as e:
            print(f"Warning: Could not initialize prediction pipeline: {str(e)}")
            self.classifier = None


# Initialize ClientApp before defining routes
clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')



@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    #os.system("python main.py") #use either one
    os.system("dvc repro")    #use either one
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if clApp.classifier is None:
            return jsonify({"error": "Prediction pipeline not initialized"}), 500
            
        image = request.json.get('image')
        if not image:
            return jsonify({"error": "No image provided"}), 400
            
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) #for AWS

