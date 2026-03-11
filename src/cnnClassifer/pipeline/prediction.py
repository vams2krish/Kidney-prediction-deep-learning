import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from cnnClassifer import logger


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
        self.model = None
        self.loaded = False
    
    def load_model(self):
        """Load the model only when needed"""
        try:
            model_path = os.path.join("model", "model.h5")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            self.model = load_model(model_path)
            self.loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e


    
    def predict(self):
        try:
            # Load model if not loaded yet
            if not self.loaded:
                self.load_model()
            
            if self.model is None:
                raise ValueError("Model failed to load")
                
            imagename = self.filename
            
            # Check if file exists
            if not os.path.exists(imagename):
                raise FileNotFoundError(f"Image file not found: {imagename}")
            
            test_image = image.load_img(imagename, target_size = (224,224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = np.argmax(self.model.predict(test_image), axis=1)
            logger.info(f"Prediction result: {result}")

            if result[0] == 1:
                prediction = 'Tumor'
                return [{ "image" : prediction}]
            else:
                prediction = 'Normal'
                return [{ "image" : prediction}]
                
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            return [{"error": f"File not found: {str(e)}"}]
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return [{"error": f"Prediction failed: {str(e)}"}]
        