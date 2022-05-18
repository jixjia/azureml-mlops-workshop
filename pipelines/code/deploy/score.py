'''
Author:     Jixin Jia (Gin)
Date:       2022.05.18
Version:    1.0
Purpose:    Create this scoring function to serve our custom Siamese Network as an API. 
            This scripts also defines how to handle input and outputs with ability to add schema validation
            and auto generate Swagger definition (not covered in today's tutorial)
'''

import os
import cv2
import base64
import json
import numpy as np
from urllib.parse import urlparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# mandatory init() function for loading model
def init():
    global model

    # AZUREML_MODEL_DIR is an built-in Environment Variable created during deployment 
    # ./azureml-models/$MODEL_NAME/$VERSION
    MODEL_PATH = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'snn')

    # load serialized trained model (.pb)
    try:
        print('[INFO] Loading trained models (or saved weights and recompile)...', end='')
        model = load_model(MODEL_PATH,  compile=False)
        print('Done')

    except Exception as e:
        print('[DEBUG]', e.args)

# mandatory run() function for handling input/outputs as a webservice
def run(raw_data):

    # parse json payload
    data = json.loads(raw_data)

    try:
        query = preprocess(data['query'])
        template = preprocess(data['template'])
            
    except Exception as e:
        return {'message':'You have successfully reached model endpoint. Make a POST with a valid base64 encoded Query and Template image to test our Custom Siamese Network model.'}


    # run inference (similarity score)
    
    preds = model.predict([template, query])
    similarity_score = preds[0][0]
    
    return float(similarity_score)


def preprocess(image_base64):
    # parse base64 image
    image = base64.b64decode(image_base64)
    image = np.asarray(bytearray(image), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    
    resize = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    resize = img_to_array(resize)
    resize = np.array(resize, dtype='float')/255.0
    resize = np.expand_dims(resize, axis=0)

    return resize