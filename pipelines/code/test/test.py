'''
Author:     Jixin Jia (Gin)
Date:       2022.05.18
Version:    1.0
Purpose:    This script will test our registered model against a pre-defined golden test set for gate-keepering 
'''

import os
import cv2
import argparse
from tensorflow.keras.models import load_model
from utils import model_utilities

# experiment tracking with azureml/mlflow
from azureml.core.run import Run
run = Run.get_context()

# get argparser
ap = argparse.ArgumentParser()
ap.add_argument('--mnt_path', type=str, help='path to test dataset on FUSE mount')
ap.add_argument('--test_input', type=str, default='mario', help='test dataset name')
ap.add_argument('--model_path', type=str, default='outputs/snn', help='path to trained model')
args = ap.parse_args()

# inherit training parameters
TEST_INPUT = os.path.join(args.mnt_path, args.test_input)
MODEL_PATH = args.model_path

# model preparation
IMG_SHAPE = (224, 224, 3)

# debug
print('[INFO] dataset path on remote compute:', TEST_INPUT)

# load trained model (.pb)
try:
    print('[INFO] Loading trained models (or saved weights and recompile)...', end='')
    model = load_model(MODEL_PATH,  compile=False)
    print('Done')

except Exception as e:
    print(e.args)

# create an outputs folder for tracking artifacts
os.makedirs('outputs', exist_ok=True)

# run inference
output_images, test_accuracy = model_utilities.test_inference(TEST_INPUT, model, IMG_SHAPE)

for idx, image in enumerate(output_images):
    # log images
    output_name = f'auto-test_{idx}.jpg'
    output_path = os.path.join('outputs', output_name)
    cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    run.log_image(name=output_name, path=output_path, plot=None, description='auto-test')
    

# log metrics
run.log('test_accuracy', test_accuracy)