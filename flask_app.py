from flask import Flask
from flask import render_template, Blueprint, send_from_directory, request
from flask_cors import CORS
import requests
import re
import json
import train
# import evaluate
import predict

import os
app = Flask(__name__)
app.config["DEBUG"]=True
import tensorflow.compat.v1 as tf
import pathlib

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  print(model_dir)
  model = model = tf.compat.v2.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model
 

@app.route("/train",methods=['GET', 'POST'])
def runTrain():
    modelName = request.args.get("modelName")
    # classes = request.args.get("classes")
    classes = ['ID Corner Break','OD Chamfer Length','OD Chamfer Angle','Flange Length and Thickness','Flange Diameter','Flange Bend Radius']
    dirList = os.listdir("./")
    # if modelName in dirList:
    #     #load model
    #     pass
    # else:
    #     #download from zoo
    #     pass
    # model = load_model(modelName)
    # print(model)

    print(dirList)
    model_dir = "training"####Change to dynamic
    pipeline_config_path = "training/ssd_mobilenet_v2_coco.config"####Change to dynamic
    num_train_steps = None
    eval_training_data=False
    if not eval_training_data:
        checkpoint_dir=None
    else:
        os.mkdir("checkpoint/"+modelName)
    if model_dir not in dirList or not os.path.isdir(model_dir):
        os.mkdir("training")

    weights = "mask_rcnn_coco.h5"##get this from frontend

    # train.training(model_dir,pipeline_config_path,num_train_steps, eval_training_data, checkpoint_dir)
    dataset_dir = "ggbDataset"####Change to dynamic
    train.run_train(dataset_dir, weights, classes)
    return "train"+modelName


@app.route("/test",methods=['GET', 'POST'])
def testModel():
    modelName = request.args.get("modelName")
    testFile = request.args.get("testFile")

    # modelName = request.args.get("modelName")
    predict.predict_main(modelName. testFile)
    return "test"

@app.route("/evaluate",methods=['GET', 'POST'])
def evaluateModel():
    # modelName = request.args.get("modelName")
    return "test"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
