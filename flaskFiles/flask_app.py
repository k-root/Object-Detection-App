from flask import Flask
from flask import current_app, Flask, redirect, url_for, Blueprint, send_from_directory, render_template, request, Response
from flask_cors import CORS
import requests
import re
import json
import train
import evaluate
import predict
import predict_individual
import time

import os

flask_app = Blueprint('flask_app', __name__)


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
 
def unzipDataset(zipFileName):
    import zipfile
    path_to_zip_file = "dataset/"+zipFileName
    directory_to_extract_to = "datasets/"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    return directory_to_extract_to+zipFileName.split(".")[0]

@flask_app.route("/train",methods=['GET', 'POST'])
def runTrain():
    # modelName = request.args.get("modelName")
    data = request.json.get("modelInput")
    print(data)
            # {'importClassCount': 1, 'importClasses': {'0': '2'},
            # 'importFolder': 'Type 3 (FRC)-20200225T105311Z-001.zip', 'select': 'chooseTrain',
            # 'selectModelName': 'mask_rcnn_kangaroo_cfg_0005.h5', 'epochs': 1,
            # 'learningRate': 0.1}
    numClasses = data['importClassCount']
    dataset_dir = unzipDataset(data['importFolder'])
    datasetName = data['importFolder'].split(".")[0]
    numEpochs = data['epochs']
    learningRate = data['learningRate']
    modelSelected = data['selectModelName']
    weights = "models/"+modelSelected
    # classes = request.args.get("classes")
    # dataset_dir = request.args.get("datasetDir")
    classes = ['Flanged Thickness', 'Pin Indent Pattern', 'Grease Hole Angular Location', 'Length', 'ID', 'Grease Hole Length Location', 'ID Corner Break', 'OD Chamfer Length', 'Grease Hole Diameter', 'OD Chamfer Angle', 'Flanged Diameter', 'OD', 'Flanged Bend Radius']
    dirList = os.listdir("./")

    print(dirList)
    # model_dir = "training"####Change to dynamic
    # pipeline_config_path = "training/ssd_mobilenet_v2_coco.config"####Change to dynamic
    # num_train_steps = None
    # eval_training_data=False
    # if not eval_training_data:
    #     checkpoint_dir=None
    # else:
    #     os.mkdir("checkpoint/"+modelName)
    # if model_dir not in dirList or not os.path.isdir(model_dir):
    #     os.mkdir("training")

    # weights = "mask_rcnn_coco.h5"##get this from frontend

    # train.training(model_dir,pipeline_config_path,num_train_steps, eval_training_data, checkpoint_dir)
    # dataset_dir = "datasets/ggbDatasetStraightFlangedFRC"####Change to dynamic
    if numEpochs==0 or not numEpochs:
        numEpochs = 10
    if learningRate==0 or not learningRate:
        learningRate = 0.001
    history = train.run_train(dataset_dir, weights, classes, numEpochs, learningRate)
    return json.dumps(history)


@flask_app.route("/test",methods=['GET', 'POST'])
def testModel():
    modelName = request.args.get("modelName")
    testFile = request.args.get("testFile")
    datasetDir = "datasets/ggbDatasetStraightFlangedFRC"

    classes = ['Flanged Thickness', 'Pin Indent Pattern', 'Grease Hole Angular Location', 'Length', 'ID', 'Grease Hole Length Location', 'ID Corner Break', 'OD Chamfer Length', 'Grease Hole Diameter', 'OD Chamfer Angle', 'Flanged Diameter', 'OD', 'Flanged Bend Radius']


    # modelName = request.args.get("modelName")
    predict.predict_main(datasetDir, modelName, testFile, classes)
    return "test"


@flask_app.route("/testIndividual",methods=['GET', 'POST'])
def testIndividualModel():
    modelName = request.args.get("modelName")
    testFile = request.args.get("testFile")
    datasetDir = "datasets/ggbDatasetStraightFlangedFRC"

    classes = ['Flanged Thickness', 'Pin Indent Pattern', 'Grease Hole Angular Location', 'Length', 'ID', 'Grease Hole Length Location', 'ID Corner Break', 'OD Chamfer Length', 'Grease Hole Diameter', 'OD Chamfer Angle', 'Flanged Diameter', 'OD', 'Flanged Bend Radius']


    # modelName = request.args.get("modelName")
    # predict.predict_main(datasetDir, modelName, testFile, classes)
    testFile = r"imageGGBTest\image3\pt$bb1212du-p1$a$en.jpg"
    predict_individual.predict(testFile, modelName, classes)
    return "test individual"


@flask_app.route("/evaluate",methods=['GET', 'POST'])
def evaluateModel():
    # modelName = request.args.get("modelName")
    # modelName = request.args.get("modelName")
    # testFile = request.args.get("testFile")
    datasetDir = "datasets/ggbDatasetStraightFlangedFRC"
    modelName = "users/user1/dataset/models/ggb_cfg20200226T0858/mask_rcnn_ggb_cfg_0004.h5"
    classes = ['Flanged Thickness', 'Pin Indent Pattern', 'Grease Hole Angular Location', 'Length', 'ID', 'Grease Hole Length Location', 'ID Corner Break', 'OD Chamfer Length', 'Grease Hole Diameter', 'OD Chamfer Angle', 'Flanged Diameter', 'OD', 'Flanged Bend Radius']
    evaluate.run_evaluate(datasetDir, modelName, classes)
    return "evaluate"


# @app.route("/getModels",methods=['GET'])
# def getModel():
#     # modelName = request.args.get("modelName")
#     models_list = os.listdir("models")
#     print(models_list)
#     return json.dumps({"models":models_list})

@flask_app.route('/zipfile', methods=['POST'])
def post():
    try:
        print(time.time())
        zipFile = request.files['file']
        zipFileName = zipFile.filename
        print(zipFileName)
        if zipFileName.split(".")[1]=="zip":
            if not os.path.isdir("dataset"):
                os.mkdir("dataset")
            zipFile.save(r"./dataset/"+ zipFileName)
        else:
            return json.dumps("Not a Zip")
        # print(request.json.get("files")) 
        # print('-------------------------------')
        # print(request.json.get("file_content"))
        print('-------------------------------')
        print('-------------------------------')
        print(time.time())
        return json.dumps('{"success"}')
    except:
        return json.dumps('{"Fail"}')
    # print(request.data.getvalue())
    
@flask_app.route('/getModelNames', methods=['GET'])
def getModelNames():
    modelsList = os.listdir("models")
    returnResp = []
    for model in modelsList:
        fileExt = model.split(".")[1]
        if fileExt in ["h5","pb"]:
            returnResp.append(model)
            print(model)
    return json.dumps(returnResp)

# @app.route("/evaluate",methods=['GET', 'POST'])
# def evaluateModel():
#     # modelName = request.args.get("modelName")
#     return "test"
# @app.route("/evaluate",methods=['GET', 'POST'])
# def evaluateModel():
#     # modelName = request.args.get("modelName")
#     return "test"

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=8080, debug=True)
