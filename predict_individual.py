from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import math
import csv
import cv2


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "ggb_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 6
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


def predictImage(imageName, model, cfg, classes):
    # print("=========i in load mask is: "+ str(i))
    # image = dataset.load_image(i)
    # imageName = dataset.source_image_link(i)
    image = cv2.imread(imageName,cv2.IMREAD_COLOR)
    
    # mask, _ = dataset.load_mask(i)
    
    print(imageName, image.shape)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]
    # define subplot
    print("Predict function return: ",yhat)
    # pyplot.subplot(n_images, 2, i*2+1)
    # # plot raw pixel data

    # pyplot.imshow(image)
    # pyplot.title('Actual')
    # # plot masks
    # for j in range(mask.shape[2]):
    # 	pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
    # # get the context for drawing boxes
    # pyplot.subplot(n_images, 2, i*2+2)
    # # plot raw pixel data
    # pyplot.imshow(image)
    # pyplot.title('Predicted')
    # ax = pyplot.gca()
    # # plot each box
    loopNum=0
    for box in yhat['rois']:
        # get coordinates
        y1, x1, y2, x2 = box
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        predictedLabelID = yhat['class_ids'][loopNum]
        # predictedLabelName = Dataset.class_names[predictedLabelID]
        # draw the box
        # ax.add_patch(rect)
        # print((x1,y1),(x2,y2))
        detectedLabel = image[y1:y1+height, x1:x1+width]
        cv2.imwrite(str(imageName).split(".")[0]+"_"+str(loopNum)+"_"+str(classes[predictedLabelID-1])+".jpg",detectedLabel)
        loopNum+=1


def predict(testFile, modelName, classes):
    # load the train dataset
    # train_set = Dataset()
    # train_set.load_dataset(datasetDir, classes, is_train=True)
    # train_set.prepare()
    # print('Train: %d' % len(train_set.image_ids))
    # # load the test dataset
    # test_set = Dataset()
    # test_set.load_dataset(datasetDir, classes, is_train=False)
    # test_set.prepare()
    # print('Test: %d' % len(test_set.image_ids))
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./users/user1/dataset/models/', config=cfg)
    # load model weights
    model_path = 'users/user1/dataset/models/mask_rcnn_ggb_cfg_0004.h5'
    model.load_weights(model_path, by_name=True)
    # plot predictions for train dataset
    # plot_actual_vs_predicted(train_set, model, cfg)
    # plot predictions for test dataset
    # plot_actual_vs_predicted(test_set, model, cfg)
    predictImage(testFile, model, cfg, classes)