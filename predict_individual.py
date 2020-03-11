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
import os
import os.path
from keras import backend as K

class Dataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, classes, is_train=True):
		# define classes####Change to dynamic   ####classes=['class1', 'class2', 'class3', .....]
		for i in range(len(classes)):
			self.add_class("dataset", i+1, classes[i])
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		test_train_split = 0.7####train:0.7 test:0.3

		print("images dataset: "+images_dir+ " annotations dataset: "+annotations_dir)
		images = listdir(images_dir)
		annotations = listdir(annotations_dir)

		total_dataset_length = len(images)
		len_train_data = math.ceil(test_train_split*total_dataset_length)
		len_test_data = total_dataset_length - len_train_data

		print("Total dataset Length: "+ str(total_dataset_length)+"\ntrain length: "+str(len_train_data)+ "\ntest length :"+str(len_test_data))

		if is_train:
			for filenumber in range(len_train_data):
				image_id = images[filenumber][:-4]
				img_path = images_dir + images[filenumber]
				ann_path = annotations_dir + image_id + '.xml'
				self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
		
		else:
			for filenumber in range(len_test_data):
				image_id = images[filenumber][:-4]
				img_path = images_dir + images[filenumber]
				ann_path = annotations_dir + image_id + '.xml'
				self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

		# for filename in listdir(images_dir):####Change to dynamic####Change to dynamic
		# 	# extract image id
		# 	image_id = filename[:-4]
		# 	# skip bad images
		# 	if image_id in ['00090']:
		# 		continue
		# 	# skip all images after 150 if we are building the train set
		# 	if is_train and int(image_id) >= 150:
		# 		continue
		# 	# skip all images before 150 if we are building the test/val set
		# 	if not is_train and int(image_id) < 150:
		# 		continue
		# 	img_path = images_dir + filename
		# 	ann_path = annotations_dir + image_id + '.xml'
		# 	# add to dataset
		# 	self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		objects = root.findall('.//object')
		for obj in objects:
			name = obj.find('name').text
			bndbox = obj.find('bndbox')
		# for box in root.findall('.//bndbox'):
			xmin = int(bndbox.find('xmin').text)
			ymin = int(bndbox.find('ymin').text)
			xmax = int(bndbox.find('xmax').text)
			ymax = int(bndbox.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append([name, coors])
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		#print("-----------------------annotation path-------------------" + path)
		boxes, w, h = self.extract_boxes(path)
		#print("-----------------------boxes-------------------", boxes)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			item = boxes[i]##[name, [coor]]
			#print("-----------------------individual box-------------------", item)
			box = item[1]
			name = item[0]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(name))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "ggb_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 13
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


def predictImage(imageName, model, cfg, class_names):
    # print("=========i in load mask is: "+ str(i))
    # image = dataset.load_image(i)
    # imageName = dataset.source_image_link(i)
    print("Entered predict image")
    image = cv2.imread(imageName,cv2.IMREAD_COLOR)
    
    # mask, _ = dataset.load_mask(i)
    
    print(imageName, image.shape)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    print("checkpoint 10")
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    print("checkpoint 11")
    # make prediction
    yhat = model.detect(sample, verbose=1)[0]
    # print("checkpoint 11a")
    # y_classes = yhat.argmax(axis=-1)
    # print("checkpoint 11b, y_classes: ", y_classes)

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
    print(class_names,":::Classes:::")
    if not os.path.isdir(str(imageName).split(".")[0]):
        os.mkdir(str(imageName).split(".")[0])
    for box in yhat['rois']:
        print("enter loop of ROIs for ",str(loopNum))
        # get coordinates
        y1, x1, y2, x2 = box
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        predictedLabelID = yhat['class_ids'][loopNum]
        predictedLabelName = class_names[predictedLabelID-1]
        print("Predicted Label name: ", predictedLabelName)
        # draw the box
        # ax.add_patch(rect)
        # print((x1,y1),(x2,y2))
        print("writing individual labels")
        detectedLabel = image[y1:y1+height, x1:x1+width]
        
        cv2.imwrite(str(imageName).split(".")[0]+"/"+str(predictedLabelID-1)+"_"+str(predictedLabelName)+".jpg",detectedLabel)
        print("writing individual labels compleeteddddd+++++")
        print("drawing bbox")

        cv2.rectangle(image, (x1,y1), (x2,y2), (0, 0, 255), 5)
        print("putting text on bbox")
        cv2.putText(image, str(predictedLabelID), (x2+10,y2), cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 0, 0), 5)
        print("drawing bbox done")
        loopNum+=1
    print("writing images")
    imageDir, imgNameWithoutPath = imageName.split("/")
    print(imageDir, imgNameWithoutPath)
    imageDestPath = imageDir+"/predict/"+imgNameWithoutPath
    cv2.imwrite(imageDestPath ,image)
    print("writing images completed")
    clearSession()
    return imageDestPath

def predict(testFile, modelName, datasetDir, classes):
    print("entered predict for "+testFile)

    # # load the train dataset
    # train_set = Dataset()
    # train_set.load_dataset(datasetDir, classes, is_train=True)
    # train_set.prepare()
    # class_names = train_set.class_names
    # print("class_names: ",class_names)


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
    # model_path = 'users/user1/dataset/models/mask_rcnn_ggb_cfg_0004.h5'
    print("loaded config, loading model")
    model_path = "users/user1/dataset/models/mask_rcnn_ggb_cfg_15e_06032020.h5"
    model.load_weights(model_path, by_name=True)
    print("Loaded model Weights")
    # plot predictions for train dataset
    # plot_actual_vs_predicted(train_set, model, cfg)
    # plot predictions for test dataset
    # plot_actual_vs_predicted(test_set, model, cfg)
    imagePath = predictImage(testFile, model, cfg, classes)
    clearSession()
    return imagePath

def clearSession():
    K.clear_session()
if __name__ == "__main__":
    imagesDir = "individualTestImages/"
    dirFiles = os.listdir(imagesDir)  
    # imageFiles = []
    classes = ['Flanged Thickness', 'Pin Indent Pattern', 'Grease Hole Angular Location', 'Length', 'ID', 'Grease Hole Length Location', 'ID Corner Break', 'OD Chamfer Length', 'Grease Hole Diameter', 'OD Chamfer Angle', 'Flanged Diameter', 'OD', 'Flanged Bend Radius']
    modelName = "ggb"

    for files in dirFiles:
        if not os.path.isdir(imagesDir+files):
            # imageFiles.append(files)
            print(files)
            testFile = imagesDir+files
            predict(testFile, modelName)
    # testFile = r"imageGGBTest\image3\pt$bb1212du-p1$a$en.jpg"
    