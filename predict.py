# detect kangaroos in photos with mask rcnn model
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


# class that defines and loads the kangaroo dataset
class Dataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, classes, is_train=True):
		# define classes####Change to dynamic   ####classes=['class1', 'class2', 'class3', .....]
		class_info=[]
		for i in range(len(classes)):
			class_info.append(self.add_class("dataset", i+1, classes[i]))
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

		# find all images
		# for filename in listdir(images_dir):
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

	# load all bounding boxes for an image
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes = list()
		# extract each bounding box
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
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			item = boxes[i]##[name, [coor]]
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

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=2):
	# load image and mask
	writeToCsvList = []
	for i in range(n_images):
		# load the image and mask
		print("=========i in load mask is: "+ str(i))
		image = dataset.load_image(i)
		imageName = dataset.source_image_link(i)
		mask, _ = dataset.load_mask(i)
		print(imageName)
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
			predictedLabelName = dataset.class_names[predictedLabelID]
			# draw the box
			# ax.add_patch(rect)
			# print((x1,y1),(x2,y2))
			detectedLabel = image[y1:y1+height, x1:x1+width]
			cv2.imwrite(str(i)+"_"+str(loopNum)+"_"+str(predictedLabelName)+".jpg",detectedLabel)
			loopNum+=1
			# writeToCsvList.append([i,])
	# show the figure
	# pyplot.show()

def predict_main(datasetDir, modelName, testFile, classes):
    # load the train dataset
    train_set = Dataset()
    train_set.load_dataset(datasetDir, classes, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # load the test dataset
    test_set = Dataset()
    test_set.load_dataset(datasetDir, classes, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
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
    plot_actual_vs_predicted(test_set, model, cfg)