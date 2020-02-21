# fit a mask rcnn on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import math

# class that defines and loads the kangaroo dataset
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

		print(f"Total dataset Length:{total_dataset_length}, train length{len_train_data}, test length{len_test_data}")

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
		print("-----------------------annotation path-------------------" + path)
		boxes, w, h = self.extract_boxes(path)
		print("-----------------------boxes-------------------", boxes)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			item = boxes[i]##[name, [coor]]
			print("-----------------------individual box-------------------", item)
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

# define a configuration for the model
class KangarooConfig(Config):
	# define the name of the configuration
	NAME = "ggb_cfg"####Change to dynamic
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 6####Change to dynamic
	# number of training steps per epoch
	STEPS_PER_EPOCH = 160####Change to dynamic

# prepare train set

def run_train(datasetDir, modelWeight, classes):
	train_set = Dataset()
	train_set.load_dataset(datasetDir, classes,  is_train=True)####Change to dynamic
	train_set.prepare()
	print('Train: %d' % len(train_set.image_ids))
	# prepare test/val set
	test_set = Dataset()
	test_set.load_dataset(datasetDir, classes, is_train=False)
	test_set.prepare()
	print('Test: %d' % len(test_set.image_ids))
	# prepare config
	config = KangarooConfig()
	config.display()
	# define the model


	###
	'''Change model_dir to users/userName/datasetName/model'''
	###
	model = MaskRCNN(mode='training', model_dir='./users/user1/dataset/models/', config=config)####Change to dynamic


	# load weights (mscoco) and exclude the output layers
	model.load_weights(modelWeight, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
	# train weights (output layers or 'heads')
	model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
