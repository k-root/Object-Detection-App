import re
import os
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import shutil
import argparse
import SMLExportInferenceGraph
from os.path import dirname

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import ops as utils_op
from Cython.Compiler import Pipeline
import cv2

sys.path.append("..")

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

##### Object Detection Imports #####

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

######################################
'''Make inference graph'''


######################################

def executeInferenceGraph(pipeline_config_path):
    files = [f for f in os.listdir(pipeline_config_path + '/savedModelData') if re.match(r'[a-zA-Z]+.*\.ckpt', f)]
    ckpt_number = re.findall(r'\d+', files[len(files) - 1])
    ckpt_path = pipeline_config_path + '\savedModelData\model.ckpt-' + ckpt_number[0]
    trained_checkpoint = os.path.join(os.getcwd(), ckpt_path)
    if not os.path.exists(pipeline_config_path + '/object_detection_graph'):
        output_directory = pipeline_config_path + '/object_detection_graph'
        SMLExportInferenceGraph.main(pipeline_config_path + '/ssd_mobilenet_v1_coco.config', trained_checkpoint,
                                     output_directory)
    else:
        shutil.rmtree(pipeline_config_path + '/object_detection_graph')
        output_directory = pipeline_config_path + '/object_detection_graph'
        SMLExportInferenceGraph.main(pipeline_config_path + '/ssd_mobilenet_v1_coco.config', trained_checkpoint,
                                     output_directory)


######################################
'''End inference Graph'''
######################################

######################################
'''Start Testing'''
######################################
PATH_TO_CKPT = ''
PATH_TO_LABELS = ''
categories = ''
category_index = ''


def define_paths(pipeline_config_path):
    global PATH_TO_CKPT
    PATH_TO_CKPT = pipeline_config_path + "/object_detection_graph/frozen_inference_graph.pb"
    global PATH_TO_LABELS
    PATH_TO_LABELS = pipeline_config_path + "/pet_label_map.pbtxt"
    NUM_CLASSES = 90

    IMAGE_SIZE = (12, 8)  # Image size

    ## loading label map ##
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    global categories
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    global category_index
    category_index = label_map_util.create_category_index(categories)


def load_model_and_return_detection_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        with tf.Session(config=config) as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
    # # Detection


def displayVideosWithDetection(path):
    cap = cv2.VideoCapture(path)
    frame_width=int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('static/test_output.avi', fourcc, 20.0, (frame_width, frame_height))
    #out = cv2.VideoWriter("static/test_output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 10,(frame_width, frame_height))
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            ima = displayImageWithDetection(frame)
            out.write(ima)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def displayImageWithDetection(image):
    # print(PATH_TO_CKPT)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np =image
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, load_model_and_return_detection_graph())
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


###########################################
'''End Testing'''


###########################################

def detect(video_location, model_name):
    video_name = video_location
    pipeline_config_path = dirname(os.getcwd()) + '/models/' + model_name
    pipeline_config_path = pipeline_config_path[:2] + '//' + pipeline_config_path[3:]
    executeInferenceGraph(pipeline_config_path)
    define_paths(pipeline_config_path)
    displayVideosWithDetection(video_name)

############################################
############################################


