

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import functools
import json
import os
import trainer
import glob
import io
import argparse
import random
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf

from os.path import dirname

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from pathlib import Path

from PIL import Image
#from utils import dataset_util
from collections import namedtuple, OrderedDict
from asyncio.tasks import sleep
from shutil import copyfile


###############################################
'''Convert XMl to CSV '''
###############################################

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

########################################################
'''Split Dataset into two parts '''
########################################################

def split_dataset(input, train, valid):
    random.seed(123)
    file = os.path.join(os.getcwd(), input)
    out1 = os.path.join(os.getcwd(), train)
    out2 = os.path.join(os.getcwd(), valid)
    with open(file, 'r', encoding="utf-8") as f, open(out1, 'w') as o1, open(out2, 'w') as o2:
        lines = sum(1 for line in f)
        f.seek(0)
        
        train = int(lines*0.80)
        valid = lines - train
        
        i = 0
        for line in f:
            if i > 0:
                r = random.random() 
                if (i < train and r < 0.8) or (lines - i > valid):
                    o1.write(line)
                    i+=1
                else:
                    o2.write(line)
            else:
                o1.write(line)
                o2.write(line)
                i+=1 

####################################################
'''Begin Generating TfRecords'''
####################################################

def class_text_to_int(row_label, mod_dir):
    pet_label_map = "item {\n id: 1\n name :"+" '"+row_label+"'\n }"
    with open(mod_dir +'/pet_label_map.pbtxt', 'w') as txt:
        txt.write(pet_label_map)
    return 1


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, models_dir):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], models_dir))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

                   
def generateTFrecords(input, output, dir , models_dir):
    
    writer = tf.python_io.TFRecordWriter(output)
    images_path = os.path.join(os.getcwd(), dir)
    examples = pd.read_csv(input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_path, models_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
    output_path = os.path.join(os.getcwd(), output)
    print('Successfully created the TFRecords: {}'.format(output_path))


#################################################################
''' End Generating TfRecords'''
#################################################################

#################################################################
'''Start Training function'''
#################################################################

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', 'savedModelData/',
                    'Directory to save the checkpoints and training summaries.')

# flags.DEFINE_string('pipeline_config_path', 'ssd_mobilenet_v1_coco.config',
#                     'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#                     'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
flags.DEFINE_string('dir', '',
                    'Path to a annotations')
flags.DEFINE_string('imdir', '',
                    'Path to a images')
flags.DEFINE_string('model', '',
                    'Unique model name.')

FLAGS = flags.FLAGS

def trainModel(pipeline_config_path, train_dir):
    assert train_dir, '`train_dir` is missing.'
    if FLAGS.task == 0: tf.gfile.MakeDirs(train_dir)
    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        if FLAGS.task == 0:
            tf.gfile.Copy(pipeline_config_path,
                    os.path.join(train_dir, 'pipeline.config'),
                    overwrite=True)
    else:
        configs = config_util.get_configs_from_multiple_files(
            model_config_path=FLAGS.model_config_path,
            train_config_path=FLAGS.train_config_path,
            train_input_config_path=FLAGS.input_config_path)
        if FLAGS.task == 0:
            for name, config in [('model.config', FLAGS.model_config_path),
                           ('train.config', FLAGS.train_config_path),
                           ('input.config', FLAGS.input_config_path)]:
            
                tf.gfile.Copy(config, os.path.join(train_dir, name),
                      overwrite=True)
    
    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)
    
    def get_next(config):
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()
    
    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                             job_name=task_info.type,
                             task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=True)

    trainer.train(
        create_input_dict_fn,
        model_fn,
        train_config,
        master,
        task,
        FLAGS.num_clones,
        worker_replicas,
        FLAGS.clone_on_cpu,
        ps_tasks,
        worker_job_name,
        is_chief,
        train_dir,
        graph_hook_fn=graph_rewriter_fn)

##########################################
'''End Training Functions'''
##########################################

         
def main(_):
    # parser = argparse.ArgumentParser(description='Give path to annotated file')
    # parser.add_argument("--dir", "-d", type=str, required = True, 
    #                help='Provide directory name as --dir or -d')
    # parser.add_argument("--imdir", "-img", type=str, required = True,
    #                     help = 'Provide image directory name as --imdir or -img')
    # parser.add_argument("--model", "-m", type=str, required = True,
    #                     help = 'Provide image directory name as --model or -m')
    
    # args = parser.parse_args()
    annotated_path = os.path.join(os.getcwd(), FLAGS.dir)
    xml_df = xml_to_csv(annotated_path)
    
    ############################
    '''Convert xml to csv '''
    ############################
    print(FLAGS.dir)
    print(FLAGS.imdir)
    model_name = FLAGS.model
    print(os.path.join(dirname(os.getcwd()),'models'))
    if not os.path.exists(os.path.join(dirname(os.getcwd()),'models')):
        os.makedirs(os.path.join(dirname(os.getcwd()),'models'))
    models_name_dir = dirname(os.getcwd()) + '/models/' + model_name
    mod_dir = models_name_dir[:2] +'//'+models_name_dir[3:].replace('\\','\\\\')          
    if not os.path.exists(models_name_dir):
        os.makedirs(models_name_dir)
    if os.path.exists(models_name_dir+'/trainingDataFiles'):
        xml_df.to_csv(models_name_dir+'/trainingDataFiles/annotated_label.csv', index=None)
    else :
        os.makedirs(models_name_dir+'/trainingDataFiles')
        xml_df.to_csv(models_name_dir+'/trainingDataFiles/annotated_label.csv', index=None)
    
    print('Successfully converted xml to csv.')
    
    print('Splitting datasets with 80% in training dataset and 20% in validation dataset and converting into TFRecords....')

    split_dataset(models_name_dir + '/trainingDataFiles/annotated_label.csv', models_name_dir + '/trainingDataFiles/annotated_train.csv', models_name_dir + '/trainingDatafiles/annotated_val.csv')
    print('Splitting completed.')
        
    ##############################
    '''Editing and copying CONFIG file to models_dir_name'''
    ##############################
    
#        
        
#   
    model_path =  dirname(os.getcwd()) +'\BaseModel\model.ckpt'
    base_model_path = model_path[:2] +'//'+model_path[3:].replace('\\','\\\\')
    f = open(dirname(os.getcwd())+'/config/ssd_mobilenet_v1_coco.config', 'r').read()
    f = f.replace('    input_path: "trainingDataFiles/train.record"\n', '    input_path: "'+mod_dir+'/trainingDataFiles/train.record"\n')
    f = f.replace('    input_path: "trainingDataFiles/val.record"\n', '    input_path: "'+mod_dir+'/trainingDataFiles/val.record"\n')
    f = f.replace('  label_map_path: "pet_labelmap.pbtxt"\n', '  label_map_path: "'+mod_dir+'/pet_label_map.pbtxt"\n')
    f = f.replace('  label_map_path: "pet_labelMap.pbtxt"\n', '  label_map_path: "'+mod_dir+'/pet_label_map.pbtxt"\n')
    f = f.replace('  fine_tune_checkpoint: "BaseModel\model.ckpt"\n','  fine_tune_checkpoint: "'+ base_model_path+'"\n')
    s = open('ssd_mobilenet_v2_coco.config','w')
    s.write(f)
    s.close()
#         print(type('    input_path: "trainingDataFiles/train.record"\n'))
#         s = '    input_path: "trainingDataFiles/train.record"\n'
#         s = s.encode('ascii', 'replace')
#         lines[lines.index('    input_path: "trainingDataFiles/train.record"\n')] = s
#         lines[lines.index('    input_path: "trainingDataFiles/val.record"\n')] ='    input_path:"'+models_name_dir+'\\trainingDataFiles\\val.record"\n'
#         lines[lines.index('  label_map_path: "trainingDataFiles/labelMap.pbtxt"\n')] = '  label_map_path:"'+models_name_dir+'\\pet_label_map.pbtxt"\n'
#         lines[lines.index('  label_map_path: "trainingDataFiles/label_Map.pbtxt"\n')] = '  label_map_path:"'+models_name_dir+'\\pet_label_map.pbtxt"\n'
#         f = open('ssd_mobilenet_v2_coco.config', 'w', encoding='utf-8')
#         f.write(b''.join(lines))
#         f.close()
    
    #copyfile('ssd_mobilenet_v2_coco.config', models_name_dir + '/ssd_mobilenet_v1_coco.config')
    if not os.path.exists(os.path.join(models_name_dir + '/ssd_mobilenet_v1_coco.config')):
        os.rename('ssd_mobilenet_v2_coco.config',models_name_dir + '/ssd_mobilenet_v1_coco.config')
    ##############################
    '''Generating TfRecords'''
    ##############################
    
    train_output = models_name_dir + '/trainingDataFiles/train.record'
    val_output = models_name_dir + '/trainingDataFiles/val.record'
    
    train_input = models_name_dir + '/trainingDataFiles/annotated_train.csv'
    val_input = models_name_dir + '/trainingDataFiles/annotated_val.csv'
    
    generateTFrecords(train_input, train_output, FLAGS.imdir, mod_dir)
    generateTFrecords(val_input, val_output, FLAGS.imdir, mod_dir)
    
    ################################
    ''' End TfRecords '''
    ################################
    
    ################################
    ''' Training starts here'''
    ################################
    pipeline_config_path = mod_dir + '\\ssd_mobilenet_v1_coco.config' 
    train_dir = models_name_dir + '\savedModelData'
    trainModel(pipeline_config_path, train_dir)
    

def runner(argsList):
    #l =  ['--dir','C:\\Users\\sundhar\\Documents\\Workspace\\raccoon_dataset\\raccoon_dataset\\annotations', '--imdir', 'C:\\Users\\sundhar\\Documents\\Workspace\\raccoon_dataset\\raccoon_dataset\\images','--model','raccoon']
    tf.app.run(main=main,argv = argsList[0:] )