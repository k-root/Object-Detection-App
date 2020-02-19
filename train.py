# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v1 as tf

import model_hparams
import model_lib

def training(model_dir, pipeline_config_path, num_train_steps, eval_training_data, checkpoint_dir, sample_1_of_n_eval_examples=1, sample_1_of_n_eval_on_train_examples=5, hparams_overrides=None, run_once=False):
#   model_dir
#   pipeline_config_path
#   num_train_steps
#   eval_training_data
#   ######'''should be optional'''######
#   sample_1_of_n_eval_examples  
#   sample_1_of_n_eval_on_train_examples
#   hparams_overrides
#   checkpoint_dir
#   run_once
  config = tf.estimator.RunConfig(model_dir=model_dir)

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(hparams_overrides),
      pipeline_config_path=pipeline_config_path,
      train_steps=num_train_steps,
      sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          sample_1_of_n_eval_on_train_examples))
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']

  if checkpoint_dir:
    if eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if run_once:
      estimator.evaluate(input_fn,
                         steps=None,
                         checkpoint_path=tf.train.latest_checkpoint(
                             checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, checkpoint_dir, input_fn,
                                train_steps, name)
  else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
    
