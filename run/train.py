"""
Image2Seq Model with Attention Pipeline
Train-Validation Script
"""
#!/usr/bin/env python3
#############################################################################
# Imports                                                                   #
#############################################################################

# Standard imports ##########################################################
import absl.logging
import argparse
import csv
import cv2
import logging
import numpy as np
import pandas as pd
from PIL import Image
import os 
from six import raise_from
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys 
from tqdm import tqdm

# Stop pycache
sys.dont_write_bytecode = True

# GPU setup #################################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Keras and tensorflow imports ##############################################
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow.keras as keras
import tensorflow.keras.backend as K

# Local imports #############################################################
# Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import image2seq
__package__ = "image2seq"

from image2seq.preprocessing.token_preprocessing \
  import token_preprocessing, matrix_shape_preprocessing
from image2seq.models.eda_xu import EDAXU
from image2seq.models.eda_xu_mlp import EDAXUMLP 
from image2seq.models.eda_xu_mlp_exp_loss import EDAXUMLPEXPLOSS
from image2seq.models.drake_concat import DRAKECONCAT

# Logging options ###########################################################
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr=False
date = pd.datetime.now().date()
hour = pd.datetime.now().hour
minute = pd.datetime.now().minute
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
    filename="image2seq/logs/train_log_{}_{}{}.txt"
              .format(date, hour, minute))
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(
  '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

#############################################################################
# Model Setup                                                               #
#############################################################################
logging.info("MODEL SETUP - Training Script - train.py")
from tensorflow.python.client import device_lib
logging.info("MODEL SETUP - CUDA VISIBLE DEVICES {}"
             .format(device_lib.list_local_devices()))
tf.compat.v1.debugging.assert_equal(True, tf.test.is_gpu_available())
tf.compat.v1.debugging.assert_equal(True, tf.test.is_built_with_cuda())

image2seq = EDAXUMLP(image_embedding_dim=256,
                     token_embedding_dim=8,
                     decoder_hidden_dim=512,
                     mlp_hidden_dim=256)
logging.info("MODEL SETUP - image2seq model {} instantiated"
             .format(image2seq.get_model_name()))
logging.info("MODEL SETUP - log file = image2seq/logs/train_log_{}_{}{}.txt"
              .format(date, hour, minute))

# Parameter options #########################################################
# CSV file of images to import
# images_seqs_csv = "/test_data/stage2_data_train/stage2_train.txt"
images_seqs_csv = "/stage2_data_train_1x1/stage2_train.txt"
train_info_csv = "/stage2_data_train_1x1/stage2_train_info.txt"

# Data config
batch_size = 256
logging.info("MODEL SETUP - Batch size {}".format(batch_size))

# Optimizer selection
optimizer = tf.compat.v1.train.AdamOptimizer()

# Training loop
num_epochs = 10
logging.info("MODEL SETUP - Number of epochs {}".format(num_epochs))

# Checkpointing #############################################################
checkpoint_directory = \
  "./image2seq/checkpoints/train/{}_{}_{}{}"\
  .format(image2seq.get_model_name(), date, hour, minute)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=image2seq)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                checkpoint_directory,
                                                max_to_keep=5)
# Don't load latest checkpoint because usually want to start from scratch
# status = checkpoint.restore(manager.latest_checkpoint)

# Output file ###############################################################
results_file = "image2seq/checkpoints/train/{}_{}_{}{}/results.txt"\
                .format(image2seq.get_model_name(), date, hour, minute)
predictions_file = "image2seq/checkpoints/train/{}_{}_{}{}/predictions.txt"\
                .format(image2seq.get_model_name(), date, hour, minute)
image2seq.set_predictions_file(predictions_file)

#############################################################################
# Pre-processing                                                            #
#############################################################################
# STEP 1: Pre-process token #################################################
list_image_paths, list_processed_matrix_seqs = \
  token_preprocessing(images_seqs_csv, 
                      batch_size=batch_size, 
                      skip_padding=True)

len_list_image_paths = len(list_image_paths)

_, list_matrix_shapes = \
  matrix_shape_preprocessing(train_info_csv,
                             batch_size=batch_size)

logging.info("PREPROCESSING - Step 1 - Token preprocessing")
tf.compat.v1.debugging.assert_equal(len(_), len_list_image_paths)
tf.compat.v1.debugging.assert_equal(len(list_matrix_shapes), 
                                    len(list_processed_matrix_seqs))
tf.compat.v1.debugging.assert_equal(len(_), len(list_matrix_shapes))

# STEP 2: Train-validation split ############################################
shuffled_image_paths, shuffled_matrix_seqs, shuffled_matrix_shapes = \
  shuffle(list_image_paths, 
          list_processed_matrix_seqs, 
          list_matrix_shapes, 
          random_state=1)

img_name_train, img_name_val, seq_train, seq_val, matrix_shapes_train, \
  matrix_shapes_val = train_test_split(shuffled_image_paths, 
                                       shuffled_matrix_seqs, 
                                       shuffled_matrix_shapes,
                                       test_size=0.1,
                                       random_state=0)

logging.info("PREPROCESSING - Step 2 - Train test split -"
             "Train examples {} Validation examples {}"
             .format(len(img_name_train), len(img_name_val)))
tf.compat.v1.debugging.assert_equal(len(img_name_train), len(seq_train))
tf.compat.v1.debugging.assert_equal(len(img_name_val), len(seq_val))
tf.compat.v1.debugging.assert_equal(len(matrix_shapes_train), len(seq_train))
tf.compat.v1.debugging.assert_equal(len(matrix_shapes_val), len(seq_val))

# STEP 3: Sort images and tokens by length ##################################
sorted_seq_train, sorted_img_name_train, sorted_matrix_shapes_train = \
  sorted((seq_train, img_name_train, matrix_shapes_train), key=len)

sorted_seq_val, sorted_img_name_val, sorted_matrix_shapes_val = \
  sorted((seq_val, img_name_val, matrix_shapes_val), key=len)

# STEP 4: Create tf dataset from generator ##################################

def train_generator():
  for x, y, z in zip(sorted_img_name_train, 
                     sorted_seq_train, 
                     sorted_matrix_shapes_train):
    yield (x, y, z)

def val_generator():
  for x, y, z in zip(sorted_img_name_val, 
                     sorted_seq_val,
                     sorted_matrix_shapes_val):
    yield (x, y, z)

train_dataset = \
  tf.data.Dataset.from_generator(
    generator=train_generator,
    output_types=(tf.string, tf.int32, tf.int32))

validation_dataset = \
  tf.data.Dataset.from_generator(
    generator=val_generator,
    output_types=(tf.string, tf.int32, tf.int32))

# STEP 5: Load numpy feature maps ###########################################
def load_numpy(img_name, seq, matrix_shapes):
  decoded_img_name = img_name.decode('utf-8')
  replaced_img_name = decoded_img_name.replace(".png", "")
  img_tensor = np.load(replaced_img_name + '.npy')
  return img_tensor, seq, matrix_shapes

train_dataset = train_dataset.map(
  lambda item1, item2, item3:
  tf.numpy_function(load_numpy, 
                    [item1, item2, item3],
                    [tf.float32, tf.int32, tf.int32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset.map(
  lambda item1, item2, item3:
  tf.numpy_function(load_numpy, 
                    [item1, item2, item3],
                    [tf.float32, tf.int32, tf.int32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# STEP 6: Pad by batch ######################################################
train_dataset = train_dataset.padded_batch(
  batch_size,
  padded_shapes=([None, None], [None], [None]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset.padded_batch(
  batch_size,
  padded_shapes=([None, None], [None], [None]))
validation_dataset = \
  validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#############################################################################
# Train-validation loop                                                     #
#############################################################################
list_train_epoch_losses = []
list_val_epoch_losses = []
list_val_edit_distance = []

for epoch in range(num_epochs):
  ###########################################################################
  # Train loop                                                              #
  ###########################################################################
  train_epoch_loss = 0
  train_num_batches = 0
  pbar = tqdm(total=len(seq_train))
  one_percent_progress = len(seq_train) / 100
  num_images_processed = 0
  logging.info("TRAINING   - Epoch {} Model {}"
               .format(epoch, image2seq.get_model_name()))

  for (train_batch, (train_img, train_target, train_detections)) \
    in enumerate(train_dataset):
    # Train Batch ###########################################################
    train_batch = train_batch + 1
    with tf.GradientTape() as tape:
      train_batch_loss, _ = image2seq([train_img, 
                                       train_target, 
                                       train_detections],
                                      dropout=True)

    # Calculate gradients ###################################################
    gradients = tape.gradient(train_batch_loss, 
                              image2seq.trainable_variables)
                          
    # Apply gradients #######################################################
    optimizer.apply_gradients(
      grads_and_vars=zip(gradients, image2seq.trainable_variables))

    # Update epoch statistics ###############################################
    train_epoch_loss += train_batch_loss
    train_num_batches = train_num_batches + 1

    if train_batch * batch_size - num_images_processed > \
      one_percent_progress:
      rolling_mean = float(train_epoch_loss) / float(train_num_batches)
      pbar.set_description("TRAINING   - Epoch {} Batch {} Rolling mean "
                           "batch loss {}"
                           .format(epoch, train_batch, rolling_mean))
      pbar.update(train_batch * batch_size - num_images_processed)
      num_images_processed = train_batch * batch_size

  # End of epoch train statistics ###########################################
  mean_train_epoch_loss = float(train_epoch_loss) / float(train_num_batches)
  list_train_epoch_losses.append(mean_train_epoch_loss)
  pbar.close()

  logging.info("TRAINING   - Epoch {}: Epoch mean losses = {}"
    .format(epoch, mean_train_epoch_loss))
  
  ###########################################################################
  # Validation loop                                                         #
  ###########################################################################
  val_epoch_loss = 0
  val_num_batches = 0
  val_epoch_edit_distance = 0
   
  for (val_batch, (val_img, val_target, val_detections)) \
    in enumerate(validation_dataset):
    # Validate batch ########################################################
    val_batch_loss, val_batch_edit_distance = image2seq([val_img, 
                                                         val_target,
                                                         val_detections], 
                                                        val_mode=True)
    logging.debug("VALIDATION - Epoch {} Batch {} Batch loss {}"
                  .format(epoch, val_batch, val_batch_loss))

    # Update epoch statistics ###############################################
    val_epoch_loss += val_batch_loss
    val_epoch_edit_distance += val_batch_edit_distance
    val_num_batches = val_num_batches + 1
  
  # End of epoch validation statistics ######################################
  mean_val_epoch_loss = float(val_epoch_loss) / float(val_num_batches)
  mean_val_edit_distance = float(val_batch_edit_distance) / \
    float(val_num_batches)
  list_val_epoch_losses.append(mean_val_epoch_loss)
  list_val_edit_distance.append(mean_val_edit_distance)

  logging.info("VALIDATION - Epoch {}: Epoch mean losses = {}"
    .format(epoch, mean_val_epoch_loss))
  logging.info("VALIDATION - Epoch {}: Epoch mean edit distance = {}"
    .format(epoch, mean_val_edit_distance))

  # Save checkpoint #########################################################
  if mean_val_epoch_loss == min(list_val_epoch_losses):
    logging.info("VALIDATION - Save checkpoint because {} loss is minimum"
                 .format(mean_val_epoch_loss))
    checkpoint_manager.save()
  else:
    logging.info("VALIDATION - Do not save checkpoint because {} loss is "
      "greater than minimum loss of {}"
      .format(mean_val_epoch_loss, min(list_val_epoch_losses)))

  ###########################################################################
  # Epoch results                                                           #
  ###########################################################################
  with open(results_file, "a+") as rf:
    rf.write("{},{},{},{}\n"\
      .format(epoch, 
              mean_train_epoch_loss, 
              mean_val_epoch_loss,
              mean_val_edit_distance))
      
# Training results ##########################################################
logging.info("TRAINING   - Finished - Losses \n{}"
  .format(list_train_epoch_losses))
logging.info("VALIDATION - Finished - Losses \n{}"
  .format(list_val_epoch_losses))
logging.info("VALIDATION - Finished - Edit Distances \n{}"
  .format(list_val_edit_distance))