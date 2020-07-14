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
import logging
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os 
from six import raise_from
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys 
from tqdm import tqdm

# Stop pycache
sys.dont_write_bytecode = True

# GPU setup #################################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
  import token_preprocessing, matrix_shape_preprocessing, \
    detected_values_preprocessing
from image2seq.models.eda_xu import EDAXU
from image2seq.models.eda_xu_mlp import EDAXUMLP 
from image2seq.models.eda_xu_mlp_exp_loss import EDAXUMLPEXPLOSS
from image2seq.models.drake_concat import DRAKECONCAT
from image2seq.models.drake_concat_mlp import DRAKECONCATMLP
from image2seq.models.drake_detections2 import DRAKEDETECTIONS2
from image2seq.models.drake_nested_single_lstm import DRAKENESTEDSINGLELSTM
from image2seq.models.drake_parallel import DRAKEPARALLEL

# Logging options ##########################################################'
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
logging.info("MODEL SETUP - Tensorflow version".format(tf.__version__))
logging.info("MODEL SETUP - Validation Script - train_full.py")
from tensorflow.python.client import device_lib
logging.info("MODEL SETUP - CUDA VISIBLE DEVICES {}"
             .format(device_lib.list_local_devices()))
# tf.compat.v1.debugging.assert_equal(True, tf.test.is_gpu_available())
# tf.compat.v1.debugging.assert_equal(True, tf.test.is_built_with_cuda())

image2seq = EDAXUMLP()
logging.info("MODEL SETUP - image2seq model {} instantiated"
             .format(image2seq.get_model_name()))
logging.info("MODEL SETUP - log file = "
             "image2seq/logs/evaluate_log_{}_{}{}.txt"
              .format(date, hour, minute))

# Parameter options #########################################################
# CSV file of images to import
# images_seqs_csv = "/graphs/0816_retinanet_1x1_demo/cropped/stage2_train.txt"
# train_info_csv = "/graphs/0816_retinanet_1x1_demo/cropped/stage2_train.txt"
images_seqs_csv = "/graphs/0828_retinanet_1x1_demo/cropped/stage2_train.txt"
train_info_csv = "/graphs/0828_retinanet_1x1_demo/cropped/stage2_train.txt"

# Data config
batch_size = 22
logging.info("MODEL SETUP - Batch size {}".format(batch_size))

# Optimizer selection
optimizer = tf.compat.v1.train.AdadeltaOptimizer()

# Checkpointing #############################################################
checkpoint_directory = "./image2seq/checkpoints/train/eda_xu_mlp_2019-08-24_1655"
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=image2seq)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                checkpoint_directory,
                                                max_to_keep=5)
status = checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Output file ###############################################################
results_file = checkpoint_directory +  "/results_evaluate.txt"
predictions_file = checkpoint_directory + "/predictions_evaluate.txt"
image2seq.set_predictions_file(predictions_file)
attention_file = checkpoint_directory

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
  detected_values_preprocessing(train_info_csv,
                                batch_size=batch_size)

logging.info("PREPROCESSING - Step 1 - Token preprocessing")
tf.compat.v1.debugging.assert_equal(len(_), len_list_image_paths)
tf.compat.v1.debugging.assert_equal(len(list_matrix_shapes), 
                                    len(list_processed_matrix_seqs))
tf.compat.v1.debugging.assert_equal(len(_), len(list_matrix_shapes))


# STEP 2: Train-validation split ############################################
# shuffled_image_paths, shuffled_matrix_seqs, shuffled_matrix_shapes = \
#   shuffle(list_image_paths, 
#           list_processed_matrix_seqs, 
#           list_matrix_shapes, 
#           random_state=1)


# img_name_train, img_name_val, seq_train, seq_val, matrix_shapes_train, \
#   matrix_shapes_val = train_test_split(shuffled_image_paths, 
#                                        shuffled_matrix_seqs, 
#                                        shuffled_matrix_shapes,
#                                        test_size=len_list_image_paths,
#                                        random_state=0)

# logging.info("PREPROCESSING - Step 2 - Train test split -"
#              "Train examples {} Validation examples {}"
#              .format(len(img_name_train), len(img_name_val)))
# tf.compat.v1.debugging.assert_equal(len(img_name_train), len(seq_train))
# tf.compat.v1.debugging.assert_equal(len(img_name_val), len(seq_val))
# tf.compat.v1.debugging.assert_equal(len(matrix_shapes_train), len(seq_train))
# tf.compat.v1.debugging.assert_equal(len(matrix_shapes_val), len(seq_val))

# STEP 3: Sort images and tokens by length ##################################
# sorted_seq_val, sorted_img_name_val, sorted_matrix_shapes_val = \
#   sorted((shuffled_matrix_seqs, 
#           shuffled_image_paths, 
#           shuffled_matrix_shapes), key=len)

# STEP 4: Create tf dataset from generator ##################################
def eval_generator():
  for x, y, z in zip(list_image_paths, 
                     list_processed_matrix_seqs,
                     list_matrix_shapes):
    yield (x, y, z)

validation_dataset = \
  tf.data.Dataset.from_generator(
    generator=eval_generator,
    output_types=(tf.string, tf.int32, tf.int32))

# STEP 5: Load images #######################################################
def load_image(image_path, seq, matrix_shapes):
  """
  Load image from image_path resizing it to match inputs required for 
  InceptionV3 - notably width and height of 299 pixels
  """
  img = tf.io.read_file(image_path)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.resize(img, (299, 299))
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, seq, matrix_shapes

validation_dataset = validation_dataset.map(
  lambda item1, item2, item3:
  tf.numpy_function(load_image, 
                    [item1, item2, item3],
                    [tf.float32, tf.int32, tf.int32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

logging.info("PREPROCESSING - Step 3 - Images processed")

# STEP 6: Pad by batch ######################################################
validation_dataset = validation_dataset.padded_batch(
  batch_size,
  padded_shapes=([None, None, None], [None], [None]))
validation_dataset = \
  validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#############################################################################
# Validation loop                                                           #
#############################################################################
list_eval_epoch_losses = []
list_eval_edit_distance = []
eval_epoch_loss = 0
eval_num_batches = 0
eval_epoch_edit_distance = 0

# Plot attention function ###################################################
def plot_attention(batch_num, batch_images, batch_predictions, 
  batch_attention_weights):
  # STEP 0: Process Inputs ##################################################
  # Input               | Encoder input       | batch_size=None x           #
  #                     | (inceptionv3)       | img_width=299 x             #
  #                     |                     | img_height=299 x            #
  #                     |                     | num_colours=3               #  
  #                     | Predictions input   | batch_size = None x         #
  #                     |                     | token_seq_len =  x          #
  #                     |                     | token_seq_len               #
  #                     | Attention weights   | batch_size=None x           #
  #                     |                     | token_seq_len =             #
  #                     |                     | token_seq_len x             #
  #                     |                     | num_features=64 x           #
  #                     |                     | score_dim=1                 #
  #_____________________|_____________________|_____________________________#
  # Note batch_token_seq_len is 1 less than token_seq_len in each model.
  batch_size = batch_predictions.shape[0]
  batch_token_seq_len = batch_predictions.shape[1]

  # Logging, Debug & Assert
  logging.debug("EVALUATION - Plot Attention - Step 0 - Process Inputs - "
                "batch_size {}".format(batch_size))
  logging.debug("EVALUATION - Plot Attention - Step 0 - Process Inputs - "
                "batch_images_shape {}"
                .format(K.int_shape(batch_images)))
  logging.debug("EVALUATION - Plot Attention - Step 0 - Process Inputs - "
                "batch_predictions_shape {}"
                .format(K.int_shape(batch_predictions)))
  logging.debug("EVALUATION - Plot Attention - Step 0 - Process Inputs - "
                "batch_attention_weights_shape {}"
                .format(K.int_shape(batch_attention_weights)))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(batch_images),
    (batch_size, 299, 299, 3))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(batch_predictions),
    (batch_size, batch_token_seq_len))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(batch_attention_weights),
    (batch_size, batch_token_seq_len, 64, 1))

  # STEP 1: Reshape Attention Weights #######################################
  # Reshape             | Attention weights   | batch_size=None x           #
  #                     |                     | token_seq_len =             #
  #                     |                     | token_seq_len x             #
  #                     |                     | num_features=64             #
  #_____________________|_____________________|_____________________________#
  batch_attention_plot = tf.reshape(batch_attention_weights, (-1, ))

  # Logging, Debug & Assert
  logging.debug("EVALUATION - Plot Attention - Step 1 - Reshape Attention "
                "Weights - batch_attention_plot shape {}"
                .format(K.int_shape(batch_attention_plot)))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(batch_attention_plot),
    (batch_size, batch_token_seq_len, 64))

  # STEP 2: Convert to numpy ################################################
  batch_attention_plot = batch_attention_plot.numpy()
  batch_images = batch_images.numpy()
  batch_predictions = batch_predictions.numpy()
  
  # STEP 3: Plot Attention ##################################################
  for batch in range(batch_size):
    batch_fig = plt.figure(figsize=(10,10))
    batch_image = batch_images[batch,:,:,:]
    for i in range(batch_token_seq_len):
      batch_att = np.resize(batch_attention_plot[batch, :, :], (8, 8))
      ax = batch_fig.add_subplot(i//2, i//2, i + 1)
      ax.set_title(batch_predictions[batch, i])
      batch_att_image = ax.imshow(batch_image)
      ax.imshow(batch_att, cmap='gray', alpha=0.6, 
                extent=batch_att_image.get_extent())
    plt.tight_layout()
    plt.savefig(attention_file + batch_num + '_' + batch + '.png')

for (eval_batch, (eval_img, eval_target, eval_detections)) \
  in enumerate(validation_dataset):
  # Validate batch ##########################################################
  eval_batch_loss, eval_batch_edit_distance = image2seq([eval_img, 
                                                        eval_target,
                                                        eval_detections], 
                                                        val_mode=True)
  logging.debug("EVALUATION - Batch {} Batch loss {}"
                .format(eval_batch, eval_batch_loss))

  # Batch attention data ####################################################
  batch_attention_weights = image2seq.get_attention_weights()
  batch_attention_predictions = image2seq.get_attention_predictions()

  # Save attention plots ####################################################
  # plot_attention(batch_num=eval_batch, 
  #                batch_images=eval_img, 
  #                batch_predictions=batch_attention_predictions,
  #                batch_attention_weights=batch_attention_weights)

  # Update epoch statistics #################################################
  eval_epoch_loss += eval_batch_loss
  eval_epoch_edit_distance += eval_batch_edit_distance
  eval_num_batches = eval_num_batches + 1

# End of epoch validation statistics ######################################
mean_eval_epoch_loss = float(eval_epoch_loss) / float(eval_num_batches)
mean_eval_edit_distance = float(eval_batch_edit_distance) / \
  float(eval_num_batches)
list_eval_epoch_losses.append(mean_eval_epoch_loss)
list_eval_edit_distance.append(mean_eval_edit_distance)

logging.info("EVALUATION - Mean losses = {}"
  .format(mean_eval_epoch_loss))
logging.info("EVALUATION - Epoch mean edit distance = {}"
  .format(mean_eval_edit_distance))

#############################################################################
# Epoch results                                                             #
#############################################################################
with open(results_file, "a+") as rf:
  rf.write("{},{}\n"\
    .format(mean_eval_epoch_loss,
            mean_eval_edit_distance))

# Training results ##########################################################
logging.info("EVALUATION - Finished - Losses \n{}"
  .format(list_eval_epoch_losses))
logging.info("EVALUATION - Finished - Edit Distances \n{}"
  .format(list_eval_edit_distance))