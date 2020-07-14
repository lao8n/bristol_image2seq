"""
Image2Seq Model with Attention Pipeline
Image pre-processing 

MODEL  : NA
INPUT  : dataset_size x image_width x image_height
OUTPUT : dataset_size x 64 x 2048
"""
#!/usr/bin/env python3
#############################################################################
# Imports                                                                   #
#############################################################################

# Standard imports ##########################################################
import csv
import numpy as np
import os 
from six import raise_from
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys 
from tqdm import tqdm

# Stop pycache
sys.dont_write_bytecode = True

# Keras and tensorflow imports ##############################################
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow.keras as keras

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, 
    os.path.join(os.path.dirname(__file__), '..', '..'))
  import image2seq
  __package__ = "image2seq"

#############################################################################
# Preprocessing helper functions                                            #
#############################################################################
def open_for_csv(path):
  """ 
  Open a file with flags suitable for csv.reader.

  This is different for python2 it means with mode 'rb', for python3 this 
  means 'r' with "universal newlines".
  """
  if sys.version_info[0] < 3:
      return open(os.getcwd() + path, 'rb')
  else:
      return open(os.getcwd() + path, 'r', newline='')

def load_image(image_path):
  """
  Load image from image_path resizing it to match inputs required for 
  InceptionV3 - notably width and height of 299 pixels
  """
  img = tf.io.read_file(image_path)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.resize(img, (299, 299))
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

#############################################################################
# Image preprocessing                                                       #
#############################################################################

# STEP 0: Source CSV file ###################################################
# csv_data_file = "/test_data/stage2_data_train/stage2_train.txt"
csv_data_file = "/stage2_data_train/stage2_train.txt"

# STEP 1: Parse CSV data ####################################################
# Extract a list of image paths and matrix sequences
list_image_paths = []
list_matrix_seqs = []
with open_for_csv(csv_data_file) as file:
  for line, row in enumerate(csv.reader(file, delimiter='_')):
    line += 1
    try:
      image_path, matrix_seq = row[:2]
    except ValueError:
      raise_from(ValueError('line {}: format should be \'image_path '
                            'matrix_seq\' '.format(line)), None)

    list_image_paths.append(image_path)

# STEP 2 - Process images ###################################################
image_model = keras.applications.InceptionV3(include_top=False,
                                             weights='imagenet')

image_model_preprocessing = \
  keras.Model(inputs=image_model.input,
              outputs=image_model.layers[-1].output)

image_dataset = tf.data.Dataset.from_tensor_slices(list_image_paths)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

num_features = \
  (keras.backend.int_shape(image_model_preprocessing.output))[3]

pbar = tqdm(total=len(list_matrix_seqs))
one_percent_progress = len(list_matrix_seqs) / 100
num = 0
num_processed = 0
for img, path in image_dataset:
  batch_features = image_model_preprocessing(img)
  batch_features = tf.reshape(batch_features, 
                              (batch_features.shape[0],
                               -1, 
                               num_features))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    path_of_feature = path_of_feature.replace(".png", "")
    np.save(path_of_feature, bf.numpy())

  num += 1
  if num - num_processed > one_percent_progress:
    pbar.update(num - num_processed)
    num_processed = num
