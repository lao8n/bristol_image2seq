"""
Image2Seq Model with Attention Pipeline
Image2Seq Model
Mean Masked Cross Entropy Loss
"""

#############################################################################
# Imports                                                                   #
#############################################################################

# Standard imports ##########################################################
import logging
import os
import sys
import numpy as np

# Stop pycache #
sys.dont_write_bytecode = True

# Keras and tensorflow imports ##############################################
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow.keras as keras
import tensorflow.keras.backend as K

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  import keras_image2seq
  __package__ = "image2seq"

from image2seq.metrics.edit_distance import edit_distance_metric

prediction = np.array([[1, 3, 2, 2, 3, 1], [4, 1, 2, 1, 0, 0]])
target = np.array([[1, 3, 4, 2, 0, 0], [3, 2, 0, 0, 0, 0]])

prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)
target = tf.convert_to_tensor(target, dtype=tf.float32)

prediction_file ="image2seq/metrics/test_file.txt"

edit_distance_metric(target=target, 
                     prediction=prediction, 
                     predictions_file=prediction_file)