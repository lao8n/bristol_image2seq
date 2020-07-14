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

from image2seq.preprocessing.token_preprocessing import caption2seq

def edit_distance_metric(target,
                         prediction,
                         predictions_file=None):

  # STEP 0: Process inputs ##################################################
  batch_size = target.shape[0]
  logging.debug("{}".format(target))
  logging.debug("{}".format(prediction))

  # Logging, Debug & Assert
  logging.debug("EDIT DISTANCE - Step 0 - Process inputs "
                "target shape {}"
                .format(K.int_shape(target)))
  logging.debug("EDIT DISTANCE - Step 0 - Process inputs "
                "prediction shape {}"
                .format(K.int_shape(prediction)))
  logging.debug("EDIT DISTANCE - Step 0 - Process inputs "
                "batch size = {}"
                .format(batch_size))
  # tf.compat.v1.debugging.assert_equal(
  #   K.int_shape(target),
  #   K.int_shape(prediction))

  # STEP 1: Remove post <end> of sequence predictions #######################
  prediction_numpy = prediction.numpy()
  for i in range(batch_size):
    end_of_seq_found = False
    for j in range(prediction_numpy.shape[1]):
      if end_of_seq_found:
        prediction_numpy[i, j] = 0
      if prediction_numpy[i, j] == 2:
        end_of_seq_found = True

  prediction = tf.convert_to_tensor(prediction_numpy)

  # STEP 2: Dense to sparse tensors #########################################
  prediction_indices = tf.where(tf.not_equal(prediction, 0))
  prediction_values = tf.gather_nd(prediction, prediction_indices)
  prediction_shape = tf.shape(prediction, out_type=tf.int64)

  sparse_prediction = tf.SparseTensor(prediction_indices,
                                      prediction_values,
                                      dense_shape=prediction_shape)

  target_indices = tf.where(tf.not_equal(target, 0))
  target_values = tf.gather_nd(target, target_indices)
  target_shape = tf.shape(target, out_type=tf.int64)

  sparse_target = tf.SparseTensor(target_indices,
                                  target_values,
                                  dense_shape=target_shape)

  # STEP 3: Calculate edit_distance #########################################
  edit_distances = tf.edit_distance(sparse_prediction,
                                    sparse_target)

  # Logging, Debug & Assert
  logging.debug("EDIT DISTANCE - Step 1 - Edit distance shape {}"
                .format(K.int_shape(edit_distances)))
  logging.debug("EDIT DISTANCE - Step 1 - Edit distance values \n{}"
                .format(edit_distances))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(edit_distances),
    (batch_size, ))

  # STEP 4: Calculate average batch edit distance ###########################
  mean_edit_distance = tf.reduce_mean(edit_distances)

  # Logging, Debug & Assert
  logging.debug("EDIT DISTANCE - Step 2 - Edit distance mean {}"
                .format(mean_edit_distance))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(mean_edit_distance),
    ())
  
  # STEP 5: Write predictions out for debugging #############################
  try:
    logging.debug("EDIT DISTANCE - Step 5 - Write predictions to file")
    with open(predictions_file, "a+") as pf:
      # pf.write("NEW EPOCH ***********************************************\n")
      for i in range(batch_size):
        pf.write("target     = {}\n".format(caption2seq(target[i])))
        pf.write("prediction = {}\n".format(caption2seq(prediction[i])))
  except:
    logging.debug("EDIT DISTANCE - Step 5 - Skip writing for epoch 0")
  
  return mean_edit_distance