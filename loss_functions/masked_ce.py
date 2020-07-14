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

def masked_ce_loss_fn(target, 
                      prediction, 
                      batch_size=64, 
                      token_vocab_size=17):
  
  # STEP 0: Debug inputs ##################################################
  # Logging, Debug & Assert
  logging.debug("MASKED CE LOSS - Step 0 - Process inputs - "
                "target shape {}"
                .format(K.int_shape(target)))
  logging.debug("MASKED CE LOSS - Step 0 - Process inputs - "
                "prediction shape {}"
                .format(K.int_shape(prediction)))
  logging.debug("MASKED CE LOSS - Step 0 - Process inputs - "
                "target \n{}"
                .format(target))
  logging.debug("MASKED CE LOSS - Step 0 - Process inputs - "
                "argmaxed prediction\n{}"
                .format(tf.argmax(prediction, axis=1)))
  tf.compat.v1.debugging.assert_equal(K.int_shape(target),
                            (batch_size))
  tf.compat.v1.debugging.assert_equal(K.int_shape(prediction),
                            (batch_size, token_vocab_size))

  # STEP 1: Cast inputs ###################################################
  target=tf.cast(target, dtype=tf.float64)
  prediction=tf.cast(prediction, dtype=tf.float64)
  
  # Logging, Debug & Assert
  logging.debug("MASKED CE LOSS - Step 1 - Cast (no info)")

  # STEP 2: Calculate losses ##############################################
  cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(
    y_true=target,
    y_pred=prediction,
    from_logits=True)

  # Logging, Debug & Assert
  logging.debug("MASKED CE LOSS- Step 2 - Cross entropy loss - "
                "cross_entropy_loss shape {}"
                .format(K.int_shape(cross_entropy_loss)))
  logging.debug("MASKED CE LOSS - Step 2 - Cross entropy loss ")
  for i, loss in enumerate(cross_entropy_loss):
    logging.debug("Index: {0:2d} Loss: {1:2.4f}".format(i, loss))
  tf.compat.v1.debugging.assert_equal(K.int_shape(cross_entropy_loss),
                            (batch_size))

  # STEP 3: Calculate masks ###############################################
  target_mask = tf.math.logical_not(tf.math.equal(target, 0))
  target_mask = tf.cast(target_mask, dtype=cross_entropy_loss.dtype)

  masked_cross_entropy_loss = \
    tf.math.multiply(cross_entropy_loss, target_mask)

  # Logging, Debug & Assert
  logging.debug("MASKED CE LOSS - Step 3 - Target Mask {}"
                .format(target_mask))
  logging.debug("MASKED CE LOSS - Step 3 - Masked cross entropy loss"
                " - masked cross_entropy_loss shape {}"
                .format(K.int_shape(
                  masked_cross_entropy_loss)))
  logging.debug("MASKED CE LOSS - Step 3 - Masked cross entropy loss ")
  for i, loss in enumerate(masked_cross_entropy_loss):
    logging.debug("Index: {0:2d} Loss: {1:2.4f}".format(i, loss))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(masked_cross_entropy_loss),
    (batch_size))  

  # STEP 4: Reduce Mean ###################################################
  single_prediction_loss = tf.reduce_mean(masked_cross_entropy_loss)

  # Logging, Debug & Assert
  logging.debug("MASKED CE LOSS - Step 4 - Single prediction loss - "
                "single_prediction_loss shape {}"
                .format(K.int_shape(single_prediction_loss)))
  logging.debug("MASKED CE LOSS - Step 4 - Single prediction loss {}"
                .format(single_prediction_loss))
  tf.compat.v1.debugging.assert_equal(
    K.int_shape(single_prediction_loss),
    ())       

  return single_prediction_loss