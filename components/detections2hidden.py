"""
Image2Seq Model with Attention Pipeline
MLP

MODEL  : 
INPUT  : batch_size x detections_seq_len x token_embedding_dim
OUTPUT : batch_size x decoder_hidden_dim
"""
#############################################################################
# Imports                                                                   #
#############################################################################

# Standard imports ##########################################################
import logging
import numpy as np
import os
import sys

# Stop pycache ##############################################################
sys.dont_write_bytecode = True

# Keras and tensorflow imports ##############################################
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow.keras as keras
import tensorflow.keras.backend as K

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, 
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
  import image2seq
  __package__ = "image2seq"

#############################################################################
# Detections2Hidden                                                         #
#############################################################################
class Detections2Hidden(keras.Model):
  def __init__(self,
               decoder_hidden_dim):
  
    super(Detections2Hidden, self).__init__()

    # MODEL PARAMETERS ######################################################
    self.decoder_hidden_dim = decoder_hidden_dim

    # LAYER 1 = Flatten #####################################################
    # Flatten has error: 
    # 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 
    # 'lower'
    self.layer1_flatten = keras.layers.Flatten()

    # LAYER 2 = Dense hidden dim output #####################################
    self.layer2_dense = keras.layers.Dense(units=decoder_hidden_dim)

    # LAYER 3 = Dropout #####################################################
    self.layer3_dropout = keras.layers.Dropout(0.1)

  def call(self, inputs, dropout=False):
    # LAYER 0 = Process inputs ##############################################
    # Input               | Detections2Hidden   | batch_size=None x         #
    #                     | Input               | detections_seq_len=       #
    #                     |                     | detections_seq_len x      #
    #                     |                     | token_embedding_dim=      #
    #                     |                     | token_embedding_dim       #
    #_____________________|_____________________|___________________________#
    detections2hidden_input = inputs[0]
    batch_size = detections2hidden_input.shape[0]
    detections_seq_len = detections2hidden_input.shape[1]
    token_embedding_dim = detections2hidden_input.shape[2]

    # Logging, Debug & Assert
    logging.debug("DETECTIONS2HIDDEN CALL - Step 0 - Process inputs - "
                  "batch_size {}".format(batch_size))
    logging.debug("DETECTIONS2HIDDEN CALL - Step 0 - Process inputs - "
                  "detections2hidden shape {}"
                  .format(K.int_shape(detections2hidden_input)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(detections2hidden_input),
      (batch_size, detections_seq_len, token_embedding_dim))

    # STEP 1 = Flatten ######################################################
    # Flatten             | Flattened           | batch_size=None x         #
    #                     | Detections          | flatten_dim=              #
    #                     |                     | detections_seq_len =      #
    #                     |                     | detections_seq_len x      #
    #                     |                     | token_embedding_dim =     #
    #                     |                     | token_embedding_dim       #
    #_____________________|_____________________|___________________________#
    detections_flattened = self.layer1_flatten(detections2hidden_input)

    # Logging, Debug & Assert
    logging.debug("DETECTIONS2HIDDEN CALL - Step 1 - Flatten inputs - "
                  "detections_flattened shape {}"
                  .format(detections_flattened))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(detections_flattened),
      (batch_size, detections_seq_len * token_embedding_dim))

    # STEP 2: Dense hidden state ############################################
    # Dense               | Dense detections    | batch_size=None x         #
    #                     |                     | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        #
    #_____________________|_____________________|___________________________#
    detections_hidden_state = self.layer2_dense(detections_flattened)

    # Logging, Debug & Assert
    logging.debug("DETECTIONS2HIDDEN CALL - Step 2 - Dense output - "
                  "detections_hidden_state shape {}"
                  .format(detections_hidden_state))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(detections_hidden_state),
      (batch_size, self.decoder_hidden_dim))

    # STEP 3 = Dropout #####################################################
    if dropout:
      detections_hidden_state = self.layer3_dropout(detections_hidden_state)
    
    return detections_hidden_state