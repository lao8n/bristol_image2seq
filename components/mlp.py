"""
Image2Seq Model with Attention Pipeline
MLP

MODEL  : mlp_hidden_dim, token_vocab_size
INPUT  : batch_size x token_seq_len=1 x mlp_input_dim
OUTPUT : batch_size x token_vocab_size
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
# MLP                                                                       #
#############################################################################
class MLP(keras.Model):
  def __init__(self, 
               mlp_hidden_dim,
               token_vocab_size):
  
    super(MLP, self).__init__()

    # MODEL PARAMETERS ######################################################
    self.mlp_hidden_dim = mlp_hidden_dim
    self.token_vocab_size = token_vocab_size

    # LAYER 1 = Dense MLP ###################################################
    self.layer1_dense = keras.layers.Dense(units=mlp_hidden_dim)

    # LAYER 2 = Reshape to drop token_seq_len dimension #####################
    self.layer2_reshape = keras.backend.reshape 

    # LAYER 3 = Dense vocab output ##########################################
    self.layer3_dense = keras.layers.Dense(units=token_vocab_size)

    # LAYER 4 = Dropout #####################################################
    self.layer4_dropout = keras.layers.Dropout(0.1)

  def call(self, inputs, dropout=False):
    # LAYER 0 = Process inputs ##############################################
    # MLP Input           | MLP Input           | batch_size=None x         #
    #                     |                     | token_seq_len=1 x         #
    #                     |                     | mlp_input_dim =           #
    #                     |                     | mlp_input_dim             #
    #_____________________|_____________________|___________________________#
    mlp_input = inputs[0]
    batch_size = mlp_input.shape[0]
    mlp_input_dim = mlp_input.shape[2]

    # Logging, Debug & Assert
    logging.debug("MLP- Layer 0 - Process inputs - "
                  "mlp_input shape {}"
                  .format(K.int_shape(mlp_input)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(mlp_input),
      (batch_size, 1, mlp_input_dim))

    # LAYER 1 = Dense MLP ###################################################
    # Dense               | MLP Output          | batch_size=None x         #
    #                     |                     | token_seq_len=1 x         #
    #                     |                     | mlp_hidden_dim =          #
    #                     |                     | mlp_hidden_dim            #
    #_____________________|_____________________|___________________________#
    mlp_output = self.layer1_dense(mlp_input)

    # Logging, Debug & Assert
    logging.debug("MLP - Layer 1 - Dense MLP - "
                  "mlp_output shape {}"
                  .format(K.int_shape(mlp_output)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(mlp_output),
      (batch_size, 1, self.mlp_hidden_dim))

    # LAYER 2 = Reshape MLP output ##########################################
    # Reshape             | Reshaped MLP output | batch_size=None x         #
    #                     |                     | mlp_hidden_dim =          #
    #                     |                     | mlp_hidden_dim            #
    #_____________________|_____________________|___________________________#
    mlp_output_reshaped = \
      self.layer2_reshape(mlp_output,
                          shape=(-1, mlp_output.shape[2]))
    
    # Logging, Debug & Assert
    logging.debug("MLP - Layer 2 - Reshaped MLP - "
                  "mlp_output_shape {}"
                  .format(K.int_shape(mlp_output_reshaped)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(mlp_output_reshaped),
      (batch_size, self.mlp_hidden_dim))

    # LAYER 3 = Dense vocab size ############################################
    # Dense               | Predicted token     | batch_size=None x         #
    #                     |                     | token_vocab_size x        #
    #                     |                     | token_vocab_size          #
    #_____________________|_____________________|___________________________#
    single_token_prediction = self.layer3_dense(mlp_output_reshaped)

    # Logging, Debug & Assert
    logging.debug("MLP - Layer 3 - Prediction - "
                  "single_token_prediction shape {}"
                  .format(K.int_shape(single_token_prediction)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(single_token_prediction),
      (batch_size, self.token_vocab_size))
    
    # Layer 7 = Dropout #####################################################
    if dropout:
      single_token_prediction = self.layer4_dropout(single_token_prediction)

    return single_token_prediction

    