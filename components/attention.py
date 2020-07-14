"""
Image2Seq Model with Attention Pipeline
Attention

MODEL  : input_embedding_dim, decoder_hidden_dim
INPUT  : batch_size x input_feature_map_dim x input_embedding_dim 
         batch_size x decoder_hidden_dim
OUTPUT : batch_size x decoder_hidden_dim
         batch_size x input_feature_map_dim x score_dim=1
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
# Attention                                                                 #
#############################################################################
class Attention(keras.Model):
  def __init__(self,
               input_embedding_dim,
               decoder_hidden_dim):
    
    super(Attention, self).__init__()  

    # MODEL PARAMETERS ######################################################
    self.input_embedding_dim = input_embedding_dim
    self.decoder_hidden_dim = decoder_hidden_dim

    # LAYER 1 = Expand dims of hidden input #################################
    self.layer1_expand = K.expand_dims 

    # LAYER 2 = Dense inputs ################################################
    self.layer2_dense_input = keras.layers.Dense(decoder_hidden_dim)
    self.layer2_dense_decoder = keras.layers.Dense(decoder_hidden_dim)

    # LAYER 3 = Tanh activation #############################################
    self.layer3_activation = keras.activations.tanh

    # LAYER 4 = Dense score #################################################
    self.layer4_dense = keras.layers.Dense(1)

    # LAYER 5 = Attention weights ###########################################
    self.layer5_activation = keras.activations.softmax 

    # LAYER 6 = Attention weighted image decoder output #####################
    self.layer6_multiplication = keras.layers.multiply

    # LAYER 7 = Context vector ##############################################
    self.layer7_sum = K.sum 

    # LAYER 8 = Dropout #####################################################
    self.layer8_dropout = keras.layers.Dropout(0.1)

  def call(self, inputs, dropout=False):
    # LAYER 0 = Attention input and decoder hidden state ####################
    # Input               | Attention input     | batch_size=None x         # 
    #                     |                     | input_feature_map_dim=    #
    #                     |                     | input_feature_map_dim x   #
    #                     |                     | input_embedding_dim=      #
    #                     |                     | input_embedding_dim       #
    #                     |                     |                           #
    #                     | Decoder hidden      | batch_size=None x         #
    #                     | state               | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        #
    # ____________________|_____________________|___________________________#
    attention_input = inputs[0]
    input_feature_map_dim = attention_input.shape[1]
    decoder_hidden_state = inputs[1]
    batch_size = decoder_hidden_state.shape[0]

    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 0 - Process inputs - "
                  "attention_input shape {}"
                  .format(K.int_shape(attention_input)))
    logging.debug("ATTENTION - Layer 0 - Process inputs - "
                  "decoder_hidden_state shape {}"
                  .format(K.int_shape(decoder_hidden_state)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(attention_input),
      (batch_size, input_feature_map_dim, self.input_embedding_dim))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(decoder_hidden_state),
      (batch_size, self.decoder_hidden_dim))

    # LAYER 1 = Expand dims of hidden input #################################
    # Expand dims         | Decoder hidden      | batch_size=None x         #
    #                     | state per feature   | input_feature_map_index   #
    #                     | map index           | =1 x                      #
    #                     |                     | decoder_hidden_dim=       # 
    #                     |                     | decoder_hidden_dim        #
    # ____________________|_____________________|___________________________#
    decoder_hidden_state_expanded = self.layer1_expand(decoder_hidden_state, 
                                                       axis=1)

    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 1 - Expand decoder dim - "
                  "decoder_hidden_state_expanded shape {}"
                  .format(K.int_shape(decoder_hidden_state_expanded)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(decoder_hidden_state_expanded),
      (batch_size, 1, self.decoder_hidden_dim))

    # LAYER 2 = Dense inputs ################################################
    # Dense               | Dense input         | batch_size=None x         #
    #                     |                     | input_feature_map_dim=    #
    #                     |                     | input_feature_map_dim x   #
    #                     |                     | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        # 
    #                     |                     |                           #
    #                     | Dense decoder       | batch_size=None x         #
    #                     | hidden state per    | input_feature_map_index   #
    #                     | encoder feature map | =1 x                      #
    #                     | index               | decoder_hidden_dim=       # 
    #                     |                     | decoder_hidden_dim        #
    # ____________________|_____________________|___________________________#  
    attention_input_dense = self.layer2_dense_input(attention_input)
    decoder_hidden_dense = \
      self.layer2_dense_decoder(decoder_hidden_state_expanded)

    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 2 - Dense attention input"
                  " - attention_input_dense shape {}"
                  .format(K.int_shape(attention_input_dense)))
    logging.debug("ATTENTION - Layer 2 - Dense decoder hidden"
                  " state - decoder_hidden_dense shape {}"
                  .format(K.int_shape(decoder_hidden_dense)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(attention_input_dense),
      (batch_size, input_feature_map_dim, self.decoder_hidden_dim))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(decoder_hidden_dense),
      (batch_size, 1, self.decoder_hidden_dim))    

    # LAYER 3 = Tanh activation #############################################
    # Tanh activation     | Attention scores    | batch_size=None x         #
    #                     |                     | input_feature_map_dim=    #
    #                     |                     | input_feature_map_dim x   #
    #                     |                     | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        #
    # ____________________|_____________________|___________________________#
    attention_scores = self.layer3_activation(attention_input_dense + 
                                              decoder_hidden_dense)

    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 3 - Tanh activation - "
                  "attention_scores shape {}"
                  .format(K.int_shape(attention_scores)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(attention_scores),
      (batch_size, input_feature_map_dim, self.decoder_hidden_dim))

    # LAYER 4 = Dense score #################################################
    # Dense Score         | Attention scores    | batch_size=None x         #
    #                     |                     | input_feature_map_dim=    #
    #                     |                     | input_feature_map_dim x   #  
    #                     |                     | score_dim=1               #
    # ____________________|_____________________|___________________________#
    attention_scores_dense = self.layer4_dense(attention_scores)

    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 4 - Dense scores - "
                  "attention_scores shape {}"
                  .format(K.int_shape(attention_scores_dense)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(attention_scores_dense),
      (batch_size, input_feature_map_dim, 1))

    # LAYER 5 = Attention weights ###########################################
    # Softmax activation  | Attention weights   | batch_size=None x         #
    #                     |                     | input_feature_map_dim=    #
    #                     |                     | input_feature_map_dim x   #
    #                     |                     | score_dim=1               #
    # ____________________|_____________________|___________________________#
    attention_weights = self.layer5_activation(attention_scores_dense, 
                                               axis=1)
    
    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 5 - Attention weights - "
                  "attention_weights shape {}"
                  .format(K.int_shape(attention_weights)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(attention_weights),
      (batch_size, input_feature_map_dim, 1))

    # LAYER 6 = Attention weighted input ####################################
    # Multiplied weights  | Attention weighted  | batch_size=None x         #
    #                     | input               | input_feature_map_dim=    #
    #                     |                     | input_feature_map_dim x   #
    #                     |                     | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        #
    # ____________________|_____________________|___________________________#
    attention_weighted_input = \
      self.layer6_multiplication([attention_weights, attention_input_dense])

    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 6 - Attention weighted "
                  "encoder output shape {}"
                  .format(K.int_shape(attention_weighted_input)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(attention_weighted_input),
      (batch_size, input_feature_map_dim, self.decoder_hidden_dim))

    # LAYER 7 = Context vector ##############################################
    # Summed weights      | Context vector      | batch_size=None x         # 
    #                     |                     | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        #  
    # ____________________|_____________________|___________________________#
    context_vector = self.layer7_sum(attention_weighted_input,
                                     axis=1)
    
    # Logging, Debug & Assert
    logging.debug("ATTENTION - Layer 7 - Context vector - "
                  "context_vector shape {}"
                  .format(K.int_shape(context_vector)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(context_vector),
      (batch_size, self.decoder_hidden_dim))

    # LAYER 8 = Dropout #####################################################
    if dropout:
      context_vector = self.layer8_dropout(context_vector)

    return context_vector, attention_weights

