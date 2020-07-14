"""
Image2Seq Model with Attention Pipeline
Token Embedding

MODEL  : token_vocab_size, token_embedding_dim
INPUT  : batch_size x token_seq_len
OUTPUT : batch_size x token_seq_len x token_embedding_dim
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
                  os.path.join(os.path.dirname(__file__), '..', '..'))
  import image2seq
  __package__ = "image2seq"

#############################################################################
# Token Embedding                                                           #
#############################################################################
class TokenEmbedding(keras.Model):
  def __init__(self, 
               token_vocab_size, 
               token_embedding_dim):
    
    super(TokenEmbedding, self).__init__()

    # MODEL PARAMETERS ######################################################
    self.token_vocab_size = token_vocab_size
    self.token_embedding_dim = token_embedding_dim

    # LAYER 1 = Token sequence embeddings ###################################
    self.layer1_embedding = \
      keras.layers.Embedding(input_dim=self.token_vocab_size,
                             output_dim=self.token_embedding_dim)
    
    # LAYER 2 = Dropout #####################################################
    self.layer2_dropout = keras.layers.Dropout(0.1)

  def call(self, inputs, dropout=False):
    # LAYER 0 = Token inputs ################################################
    # Input               | Batch of token seqs | batch_size=None x         #
    #                     |                     | token_seq_len =           #
    #                     |                     | token_seq_len             #
    # ____________________|_____________________|___________________________#
    token_seq_inputs = inputs[0]
    batch_size = token_seq_inputs.shape[0]
    token_seq_len = token_seq_inputs.shape[1]

    # Logging, Debug & Assert
    logging.debug("TOKEN EMBEDDING - Layer 0 - Process inputs - "
                  "token_seq_inputs shape {}"
                  .format(K.int_shape(token_seq_inputs)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(token_seq_inputs),
      (batch_size, token_seq_len))

    # LAYER 1 = Token sequence embeddings ###################################
    # Embedding           | Token embedding     | batch_size=None x         #
    #                     |                     | token_seq_len =           #
    #                     |                     | token_seq_len x           #
    #                     |                     | token_embedding_dim=      #
    #                     |                     | token_embedding_dim       # 
    # ____________________|_____________________|___________________________#
    token_seq_embeddings = self.layer1_embedding(token_seq_inputs)

    # Logging, Debug & Assert
    logging.debug("TOKEN EMBEDDING - Layer 1 - Token sequence embedding - "
                  "token_seq_embeddings shape {}"
                  .format(K.int_shape(token_seq_embeddings)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(token_seq_embeddings),
      (batch_size, token_seq_len, self.token_embedding_dim))

    # LAYER 2 = Optional training layer #####################################
    if dropout:
      token_seq_embeddings = self.layer2_dropout(token_seq_embeddings)

    return token_seq_embeddings