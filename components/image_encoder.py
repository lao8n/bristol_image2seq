"""
Image2Seq Model with Attention Pipeline
Image Encoder

MODEL  : image_embedding_dim 
INPUT  : batch_size x feature_map_wxh=64 x num_features=2048
OUTPUT : batch_size x feature_map_wxh=64 x image_embedding_dim
"""

#############################################################################
# Imports                                                                   #
#############################################################################

# Standard imports ##########################################################
import logging
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

# Local imports #############################################################

#############################################################################
# Image Encoder                                                             #
#############################################################################
class ImageEncoder(keras.Model):
  def __init__(self, 
               image_embedding_dim):
    
    super(ImageEncoder, self).__init__()

    # MODEL PARAMETERS ######################################################
    self.image_embedding_dim = image_embedding_dim

    # LAYER 1 = Dense features ##############################################
    self.layer1_dense = keras.layers.Dense(units=image_embedding_dim)
  
    # LAYER 2 = Encoder output ##############################################
    self.layer2_activation = keras.activations.relu

    # LAYER 3 = Encoder output with dropout #################################
    self.layer3_dropout = keras.layers.Dropout(0.1)

  def call(self, inputs, dropout=False):
    # LAYER 0 = Image input #################################################
    # Input               | Batch of images     | batch_size=None x         #
    #                     |                     | feature_map_wxh=None(64)  #
    #                     |                     | num_features=2048         #
    # ____________________|_____________________|___________________________#
    image_inputs = inputs[0]
    batch_size = image_inputs.shape[0]

    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER - Layer 0 - Process inputs - "
                  "image_inputs shape {}"
                  .format(K.int_shape(image_inputs)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(image_inputs),
      (batch_size, 64, 2048))

    # LAYER 1 = Dense features ##############################################
    # Dense               | Dense features      | batch_size=None x         #
    #                     |                     | feature_map_wxh=None(64) x#
    #                     |                     | image_embedding_dim=      #
    #                     |                     | image_embedding_dim       #
    # ____________________|_____________________|___________________________#
    dense_features = self.layer1_dense(image_inputs)

    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER - Layer 1 - Dense features - "
                  "dense_features shape {}"
                  .format(K.int_shape(dense_features)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(dense_features),
      (batch_size, 64, self.image_embedding_dim))

    # LAYER 2 = Encoder output ##############################################
    # Activations         | Encoder output      | batch_size=None x         #
    #                     |                     | feature_map_wxh=None(64) x#
    #                     |                     | image_embedding_dim=      #
    #                     |                     | image_embedding_dim       # 
    # ____________________|_____________________|___________________________#
    encoder_output = self.layer2_activation(dense_features)
  
    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER - Layer 2 - Encoder output - "
                  "encoder_output shape {}"
                  .format(K.int_shape(encoder_output)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(encoder_output),
      (batch_size, 64, self.image_embedding_dim))

    # LAYER 3 = Optional training layer #####################################
    if dropout:
      encoder_output = self.layer3_dropout(encoder_output)

    return encoder_output