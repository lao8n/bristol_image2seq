"""
Image2Seq Model with Attention Pipeline
Image Encoder with InceptionV3

MODEL  : image_embedding_dim 
INPUT  : batch_size x 299 x 299 x 3
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
  import keras_image2seq
  __package__ = "image2seq"

# Local imports #############################################################

#############################################################################
# IMAGE ENCODER IV3                                                             #
#############################################################################
class ImageEncoderIV3(keras.Model):
  def __init__(self, 
               image_embedding_dim,
               image_width=299,
               image_height=299,
               num_layers_freeze=0):
    
    super(ImageEncoderIV3, self).__init__()

    # MODEL PARAMETERS ######################################################
    self.image_embedding_dim = image_embedding_dim
    self.image_width = image_width
    self.image_height = image_height
    
    # LAYER 1 = InceptionV3 model feature map ###############################
    # Note that image_model.layers[-1].output == image_model.output
    # TODO: Could this become a bottleneck? Should each image be processed 
    # first by InceptionV3?
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')

    for i, layer in enumerate(image_model.layers):
      # layers are counted from 1 not 0
      if i + 1 <= num_layers_freeze:
        layer.trainable=False

    self.layer1_image_model = \
      keras.Model(inputs=image_model.input, 
                  outputs=image_model.layers[-1].output)
    
    # LAYER 2 = Reshaped feature map ########################################
    # Combine width and height output of InceptionV3 into one dimensional 
    # vector of size wxh  
    num_features = \
      (keras.backend.int_shape(self.layer1_image_model.output))[3]
    # don't include batch_size in Reshape
    self.layer2_reshape = keras.layers.Reshape((-1, num_features))

    # LAYER 3 = Dense features ##############################################
    self.layer3_dense = keras.layers.Dense(units=image_embedding_dim)
  
    # LAYER 4 = Encoder output ##############################################
    self.layer4_activation = keras.activations.relu

    # LAYER 5 = Encoder output with dropout #################################
    self.layer5_dropout = keras.layers.Dropout(0.1)

  def call(self, inputs, dropout=False):
    # LAYER 0 = Image input #################################################
    # Input               | Batch of images     | batch_size=None x         #
    #                     |                     | w=None(299) x             #
    #                     |                     | h=None(299) x             #
    #                     |                     | num_colours=3             #
    # ____________________|_____________________|___________________________#
    image_inputs = inputs[0]
    batch_size = image_inputs.shape[0]

    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER IV3 - Layer 0 - Process inputs - "
                  "image_inputs shape {}"
                  .format(K.int_shape(image_inputs)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(image_inputs),
      (batch_size, self.image_width, self.image_height, 3))

    # LAYER 1 = InceptionV3 model feature map ###############################
    # Inception V3        | Image feature map   | batch_size=None x         #
    #                     |                     | feature_map_w=None(8) x   #
    #                     |                     | feature_map_h=None(8) x   #
    #                     |                     | num_features=2048         #
    # ____________________|_____________________|___________________________#
    feature_map = self.layer1_image_model(image_inputs)

    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER IV3 - Layer 1 - InceptionV3 output - "
                  "feature_map shape {}"
                  .format(K.int_shape(feature_map)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(feature_map),
      (batch_size, 8, 8, 2048))

    # LAYER 2 = Reshaped feature map ########################################
    # Reshape             | Reshaped image      | batch_size=None x         #
    #                     | feature map         | feature_map_wxh=None(64) x#
    #                     |                     | num_features=2048         #
    # ____________________|_____________________|___________________________# 
    reshaped_feature_map = self.layer2_reshape(feature_map)
  
    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER IV3 - Layer 2 - Reshaped feature map - "
                  "reshaped_feature_map shape {}"
                  .format(K.int_shape(reshaped_feature_map)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(reshaped_feature_map),
      (batch_size, 64, 2048))

    # LAYER 3 = Dense features ##############################################
    # Dense               | Dense features      | batch_size=None x         #
    #                     |                     | feature_map_wxh=None(64) x#
    #                     |                     | image_embedding_dim=      #
    #                     |                     | image_embedding_dim       #
    # ____________________|_____________________|___________________________#
    dense_features = self.layer3_dense(reshaped_feature_map)

    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER IV3 - Layer 3 - Dense features - "
                  "dense_features shape {}"
                  .format(K.int_shape(dense_features)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(dense_features),
      (batch_size, 64, self.image_embedding_dim))

    # LAYER 4 = Encoder output ##############################################
    # Activations         | Encoder output      | batch_size=None x         #
    #                     |                     | feature_map_wxh=None(64) x#
    #                     |                     | image_embedding_dim=      #
    #                     |                     | image_embedding_dim       # 
    # ____________________|_____________________|___________________________#
    encoder_output = self.layer4_activation(dense_features)
  
    # Logging, Debug & Assert
    logging.debug("IMAGE ENCODER IV3 - Layer 4 - Encoder output - "
                  "encoder_output shape {}"
                  .format(K.int_shape(encoder_output)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(encoder_output),
      (batch_size, 64, self.image_embedding_dim))

    # LAYER 5 = Optional training layer #####################################
    if dropout:
      encoder_output = self.layer5_dropout(encoder_output)

    return encoder_output
