"""
Image2Seq Model with Attention Pipeline
Image2Seq Model = Encoder-Decoder with Attention Drake Concatenated input

"""
#############################################################################
# Imports                                                                   #
#############################################################################

# Standard imports ##########################################################
import logging
import numpy as np
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
  import image2seq
  __package__ = "image2seq"

# Local imports #############################################################
from image2seq.components.attention import Attention 
from image2seq.components.image_encoder import ImageEncoder
from image2seq.components.image_encoder_inception_v3 import ImageEncoderIV3
from image2seq.components.mlp import MLP
from image2seq.components.token_embedding import TokenEmbedding 
from image2seq.components.detections2hidden import Detections2Hidden
from image2seq.loss_functions.masked_ce import masked_ce_loss_fn
from image2seq.metrics.edit_distance import edit_distance_metric

#############################################################################
# Image2Seq Model                                                           #
#############################################################################
class DRAKECONCAT(keras.Model):
  def __init__(self,
               image_embedding_dim=512,
               token_embedding_dim=8,
               token_vocab_size=17,
               decoder_hidden_dim=1024,
               mlp_hidden_dim=1024,
               image_encoder="inceptionv3"):
    
    super(DRAKECONCAT, self).__init__()

    # Model Parameters ######################################################
    self.image_embedding_dim = image_embedding_dim
    self.token_embedding_dim = token_embedding_dim
    self.token_vocab_size = token_vocab_size
    self.decoder_hidden_dim = decoder_hidden_dim
    self.mlp_hidden_dim = mlp_hidden_dim
    self.batch_size = None
    self.predictions_file = None
    self.image_encoder = image_encoder

    # MODEL 1 = Image Encoding ##############################################
    if image_encoder=="inceptionv3":
      self.model1_image_encoding = \
        ImageEncoderIV3(image_embedding_dim=image_embedding_dim)
    else:
      self.model1_image_encoding = \
        ImageEncoder(image_embedding_dim=image_embedding_dim)
    
    # MODEL 2 = Token Sequence Embedding ####################################
    self.model2_token_embedding = \
      TokenEmbedding(token_vocab_size=token_vocab_size,
                     token_embedding_dim=token_embedding_dim)
    
    # MODEL 3 = Attention ###################################################
    self.model3_attention = \
      Attention(input_embedding_dim=image_embedding_dim,
                decoder_hidden_dim=decoder_hidden_dim)

    # MODEL 4 = MLP #########################################################
    self.model4_mlp = \
      MLP(mlp_hidden_dim=mlp_hidden_dim,
          token_vocab_size=token_vocab_size)

    # MODEL 5 - Detection2Hidden ############################################
    self.model5_detections2hidden = \
      Detections2Hidden(decoder_hidden_dim=decoder_hidden_dim)

    # LAYER 1 = Expand dims #################################################
    self.layer1_expand_dims = keras.backend.expand_dims 

    # LAYER 2 = Concatenate encoder output with attention ###################
    self.layer2_concatenate = keras.layers.concatenate 

    # LAYER 3 = LSTM ########################################################
    self.layer3_lstm = keras.layers.LSTM(decoder_hidden_dim, 
                                         return_sequences=True,
                                         return_state=True, 
                                         dropout=0.1)

  def call(self, inputs, val_mode=False, dropout=False):      
    # Train or validation mode ##############################################
    if val_mode:
      logging.debug("MODEL DRAKE CONCAT CALL - Train mode")
    else:
      logging.debug("MODEL DRAKE CONCAT CALL - Validation mode")
    
    # STEP 0: Process Inputs ################################################
    # Input               | Image feature maps  | batch_size=None x         #
    #                     |                     | feature_map_wxh=None(64) x#
    #                     |                     | num_features=2048         #
    #                     | Token input         | batch_size = None x       #
    #                     |                     | token_seq_len =  x        #
    #                     |                     | token_seq_len             #
    #                     | RetinaNet input     | batch_size = None x       #
    #                     |                     | detections_seq_len =      #
    #                     |                     | detections_seq_len        #
    #_____________________|_____________________|___________________________#
    input_image = inputs[0]
    input_tokens = inputs[1]
    retinanet_input = inputs[2]
    self.batch_size = input_tokens.shape[0]
    batch_token_seq_len = input_tokens.shape[1]
    batch_detections_seq_len = retinanet_input.shape[1]

    # Logging, Debug & Assert 
    logging.debug("MODEL DRAKE CONCAT CALL - Step 0 - Process inputs - "
                  "batch_size {}".format(self.batch_size))
    logging.debug("MODEL DRAKE CONCAT CALL - Step 0 - Process inputs - "
                  "input_image shape {}"
                  .format(K.int_shape(input_image)))
    logging.debug("MODEL DRAKE CONCAT CALL - Step 0 - Process inputs - "
                  "input_tokens shape {}"
                  .format(K.int_shape(input_tokens)))
    logging.debug("MODEL DRAKE CONCAT CALL - Step 0 - Process inputs - "
                  "retinanet_input shape {}"
                  .format(K.int_shape(retinanet_input)))
    if self.image_encoder == "inceptionv3":
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(input_image),
        (self.batch_size, 299, 299, 3))
    else:
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(input_image),
        (self.batch_size, 64, 2048))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(input_tokens),
      (self.batch_size, batch_token_seq_len))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(retinanet_input),
      (self.batch_size, batch_detections_seq_len))

    # STEP 1: Set Decoder Hidden State ######################################
    # Assign              | Initial decoder     | batch_size=None x         #
    #                     | hidden state        | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        #
    #_____________________|_____________________|___________________________#
    retinanet_detections_token_embedded = \
      self.model2_token_embedding([retinanet_input],
                                  dropout=dropout)

    # Logging, Debug & Assert
    logging.debug("MODEL DRAKE CONCAT CALL - Step 1 - RetinaNet "
                  "detections token embedding shape {}"
                  .format(K.int_shape(retinanet_detections_token_embedded)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(retinanet_detections_token_embedded),
      (self.batch_size, batch_detections_seq_len, self.token_embedding_dim))
    
    decoder_hidden_state = \
      self.model5_detections2hidden([retinanet_detections_token_embedded],
                                    dropout=dropout)
    
    # Logging, Debug & Assert
    logging.debug("MODEL DRAKE CONCAT CALL - Step 1 - Initial hidden "
                  "state - decoder_hidden_state shape {}"
                 .format(K.int_shape(decoder_hidden_state)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(decoder_hidden_state), 
      (self.batch_size, self.decoder_hidden_dim))
    
    # STEP 2: Image Encoding ################################################
    # Dense + Activations | Image encoder       | batch_size=None x         #
    #                     | output              | feature_map_wxh=None(64)  #
    #                     |                     | image_embedding_dim=      #
    #                     |                     | image_embedding_dim       #
    #_____________________|_____________________|___________________________#    
    input_image_features = \
      self.model1_image_encoding([input_image],
                                 dropout=dropout)
    
    # Logging, Debug & Assert
    logging.debug("MODEL DRAKE CONCAT CALL - Step 2 - Image encoding dense"
                  " and activations - input_image_features shape {}"
                  .format(K.int_shape(input_image_features)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(input_image_features),
      (self.batch_size, 64, self.image_embedding_dim))

    # STEP 3: Token Embedding for all batch input sequences #################
    # Embedding           | Token embedding     | batch_size=None x         #
    #                     |                     | token_seq_len =           #
    #                     |                     | token_seq_len x           #
    #                     |                     | token_embedding_dim=      #
    #                     |                     | token_embedding_dim       # 
    # ____________________|_____________________|___________________________#
    input_token_embeddings = \
      self.model2_token_embedding([input_tokens],
                                  dropout=dropout)
    
    # Logging, Debug & Assert 
    logging.debug("MODEL DRAKE CONCAT CALL - Step 3 - Token embeddings - "
                  "target_token_embeddings shape {}"
                  .format(keras.backend.int_shape(input_token_embeddings)))
    tf.compat.v1.debugging.assert_equal(
      keras.backend.int_shape(input_token_embeddings),
      (self.batch_size, batch_token_seq_len, self.token_embedding_dim))    

    # STEP 4: Decoder inputs is a 'GO' ######################################
    # Slice + Expand dims | GO column           | batch_size=None x         #
    #                     |                     | token_seq_len = 1 x       #
    #                     |                     | token_embedding_dim=      #
    #                     |                     | token_embedding_dim       # 
    # ____________________|_____________________|___________________________#
    # For first character input is always  GO = 1 at index 0
    # Both for teaching forcing mode and validation mode
    decoder_token_input = \
      K.expand_dims(input_token_embeddings[:, 0], 1)

    # Logging, Debug, & Assert
    logging.debug("MODEL DRAKE CONCAT CALL - Step 4 - Decoder inputs - "
                  "decoder_teaching_forcing_inputs shape {}"
                  .format(K.int_shape(
                    decoder_token_input)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(decoder_token_input),
      (self.batch_size, 1, self.token_embedding_dim))

    # STEP 5: Loop through token sequence ###################################
    batch_loss = 0
    batch_mean_edit_distance = 0
    if val_mode:
      list_predictions = []
    for i in range(1, batch_token_seq_len):
      # STEP 5.1: Attention #################################################
      # Summed weights    | Context vector      | batch_size=None x         # 
      #                   |                     | decoder_hidden_dim=       #
      #                   |                     | decoder_hidden_dim        #  
      # __________________|_____________________|___________________________#
      context_vector, attention_weights = \
        self.model3_attention([input_image_features, decoder_hidden_state],
                              dropout=dropout)
      
      # Logging, Debug & Assert 
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.1 - Attention - "
                    "Context vector shape {}"
                    .format(K.int_shape(context_vector)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(context_vector),
        (self.batch_size, self.decoder_hidden_dim))

      # STEP 5.2: LSTM Input ################################################
      # Expand +          | LSTM input          | batch_size=None x         # 
      # Concatenate       |                     | token_seq_len=1 x         #
      #                   |                     | lstm_input_dim=           #  
      #                   |                     | decoder_hidden_dim +      #
      #                   |                     | token_embedding_dim       #
      # __________________|_____________________|___________________________#
      context_vector_expanded = self.layer1_expand_dims(context_vector, 1)

      # Logging, Debug & Assert 
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.2 - Expand context "
                    "vector - context_vector_expanded shape {}"
                    .format(K.int_shape(context_vector_expanded)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(context_vector_expanded),
        (self.batch_size, 1, self.decoder_hidden_dim)) 

      lstm_input = self.layer2_concatenate([context_vector_expanded,
                                            decoder_token_input],
                                           axis=-1)

      # Logging, Debug & Assert  
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.2 - Concat context"
                    "vector and token embedding - lstm_input shape {}"
                    .format(K.int_shape(lstm_input)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(lstm_input),
        (self.batch_size, 1, 
         self.token_embedding_dim + self.decoder_hidden_dim))      

      # STEP 5.3: LSTM ######################################################
      # LSTM return       | LSTM Output         | batch_size=None x         #
      # sequences and     |                     | token_seq_len=1 x         #
      # state             |                     | decoder_hidden_dim=       #
      #                   |                     | decoder_hidden_dim        #
      #                   |                     |                           #    
      #                   | LSTM Hidden State   | batch_size=None x         #
      #                   |                     | decoder_hidden_dim=       #
      #                   |                     | decoder_hidden_dim        #
      #                   |                     |                           #    
      #                   | LSTM Cell State     | batch_size=None x         #
      #                   |                     | decoder_hidden_dim=       #
      #                   |                     | decoder_hidden_dim        #
      # __________________|_____________________|___________________________#      
      # If dropout = true then in training mode 
      # If dropout = false then in validation mode
      lstm_output, decoder_hidden_state, decoder_cell_state = \
        self.layer3_lstm(lstm_input, training=dropout)   
      
      # Logging, Debug & Assert
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.3 - LSTM output - "
                    "lstm_output shape {}"
                    .format(K.int_shape(lstm_output)))
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.3 - LSTM output - "
                    "decoder_hidden_state shape {}"
                    .format(K.int_shape(decoder_hidden_state)))
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.3 - LSTM output - "
                    "decoder_cell_state shape {}"
                    .format(K.int_shape(decoder_cell_state)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(lstm_output),
        (self.batch_size, 1, self.decoder_hidden_dim))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(decoder_hidden_state),
        (self.batch_size, self.decoder_hidden_dim))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(decoder_cell_state),
        (self.batch_size, self.decoder_hidden_dim))
      
      # STEP 5.4: MLP #######################################################
      # Dense             | Predicted token     | batch_size=None x         #
      #                   |                     | token_vocab_size x        #
      #                   |                     | token_vocab_size          #
      #___________________|_____________________|___________________________#
      single_token_prediction = self.model4_mlp([lstm_output],
                                                dropout=dropout)

      # Logging. Debug & Assert
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.4 - MLP output - "
                    "single_token_prediction shape {}"
                    .format(K.int_shape(single_token_prediction)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(single_token_prediction),
        (self.batch_size, self.token_vocab_size))

      # STEP 5.5: Calculate loss ############################################
      # Loss              | Single token loss   | int                       #
      #___________________|_____________________|___________________________#
      batch_loss += masked_ce_loss_fn(target=input_tokens[:, i],
                                      prediction=single_token_prediction,
                                      batch_size=self.batch_size,
                                      token_vocab_size=self.token_vocab_size)
  
      # Logging, Debug & Assert
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.5 - "
                    "Single prediction loss {}"
                    .format(batch_loss))

      # STEP 5.6 Update decoder input #######################################
      # Decoder input     | New decoder         | batch_size=None x         #
      #                   | hidden state        | decoder_hidden_dim=       #
      #                   |                     | decoder_hidden_dim        #
      #___________________|_____________________|___________________________#      
      if val_mode:
        # In validation mode use argmax output from decoder
        argmax_prediction = tf.argmax(single_token_prediction,
                                      axis=1,
                                      output_type=tf.dtypes.int32)
        list_predictions.append(argmax_prediction)
        argmax_prediction_expanded = K.expand_dims(argmax_prediction)
        decoder_token_input = \
          self.model2_token_embedding([argmax_prediction_expanded])
      else:
        # In training mode use teacher forcing inputs
        decoder_token_input = \
          K.expand_dims(input_token_embeddings[:, i], 1)

      # Logging, Debug & Assert
      logging.debug("MODEL DRAKE CONCAT CALL - Step 5.6 - Update decoder "
                    " input - decoder_token_input shape {}"
                    .format(K.int_shape(decoder_token_input)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(decoder_token_input),
        (self.batch_size, 1, self.token_embedding_dim))
    
    # STEP 6: Calculate levenstein distance 
    if val_mode:
      stack_predictions = tf.stack(list_predictions, axis=1)

      # Logging, Debug & Assert
      # logging.debug("MODEL EDA XU CALL - Step 6 - Stack predictions  \n{}"
      #               .format(stack_predictions))
      logging.debug("MODEL DRAKE CONCAT CALL - Step 6 - Stack predictions "
                    "shape {}"
                    .format(K.int_shape(stack_predictions)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(stack_predictions),
        (self.batch_size, batch_token_seq_len - 1))

      batch_mean_edit_distance = \
        edit_distance_metric(target=input_tokens[:,1:],
                             prediction=stack_predictions,
                             predictions_file=self.predictions_file)
      
    # STEP 7: Return word sequence batch loss ###############################
    return batch_loss, batch_mean_edit_distance

  def get_model_name(self):
    return "drake_concat"

  def set_predictions_file(self, predictions_file):
    self.predictions_file = predictions_file