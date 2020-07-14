"""
Image2Seq Model with Attention Pipeline
Image2Seq Model = Encoder-Decoder with Attention - Drake nested with encoder-
decoder with attention with MLP

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
from image2seq.loss_functions.masked_ce import masked_ce_loss_fn
from image2seq.metrics.edit_distance import edit_distance_metric

#############################################################################
# Image2Seq Model                                                           #
#############################################################################
class DRAKENESTED(keras.Model):
  def __init__(self,
               image_embedding_dim=512,
               token_embedding_dim=8,
               token_vocab_size=17,
               decoder_hidden_dim=1024,
               mlp_hidden_dim=1024,
               image_encoder="inceptionv3"):
    
    super(DRAKENESTED, self).__init__()

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
    
    # MODEL 3 = Attention (inner) ###########################################
    self.model3_attention = \
      Attention(input_embedding_dim=image_embedding_dim,
                decoder_hidden_dim=decoder_hidden_dim)

    # MODEL 4 = MLP #########################################################
    self.model4_mlp = \
      MLP(mlp_hidden_dim=mlp_hidden_dim,
          token_vocab_size=token_vocab_size)

    # MODEL 5 = Attention (outer) ###########################################
    self.model5_attention = \
      Attention(input_embedding_dim=image_embedding_dim,
                decoder_hidden_dim=decoder_hidden_dim)
    
    # LAYER 1 = Expand dims #################################################
    self.layer1_expand_dims = keras.backend.expand_dims 

    # LAYER 2 = Concatenate encoder output with attention ###################
    self.layer2_concatenate = keras.layers.concatenate 

    # LAYER 3 = LSTM (inner) ################################################
    # TODO: Still need to return sequences even though only of length 1 in 
    # order to get both state and output (?)
    # self.layer3_lstm = keras.layers.LSTM(decoder_hidden_dim, 
    #                                      return_sequences=True,
    #                                      return_state=True, 
    #                                      dropout=0.2,
    #                                      recurrent_dropout=0.2)
    self.layer3_lstm = keras.layers.CuDNNLSTM(decoder_hidden_dim,
                                              return_sequences=True,
                                              return_state=True)
  
    # LAYER 4 = Concatenate lstm output with attention ######################
    self.layer4_concatenate = keras.layers.concatenate

    # LAYER 5 = Expand dims #################################################
    self.layer5_expand_dims = keras.backend.expand_dims 

    # LAYER 6 = Concatenate encoder output with attention ###################
    self.layer6_concatenate = keras.layers.concatenate 

    # LAYER 7 = LSTM (outer) ################################################
    # self.layer7_lstm = keras.layers.LSTM(decoder_hidden_dim, 
    #                                      return_sequences=True,
    #                                      return_state=True, 
    #                                      dropout=0.2,
    #                                      recurrent_dropout=0.2)
    self.layer7_lstm = keras.layers.CuDNNLSTM(decoder_hidden_dim,
                                              return_sequences=True,
                                              return_state=True)
      
  def call(self, inputs, val_mode=False, dropout=False, train_mode="full"):
    # Train or validation mode ##############################################
    if val_mode:
      logging.debug("MODEL DRAKE NESTED CALL - Train mode")
    else:
      logging.debug("MODEL DRAKE NESTED CALL - Validation mode")

    # STEP 0: Process Inputs ################################################
    # Input               | Encoder input       | batch_size=None x         #
    #                     |                     | feature_map_wxh=None(64) x#
    #                     |                     | image_embedding_dim=      #
    #                     |                     | image_embedding_dim       # 
    #                     | Token input         | batch_size = None x       #
    #                     |                     | token_seq_len =  x        #
    #                     |                     | token_seq_len             #
    #_____________________|_____________________|___________________________#
    input_image = inputs[0]
    input_tokens = inputs[1]
    self.batch_size = input_tokens.shape[0]
    batch_token_seq_len = input_tokens.shape[1]

    # Logging, Debug & Assert 
    logging.debug("MODEL DRAKE NESTED CALL - Step 0 - Process inputs - "
                  "batch_size {}".format(self.batch_size))
    logging.debug("MODEL DRAKE NESTED CALL - Step 0 - Process inputs - "
                  "input_image shape {}"
                  .format(K.int_shape(input_image)))
    logging.debug("MODEL DRAKE NESTED CALL - Step 0 - Process inputs - "
                  "input_tokens shape {}"
                  .format(K.int_shape(input_tokens)))
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

    # STEP 1: Reset Decoder Hidden State ####################################
    # Zeroes              | Initial decoder     | batch_size=None x         #
    #                     | hidden state        | decoder_hidden_dim=       #
    #                     |                     | decoder_hidden_dim        #
    #                     | Initial outer       | batch_size=None x         #
    #                     | decoder hidden      | decoder_hidden_dim=       #
    #                     | state               | decoder_hidden_dim        #
    #_____________________|_____________________|___________________________#
    decoder_hidden_state = \
      keras.backend.zeros(shape=(self.batch_size, self.decoder_hidden_dim))
    outer_decoder_hidden_state = \
      keras.backend.zeros(shape=(self.batch_size, self.decoder_hidden_dim))
    
    # Logging, Debug & Assert
    logging.debug("MODEL DRAKE NESTED CALL - Step 1 - Reset decoder hidden "
                  "state - decoder_hidden_state shape {}"
                 .format(K.int_shape(decoder_hidden_state)))
    logging.debug("MODEL DRAKE NESTED CALL - Step 1 - Reset outer decoder "
                  "hidden state - decoder_hidden_state shape {}"
                 .format(K.int_shape(outer_decoder_hidden_state)))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(decoder_hidden_state), 
      (self.batch_size, self.decoder_hidden_dim))
    tf.compat.v1.debugging.assert_equal(
      K.int_shape(outer_decoder_hidden_state), 
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
    logging.debug("MODEL DRAKE NESTED CALL - Step 2 - Image encoding dense"
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
    logging.debug("MODEL DRAKE NESTED CALL - Step 3 - Token embeddings - "
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
    logging.debug("MODEL DRAKE NESTED CALL - Step 4 - Decoder inputs - "
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
      # Nested mode trains the subnet on only the first five characters 
      # Note this approach assumes embedded padding. 
      if train_mode == "nested" and i >= 6: 
        logging.debug("MODEL DRAKE NESTED CALL - Step 5 - Skip loop if "
                      "in nested mode and index is > 6")
        continue;
      # STEP 5.1: Outer attention ###########################################
      # Summed weights    | Outer context       | batch_size=None x         # 
      #                   | vector              | decoder_hidden_dim=       #
      #                   |                     | decoder_hidden_dim        #  
      # __________________|_____________________|___________________________#
      outer_context_vector, outer_attention_weights = \
        self.model5_attention([input_image_features, 
                               outer_decoder_hidden_state],
                              dropout=dropout)

      # Logging, Debug & Assert 
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.1 - Outer attention - "
                    "Context vector shape {}"
                    .format(K.int_shape(outer_context_vector)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(outer_context_vector),
        (self.batch_size, self.decoder_hidden_dim))

      # STEP 5.2. Outer LSTM Input ##########################################
      # Expand +          | Outer LSTM input    | batch_size=None x         # 
      # Concatenate       |                     | token_seq_len=1 x         #
      #                   |                     | lstm_input_dim=           #  
      #                   |                     | decoder_hidden_dim +      #
      #                   |                     | token_embedding_dim       #
      # __________________|_____________________|___________________________#
      outer_context_vector_expanded = \
        self.layer5_expand_dims(outer_context_vector, 1)

      # Logging, Debug & Assert 
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.2 - Expand context "
                    "vector - context_vector_expanded shape {}"
                    .format(K.int_shape(outer_context_vector_expanded)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(outer_context_vector_expanded),
        (self.batch_size, 1, self.decoder_hidden_dim)) 

      outer_lstm_input = \
        self.layer6_concatenate([outer_context_vector_expanded,
                                 decoder_token_input],
                                axis=-1)

      # Logging, Debug & Assert  
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.2 - Concat context"
                    "vector and token embedding - lstm_input shape {}"
                    .format(K.int_shape(outer_lstm_input)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(outer_lstm_input),
        (self.batch_size, 1, 
         self.token_embedding_dim + self.decoder_hidden_dim))

      # STEP 5.3: Outer LSTM ################################################
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
      # outer_lstm_output, outer_decoder_hidden_state, \
      #   outer_decoder_cell_state = \
      #   self.layer7_lstm(outer_lstm_input, training=dropout) 
      outer_lstm_output, outer_decoder_hidden_state, \
        outer_decoder_cell_state = \
        self.layer7_lstm(outer_lstm_input) 

      # Logging, Debug & Assert
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.3 - LSTM output - "
                    "lstm_output shape {}"
                    .format(K.int_shape(outer_lstm_output)))
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.3 - LSTM output - "
                    "decoder_hidden_state shape {}"
                    .format(K.int_shape(outer_decoder_hidden_state)))
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.3 - LSTM output - "
                    "decoder_cell_state shape {}"
                    .format(K.int_shape(outer_decoder_cell_state)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(outer_lstm_output),
        (self.batch_size, 1, self.decoder_hidden_dim))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(outer_decoder_hidden_state),
        (self.batch_size, self.decoder_hidden_dim))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(outer_decoder_cell_state),
        (self.batch_size, self.decoder_hidden_dim))    
      
      # STEP 5.4: Inner attention ###########################################
      # Summed weights    | Context vector      | batch_size=None x         # 
      #                   |                     | decoder_hidden_dim=       #
      #                   |                     | decoder_hidden_dim        #  
      # __________________|_____________________|___________________________#
      # If in full or outer training mode and reach the end of a sub-sequence
      # feed in outer attention output rather than lstm output into 
      # inner attention
      if train_mode != "nested" and (i - 1) % 5 == 0 and i != 1:
        decoder_hidden_state = outer_context_vector

      context_vector, attention_weights = \
        self.model3_attention([input_image_features, decoder_hidden_state],
                              dropout=dropout)
      
      # Logging, Debug & Assert 
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.4 - Inner attention - "
                    "Context vector shape {}"
                    .format(K.int_shape(context_vector)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(context_vector),
        (self.batch_size, self.decoder_hidden_dim))
      
      # STEP 5.5: LSTM Input ################################################
      # Expand +          | LSTM input          | batch_size=None x         # 
      # Concatenate       |                     | token_seq_len=1 x         #
      #                   |                     | lstm_input_dim=           #  
      #                   |                     | decoder_hidden_dim +      #
      #                   |                     | token_embedding_dim       #
      # __________________|_____________________|___________________________#
      context_vector_expanded = self.layer1_expand_dims(context_vector, 1)

      # Logging, Debug & Assert 
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.5 - Expand context "
                    "vector - context_vector_expanded shape {}"
                    .format(K.int_shape(context_vector_expanded)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(context_vector_expanded),
        (self.batch_size, 1, self.decoder_hidden_dim)) 

      lstm_input = self.layer2_concatenate([context_vector_expanded,
                                            decoder_token_input],
                                           axis=-1)

      # Logging, Debug & Assert  
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.5 - Concat context"
                    "vector and token embedding - lstm_input shape {}"
                    .format(K.int_shape(lstm_input)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(lstm_input),
        (self.batch_size, 1, 
         self.token_embedding_dim + self.decoder_hidden_dim))      

      # STEP 5.6: LSTM ######################################################
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
      # lstm_output, decoder_hidden_state, decoder_cell_state = \
      #   self.layer3_lstm(lstm_input, training=dropout)  
      lstm_output, decoder_hidden_state, decoder_cell_state = \
        self.layer3_lstm(lstm_input) 
      
      # Logging, Debug & Assert
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.6 - LSTM output - "
                    "lstm_output shape {}"
                    .format(K.int_shape(lstm_output)))
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.6 - LSTM output - "
                    "decoder_hidden_state shape {}"
                    .format(K.int_shape(decoder_hidden_state)))
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.6 - LSTM output - "
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
      
      # STEP 5.7: MLP #######################################################
      # Dense             | Predicted token     | batch_size=None x         #
      #                   |                     | token_vocab_size x        #
      #                   |                     | token_vocab_size          #
      #___________________|_____________________|___________________________#
      mlp_input = self.layer4_concatenate([context_vector_expanded,
                                           lstm_output],
                                           axis=-1)

      single_token_prediction = self.model4_mlp([mlp_input],
                                                dropout=dropout)

      # Logging. Debug & Assert
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.7 - MLP output - "
                    "single_token_prediction shape {}"
                    .format(K.int_shape(single_token_prediction)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(single_token_prediction),
        (self.batch_size, self.token_vocab_size))

      # STEP 5.8: Calculate loss ############################################
      # Loss              | Single token loss   | int                       #
      #___________________|_____________________|___________________________#
      batch_loss += masked_ce_loss_fn(target=input_tokens[:, i],
                                      prediction=single_token_prediction,
                                      batch_size=self.batch_size,
                                      token_vocab_size=self.token_vocab_size)

      # Logging, Debug & Assert
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.8 - "
                    "Single prediction loss {}"
                    .format(batch_loss))

      # STEP 5.9 Update decoder input #######################################
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
      logging.debug("MODEL DRAKE NESTED CALL - Step 5.9 - Update decoder "
                    " input - decoder_token_input shape {}"
                    .format(K.int_shape(decoder_token_input)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(decoder_token_input),
        (self.batch_size, 1, self.token_embedding_dim))
    
    # STEP 6: Calculate levenstein distance 
    if val_mode:
      stack_predictions = tf.stack(list_predictions, axis=1)
      stack_predictions_len = stack_predictions.shape[1]

      # Logging, Debug & Assert
      logging.debug("MODEL DRAKE NESTED CALL - Step 6 - Stack predictions "
                    "shape {}"
                    .format(K.int_shape(stack_predictions)))
      tf.compat.v1.debugging.assert_equal(
        K.int_shape(stack_predictions),
        (self.batch_size, stack_predictions_len))

      batch_mean_edit_distance = \
        edit_distance_metric(
          target=input_tokens[:, 1:stack_predictions_len +1],
          prediction=stack_predictions,
          predictions_file=self.predictions_file)
      
    # STEP 7: Return word sequence batch loss ###############################
    return batch_loss, batch_mean_edit_distance

  def get_outer_trainable_variables(self):
    outer_trainable_variables = \
      self.model5_attention.trainable_variables + \
      self.layer7_lstm.trainable_variables
    return outer_trainable_variables

  def get_model_name(self):
    return "drake_nested"

  def set_predictions_file(self, predictions_file):
    self.predictions_file = predictions_file