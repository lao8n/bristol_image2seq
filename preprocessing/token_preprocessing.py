"""
Image2Seq Model with Attention Pipeline
Token pre-processing 

MODEL  : NA
INPUT  : 
OUTPUT : 
"""
#!/usr/bin/env python3
#############################################################################
# Imports                                                                   #
#############################################################################

# Standard imports ##########################################################
import csv
import logging
import numpy as np
from PIL import Image
import os 
from six import raise_from
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys 
import random

# Stop pycache
sys.dont_write_bytecode = True

# Keras and tensorflow imports ##############################################
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow.keras as keras

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, 
    os.path.join(os.path.dirname(__file__), '..', '..'))
  import image2seq
  __package__ = "image2seq"

#############################################################################
# Preprocessing helper functions                                            #
#############################################################################
def open_for_csv(path):
  """ 
  Open a file with flags suitable for csv.reader.

  This is different for python2 it means with mode 'rb', for python3 this 
  means 'r' with "universal newlines".
  """
  if sys.version_info[0] < 3:
      return open(os.getcwd() + path, 'rb')
  else:
      return open(os.getcwd() + path, 'r', newline='')

def caption2seq(caption_seq):
  caption_seq = caption_seq.numpy()
  matrix_seq = ''
  for c in caption_seq:
    if 3 <= c <= 12:
      matrix_seq = matrix_seq + str(c - 3)
    elif c == 13:
      matrix_seq = matrix_seq +  '-'
    elif c == 14:
      matrix_seq = matrix_seq + '/'
    elif c == 15:
      matrix_seq = matrix_seq +  ','
    else:
      matrix_seq = matrix_seq +  ''

  return matrix_seq

def process_caption(matrix_seq, skip_padding=True):
  """
  Converts matrix sequences of characters into digits between 1 and 16.

  Token sequence:
  0 = ' ', 1 = GO, 2 = EOS, 3-12, 0-9, 13= -, 14 = /, 15 = , 16 = n
  """
  GO = 1
  EOS = 2
  processed_seq = [GO]

  for c in matrix_seq:
    # space padding ' '
    if ord(c) == 32:
      if skip_padding:
        pass
      else:
        processed_seq.append(0)
    # numbers 0 - 9
    elif 47 < ord(c) < 58:
      processed_seq.append(ord(c) - 48 + 3)
    # minus sign -
    elif ord(c) == 45:
      processed_seq.append(13)
    # forward slash /
    elif ord(c) == 47:
      processed_seq.append(14)
    # comma ,
    elif ord(c) == 44:
      processed_seq.append(15)
    # newline n
    elif ord(c) == 110:
      processed_seq.append(16)
    # quotation mark "
    elif ord(c)  == 34:
      pass
    else:
      logging.info("Error: ascii value " + str(c) + " is not a valid.")
    
  processed_seq.append(EOS)
  processed_seq = np.array(processed_seq)

  return processed_seq

def process_detection(matrix_seq, skip_padding=True):
  """
  Converts matrix sequences of characters into digits between 1 and 16.

  Token sequence:
  0 = ' ', 1 = GO, 2 = EOS, 3-12, 0-9, 13= -, 14 = /, 15 = , 16 = n
  """
  GO = 1
  EOS = 2
  processed_seq = []

  for c in matrix_seq:
    # space padding ' '
    if ord(c) == 32:
      if skip_padding:
        pass
      else:
        processed_seq.append(0)
    # numbers 0 - 9
    elif 47 < ord(c) < 58:
      processed_seq.append(ord(c) - 48 + 3)
    # minus sign -
    elif ord(c) == 45:
      processed_seq.append(13)
    # forward slash /
    elif ord(c) == 47:
      processed_seq.append(14)
    # comma ,
    elif ord(c) == 44:
      pass
    # newline n
    elif ord(c) == 110:
      pass
    # quotation mark "
    elif ord(c)  == 34:
      pass
    else:
      logging.info("Error: ascii value " + str(c) + " is not a valid.")
  
  processed_seq = random.shuffle(processed_seq)

  processed_seq = np.array(processed_seq)

  return processed_seq

def process_parallel_captions(matrix_seq):
  """
  Converts matrix sequences of characters into digits between 1 and 16.

  Token sequence:
  0 = ' ', 1 = GO, 2 = EOS, 3-12 =  0-9, 13= -, 14 = /, 15 = , 16 = n
  17-25 = array location (row by col (1,1), (1,2), (1,3), (2,1) etc)
  """
  GO = 1
  EOS = 2
  processed_seq = []
  # in a 3x3 matrix there are 9 array locations
  # do not append GO, instead have teh location token determine go. 
  # for j in range(9):
  #   processed_seq.append([GO])
  for j in range(9):
    processed_seq.append([])
  j = 0
  start_flag = True
  for i, c in enumerate(matrix_seq):
    if j < 9:
      # 'GO' for each matrix location
      if start_flag:
        processed_seq[j].append(j + 17)
        start_flag=False
      # space padding ' '
      if ord(c) == 32 and start_flag==False:
        if processed_seq[j][-1] == EOS or processed_seq[j][-1] == 0:
          processed_seq[j].append(0)
        else:
          processed_seq[j].append(EOS)
      # numbers 0 - 9
      elif 47 < ord(c) < 58:
        processed_seq[j].append(ord(c) - 48 + 3)
      # minus sign -
      elif ord(c) == 45:
        processed_seq[j].append(13)
      # forward slash /
      elif ord(c) == 47:
        processed_seq[j].append(14)
      # comma ,
      elif ord(c) == 44:
        processed_seq[j].append(15)
      # newline n
      elif ord(c) == 110:
        processed_seq[j].append(16)
      # quotation mark "
      elif ord(c)  == 34:
        pass
      else:
        logging.info("Error: ascii value " + str(c) + " is not a valid.")
      # append to new sequence
      if (i + 1) % 5 == 0:
        j += 1
        start_flag=True
  
  processed_seq = np.array(processed_seq)

  return processed_seq

#############################################################################
# Preprocessing                                                             #
#############################################################################
def token_preprocessing(csv_data_file, batch_size=64, skip_padding=True, 
                        parallel_caption=False):
  # STEP 1: Parse CSV data ##################################################
  # Extract a list of image paths and matrix sequences
  list_image_paths = []
  list_matrix_seqs = []
  with open_for_csv(csv_data_file) as file:
    for line, row in enumerate(csv.reader(file, delimiter='_')):
      line += 1
      try:
        image_path, matrix_seq = row[:2]
      except ValueError:
        raise_from(ValueError('line {}: format should be \'image_path '
                              'matrix_seq\' '.format(line)), None)

      list_image_paths.append(image_path)
      if parallel_caption:
        list_matrix_seqs.append(
          process_parallel_captions(matrix_seq))
      else:
        list_matrix_seqs.append(process_caption(matrix_seq, 
                                                skip_padding=skip_padding))
      
  len_list_matrix_seqs = len(list_matrix_seqs)

  # Logging, Debug & Assert 
  logging.info("TOKEN PREPROCESSING - Step 1 - {} matrix sequences loaded"
              .format(len_list_matrix_seqs))
  tf.compat.v1.debugging.assert_greater_equal(len_list_matrix_seqs,
                                              batch_size)
  tf.compat.v1.debugging.assert_equal(len(list_image_paths), 
                                      len_list_matrix_seqs)

  return list_image_paths, list_matrix_seqs

def matrix_shape_preprocessing(csv_data_file, batch_size=64):
  # STEP 1. Parse CSV data ##################################################
  list_image_paths = []
  list_matrix_shapes = []
  with open_for_csv(csv_data_file) as file:
    for line, row in enumerate(csv.reader(file, delimiter=',')):
      line +=1
      try:
        image_path, _, _, _, num_rows, num_cols, _ = row[:]
      except ValueError:
        raise_from(ValueError('line {}: format should be \'file_location '
                              ',... ,... ,... , num_rows, num_cols, ...\' '
                              .format(line)), None)
      list_image_paths.append(image_path)
      list_matrix_shapes.append(process_caption(num_rows) + 
                                process_caption(num_cols))
  len_list_matrix_shapes = len(list_matrix_shapes)

  # Logging, Debug & Assert
  logging.info("MATRIX SHAPE PREPROCESSING - Step 1 - {} "
               .format(len_list_matrix_shapes))
  tf.compat.v1.debugging.assert_greater_equal(len_list_matrix_shapes, 
                                              batch_size)
  tf.compat.v1.debugging.assert_equal(len(list_image_paths),
                                      len_list_matrix_shapes)
  
  return list_image_paths, list_matrix_shapes

def detected_values_preprocessing(csv_data_file, batch_size=64):
  list_image_paths = []
  list_matrix_seqs = []
  with open_for_csv(csv_data_file) as file:
    for line, row in enumerate(csv.reader(file, delimiter='_')):
      line += 1
      try:
        image_path, matrix_seq = row[:2]
      except ValueError:
        raise_from(ValueError('line {}: format should be \'image_path '
                              'matrix_seq\' '.format(line)), None)

      list_image_paths.append(image_path)
      list_matrix_seqs.append(process_caption(matrix_seq, 
                                              skip_padding=False))
  len_list_matrix_seqs = len(list_matrix_seqs)

  # Logging, Debug & Assert 
  logging.info("TOKEN PREPROCESSING - Step 1 - {} matrix sequences loaded"
              .format(len_list_matrix_seqs))
  tf.compat.v1.debugging.assert_greater_equal(len_list_matrix_seqs,
                                              batch_size)
  tf.compat.v1.debugging.assert_equal(len(list_image_paths), 
                                      len_list_matrix_seqs)

  return list_image_paths, list_matrix_seqs

# _, processed = token_preprocessing(csv_data_file="/graphs/0813_retinanet_3x3_demo/cropped/temp.txt",
#                                    batch_size=10,
#                                    parallel_caption=True)
# for seq in processed:
#   print(seq)