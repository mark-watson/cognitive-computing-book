#! /usr/bin/env python

# Copyright 2016 by Mark Watson and Denny Britz.
#
# LICENSE: Apache 2
#
# Derived from an example by Denny Britz (Copyright 2016) (https://github.com/dennybritz/cnn-text-classification-tf),
# which was derived from an example in Yoon Kim's paper http://arxiv.org/abs/1408.5882
#
# This code runs under TensorFlow version 0.11 and above using Python 2.7
#
# the output data directory 'trained_models' must exist before a training run
#
## to train:
# ./main.py
#
## to test/predict:
# ./main.py -test"
## to view results and intermediate calculations in tensorboard:
# tensorboard --logdir=runs/ --port=8080


import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import os
import sys
import datetime
import re
from os import listdir
from os.path import isfile, join
import csv
from stemming.porter2 import stem

## if you run out of memory, keep decreasing the BATCH_SIZE until the training job runs:
BATCH_SIZE = 32
NUM_EPOCHS = 10

## flag to determining if both input training data and testing samples should be word-stemmed:
USE_STEMMING = True

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

# model hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "for character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "similar to 3grams, 4grams, 5grams")
tf.flags.DEFINE_integer("num_filters", 64, "filters")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# training parameters
tf.flags.DEFINE_integer("batch_size", BATCH_SIZE, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", NUM_EPOCHS, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 25, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

## global variables:
loss = None
accuracy = None
input_x = None
input_y = None
dropout_keep_prob = None

noise_words = {'the','s','i','a','an','it'} # add more if you want

def clean_text(string):
  string = string.lower()
  string = re.sub(r"[^a-z]", " ", string)
  tokens = string.split(' ')
  # use stemming (you might not want to do this):
  if USE_STEMMING:
    tokens = [stem(a) for a in tokens if not a in noise_words]
  else:
    tokens = [a for a in tokens if not a in noise_words]
  return ' '.join(tokens[:50]).replace("  "," ")  ## for development, just keep first 50 words

def prepare_input_data():
  training_files = [f for f in listdir('data') if isfile(join('data', f))]
  print("training files: {}".format(training_files))
  class_count = 0
  number_of_classes = len(training_files)

  examples = []
  classifications = []
  for tfile in training_files:
    print("processing file: {}".format(tfile))
    example = list(open(join('data', tfile), "r").readlines())
    example = [clean_text(s.strip()) for s in example if len(s) > 20]
    output_map = ([0] * class_count) + [1] + ([0] * ((number_of_classes - 1) - class_count))
    examples += example
    classifications += [[output_map for _ in example]]
    class_count += 1
  y = np.concatenate(classifications)
  return [examples, y]

def batch_iter(data, shuffle = True):
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int(len(data) / BATCH_SIZE) + 1
  for epoch in range(NUM_EPOCHS):
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[shuffle_indices]
    else:
      shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * BATCH_SIZE
      end_index = min((batch_num + 1) * BATCH_SIZE, data_size)
      yield shuffled_data[start_index:end_index]

def cnn_text_classifier(input_length, num_classes, vocabulary_size, embedding_size,
                        filter_sizes, num_filters, l2_reg_lambda=0.0):
  global loss, accuracy, input_x, input_y, dropout_keep_prob
  # Placeholders for input, output and dropout
  input_x = tf.placeholder(tf.int32, [None, input_length], name="input_x")
  input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
  dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
  l2_loss = tf.constant(0.0) # l2 regularization loss
  with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

  pooled_outputs = [] # convolution + maxpool layer for each filter
  for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
      # Convolution Layer
      filter_shape = [filter_size, embedding_size, 1, num_filters]
      W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
      b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
      conv = tf.nn.conv2d(embedded_chars_expanded,  W, strides=[1, 1, 1, 1],
                          padding="VALID", name="conv")
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
      pooled = tf.nn.max_pool(  # for downsampling (pooling) over the outputs
        h, ksize=[1, input_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1], padding='VALID', name="pool")
      pooled_outputs.append(pooled)

  num_filters_total = num_filters * len(filter_sizes)
  h_pool = tf.concat(pooled_outputs, 3)
  h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
  with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
  with tf.name_scope("output"):
    W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
  with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
  with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



if len(sys.argv) == 1 or sys.argv[1] != "-test":     # TRAINING:
  print("\n------ Starting a training run...")
  
  # Read in training data from directory ./data.
  # Each file represents an input class.
  x_text, y = prepare_input_data()
  max_document_length = max([len(x.split(" ")) for x in x_text])
  print("maximum document length = {}".format(max_document_length))
  vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
  x = np.array(list(vocab_processor.fit_transform(x_text)))
  np.random.seed(10)
  shuffle_indices = np.random.permutation(np.arange(len(y)))
  x_shuffled = x[shuffle_indices]
  y_shuffled = y[shuffle_indices]

  # create separate training and test sets:
  dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
  x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
  y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
  print("# unique words = {:d}".format(len(vocab_processor.vocabulary_)))
  
  with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      cnn = cnn_text_classifier(
        input_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocabulary_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)
      # create a training operation:
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(1e-3)
      grads_and_vars = optimizer.compute_gradients(loss)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

      grad_summaries = []
      for g, v in grads_and_vars:
        if g is not None:
          grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
          sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
          grad_summaries.append(grad_hist_summary)
          grad_summaries.append(sparsity_summary)
      grad_summaries_merged = tf.summary.merge(grad_summaries)
      timestamp = "0000001"  ## str(int(time.time()))
      out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
      loss_summary = tf.summary.scalar("loss", loss)
      acc_summary = tf.summary.scalar("accuracy", accuracy)
      train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
      train_summary_dir = os.path.join(out_dir, "summaries", "train")
      train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
      dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
      dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
      dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
      checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
      checkpoint_prefix = os.path.join(checkpoint_dir, "model")
      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
      saver = tf.train.Saver(tf.global_variables())
      vocab_processor.save(os.path.join(out_dir, "vocab"))
      sess.run(tf.global_variables_initializer())

      def train_step(x_batch, y_batch):
        global loss, accuracy
        feed_dict = {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 0.5}
        _, step, summaries, local_loss, local_accuracy = sess.run(
          [train_op, global_step, train_summary_op, loss, accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: training step {}, loss {:g}, accuracy {:g}".format(time_str, step, local_loss, local_accuracy))
        train_summary_writer.add_summary(summaries, step)

      def dev_step(x_batch, y_batch, writer=None):
        global loss, accuracy
        feed_dict = {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0}
        step, summaries, local_loss, local_accuracy = sess.run(
          [global_step, dev_summary_op, loss, accuracy],
          feed_dict)
        time1 = datetime.datetime.now().isoformat()
        print("{}: developer step {}, loss {:g}, acc {:g}".format(time1, step, local_loss, local_accuracy))
        if writer:
          writer.add_summary(summaries, step)

      batches = batch_iter(list(zip(x_train, y_train)))
      for batch in batches:
        x_batch, y_batch = zip(*batch)
        #print("** x_batch = {}".format(x_batch))
        #print("** y_batch = {}".format(y_batch))
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
          dev_step(x_dev, y_dev, writer=dev_summary_writer)
        if current_step % FLAGS.checkpoint_every == 0:
          ##path = saver.save(sess, "./trained_models/cnn_model", global_step=current_step)
          path = saver.save(sess, checkpoint_prefix, global_step=current_step)
          print("Saved model checkpoint to {}\n".format(path))
  print("Done training.")

if len(sys.argv) > 1 and sys.argv[1] == "-test":             ## TESTING:
  print("\n------ Starting a prediction/test run...")
  tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
  FLAGS = tf.flags.FLAGS
  FLAGS._parse_flags()
  #checkpoint_dir = "trained_models"
  # Test trained model

  ## classes: training files: ['chemistry2', 'computers', 'economics', 'music2']

  # Here is some test data. You will want to read in your own data to evaluate here:
  x_raw = ["isomers of amyl alcohol are known dimethyl ethyl carbinol. the chemical reaction in the chemistry lab was successful and chemical element was found in an oxidation state and chemistry is also concerned with the interactions between atoms",
           "computer programming languages include lisp, Pascal, Java. Algorithms and data structures is the study of commonly used computational methods",
           "economic thought on trade agremments and unemployment  economists advocate that economic analysis of growth and trade in goods and services compared to wages and employment microeconomics and macroeconomics",
           "baroque artistic style classic music is soothing. his work as a singer and songwriter"
           ]
  # if we stemmed the training data, then also stem the test text:
  #  [f for f in listdir('data') if isfile(join('data', f))]   a = [[float(x) for x in suba] for suba in a]
  x_raw = [clean_text(s) for s in x_raw]
  print(x_raw)
  
  y_test = [0,1,2,3]

  # Map data into vocabulary. First find a vocabulary file that defines a unique word index
  # for each unique word in the training text data:
  #vocab_path = ''
  #for r, d, f in os.walk("./runs"):
  #  for files in f:
  #    if files == "vocab":
  #      vocab_path = os.path.join(r, files)
  timestamp = "0000001"
  out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
  print("in test mode: out_dir = " + out_dir)
  checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
  print("in test mode: checkpoint_dir = " + checkpoint_dir)
  ##checkpoint_prefix = os.path.join(checkpoint_dir, "model")

  vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
  print("* vocab_path = |{}|".format(vocab_path))
  vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
  x_test = np.array(list(vocab_processor.transform(x_raw)))
  # print(x_test)

  checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
  print("* checkpoint_file = |{}|".format(checkpoint_file))
  graph = tf.Graph()
  with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
      saver.restore(sess, checkpoint_file)
      input_x = graph.get_operation_by_name("input_x").outputs[0]
      dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
      predictions = graph.get_operation_by_name("output/predictions").outputs[0]

      # just use for one epoch for predictions:
      NUM_EPOCHS = 1
      batches = batch_iter(list(x_test), shuffle = False)
      all_predictions = []
      for x_test_batch in batches:
        #print("! x_test_batch = {}".format(x_test_batch))
        batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        #print("! batch_predictions = {}".format(batch_predictions))
        all_predictions = np.concatenate([all_predictions, batch_predictions])
  if y_test is not None:
    print("! all_predictions = {}".format(all_predictions))
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

  # write predictions to spreadsheet file prediction.csv:
  predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
  out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
  print("Saving evaluation to {0}".format(out_path))
  with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
