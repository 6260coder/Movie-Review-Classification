
# coding: utf-8

import data_helpers
import tensorflow as tf

import Sequence_Classification_Model

# data preparation
pos_samples_in_indices, neg_samples_in_indices, token_to_index_vocab, vocab_in_embeddings = data_helpers.load_pickled_data()

pos_labels = [[0, 1]] * len(pos_samples_in_indices)
neg_labels = [[1, 0]] * len(neg_samples_in_indices)
samples_in_indices = pos_samples_in_indices + neg_samples_in_indices
labels = pos_labels + neg_labels
samples_in_indices, labels = data_helpers.shuffle_data_and_labels(samples_in_indices,
                                              labels)
dev_sample_percentage = 0.01

samples_train, samples_dev, labels_train, labels_dev = data_helpers.partition_data_and_labels(samples_in_indices, 
                                                    labels, 
                                                    dev_sample_percentage)

print(len(vocab_in_embeddings))
print(len(token_to_index_vocab))
print(samples_dev.shape)
print(samples_in_indices.shape)

sequences = tf.placeholder(dtype=tf.int32, 
                           shape=[None, samples_in_indices.shape[1]])
labels = tf.placeholder(dtype=tf.int32, 
                        shape=[None, labels.shape[1]])
embedding_vocab = tf.placeholder(dtype=tf.float32, 
                                 shape=(len(vocab_in_embeddings), len(vocab_in_embeddings[0])))

# model creation
def GRU_cell(size):
    rnn_cell = tf.contrib.rnn.GRUCell(size)
    return rnn_cell

params = {"rnn_cell":GRU_cell, 
          "rnn_cell_size":200, 
          "optimizer":tf.train.RMSPropOptimizer(0.002), 
          "gradient_clipping":0}

model = Sequence_Classification_Model.SequenceClassificationModel(sequences, 
                                                                  labels, 
                                                                  embedding_vocab, 
                                                                  params)


# training
batch_size = 100

num_epoches = 100
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoches):
        print("epoch {}:".format(epoch))
        batch_generator = data_helpers.batch_generator(samples_train, 
                                                       labels_train, 
                                                       batch_size)
        for current_batch_num,             batch_samples,             batch_labels in batch_generator:
            feed_dict = {model.sequences:batch_samples, 
                         model.labels:batch_labels, 
                         model.embedding_vocab:vocab_in_embeddings}
            _, loss, acc = sess.run([model.optimize, model.loss, model.accuracy], 
                                    feed_dict)
            if current_batch_num % 5 == 0:
                print("{} - loss: {}, accuracy: {}".format(current_batch_num, loss, acc))
        print("===================================================================")
        print("--Evaluating...")
        feed_dict = {model.sequences:samples_dev, 
                     model.labels:labels_dev, 
                     model.embedding_vocab:vocab_in_embeddings}
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict)
        print("loss: {}, accuracy: {}".format(loss, acc))