
# coding: utf-8


import tensorflow as tf


class SequenceClassificationModel(object):
    def __init__(self, sequences, labels, embedding_vocab, params):
        self.sequences = sequences
        self.labels = labels
        self.params = params
        self.embedding_vocab = embedding_vocab
        self.length = self.length()
        self.scores = self.scores()
        self.loss = self.loss()
        self.accuracy = self.accuracy()
        self.optimize = self.optimize()
        
    def scores(self):
        self.embedded_sequences = tf.nn.embedding_lookup(self.embedding_vocab, self.sequences)

        outputs,_ = tf.nn.dynamic_rnn(self.params["rnn_cell"](self.params["rnn_cell_size"]), 
                                     self.embedded_sequences, 
                                     dtype=tf.float32, 
                                     sequence_length=self.length)
        
        last_relevant_outputs = self.last_relevant(outputs, self.length)
        
        num_classes = int(self.labels.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal(shape=[self.params["rnn_cell_size"], num_classes], 
                                                 stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        scores = tf.nn.xw_plus_b(last_relevant_outputs, weight, bias)
        return scores
        
    def loss(self):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, 
                                                                                       labels=self.labels))
        return cross_entropy_loss
        
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.labels, axis=1), 
                                       tf.argmax(self.scores, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy
        
    def optimize(self):
        gradients = self.params["optimizer"].compute_gradients(self.loss)
        if self.params["gradient_clipping"]:
            limit = self.params["gradient_clipping"]
            gradients = [(None, var)                         if (grad is None)                         else (tf.clip_by_value(grad, -limit, limit), var)                         for grad, var in gradinets]
        optimize = self.params["optimizer"].apply_gradients(gradients)
        return optimize
        
    # Calculates the actual length of each sequence before padding.
    def length(self):
        signed = tf.sign(tf.abs(self.sequences))
        length = tf.reduce_sum(signed, axis=1)
        length = tf.cast(length, tf.int32)
        return length
    
    # Gathers the last relevant output for each sequence. 
    # Flattens the outputs tensor to be of shape [-1, output_size] and retrieve outputs of interest
    # with carefully calculated indices into the first dimension.
    def last_relevant(self, outputs, length):
        batch_size = tf.shape(outputs)[0]
        max_time = tf.shape(outputs)[1]
        output_size = tf.shape(outputs)[2]
        indices = tf.range(0, batch_size) * max_time + (length - 1)
        flattened_ouputs = tf.reshape(outputs, [-1, output_size])
        last_relevant_outputs = tf.gather(flattened_ouputs, indices)
        return last_relevant_outputs