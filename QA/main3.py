import Sentence_Data_Processor
import tensorflow as tf


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

data = Sentence_Data_Processor.Data_holder()
data.set_batch()

batch_paragraph, batch_question, batch_length, batch_label, _, _ = data.get_next_batch()

inp = tf.placeholder(dtype=tf.float32, shape=[None, None, 100])

cell_rnn = tf.nn.rnn_cell.BasicLSTMCell(5)
cell_rnn2 = tf.nn.rnn_cell.BasicLSTMCell(3)

with tf.variable_scope("Encoding_Q") as scope:
    output_Qr, encoding_Qr = tf.nn.dynamic_rnn(cell=cell_rnn, inputs=inp, dtype=tf.float32)
    H_Q = output_Qr

    scope.reuse_variables()

with tf.variable_scope("Encoding_Q2") as scope:
    output_Qr, encoding_Qr2 = tf.nn.dynamic_rnn(cell=cell_rnn2, inputs=H_Q, dtype=tf.float32)
    H_Q2 = tf.concat([encoding_Qr2[0], encoding_Qr2[1]], axis=1)

    scope.reuse_variables()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

feed_dict = {inp: batch_paragraph}

result = sess.run(H_Q2, feed_dict=feed_dict)

print(result)

"""
print(result.shape)
for i in range(result.shape[1]):
    print(result[0][i])
    input()
"""
