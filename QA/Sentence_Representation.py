import tensorflow as tf
import numpy as np

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=self.momentum,
                                        updates_collections=None,
                                        epsilon=self.epsilon,
                                        scale=True,
                                        is_training=train,
                                        scope=self.name)


class Conv_Rep:

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.05, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

        return dot_products / (norm1 * norm2)

    def __init__(self, isEvaluate=False):
        self.test_Batch_Size = 5
        self.Batch_Size = 100

        self.P_Length = 850
        self.Q_Length = 85
        self.Word_Embedding_Dimension = 100
        self.Contextual_Embedding_Dimension = 200

        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        self.X_P = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Q = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

        self.cell_Enc_fw = tf.contrib.rnn.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        self.cell_Enc_fw2 = tf.contrib.rnn.BasicLSTMCell(self.Contextual_Embedding_Dimension)

        self.cell_Enc_bw = tf.contrib.rnn.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        self.cell_Enc_bw2 = tf.contrib.rnn.BasicLSTMCell(self.Contextual_Embedding_Dimension)

    def conv_2d(self, x, d, w, di, name, reuse=True, l2_reg=0.04):
        with tf.variable_scope(name) as scope:
            conv = tf.contrib.layers.conv2d(
                inputs=x,
                num_outputs=di,
                kernel_size=d,
                stride=1,
                padding="VALID",
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                reuse=reuse,
                trainable=True,
                scope=scope
            )
        return conv

    def Encoder_Conv(self, length, X, reuse=False):
        p_length = length

        with tf.variable_scope("Contextual_Encoding") as scope:
            context_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Enc_fw,
                                                                      cell_bw=self.cell_Enc_bw,
                                                                      inputs=X,
                                                                      sequence_length=self.seq_length(X),
                                                                      dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = context_Pr
            X_Pr = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)
            scope.reuse_variables()

        with tf.variable_scope("Encoding1") as scope:
            p_length -= 9

            H_P1 = self.conv_2d(x=tf.expand_dims(X_Pr, axis=-1), d=10, w=10, di=8, name='conv1',reuse=reuse)
            scope.reuse_variables()

        with tf.variable_scope("Encoding2") as scope:
            p_length -= 6

            H_P2 = self.conv_2d(x=H_P1, d=7, w=7, di=4, name='conv2',reuse=reuse)
            H_P2_res = tf.reshape(H_P2, shape=[tf.shape(H_P2)[0], tf.shape(H_P2)[1], 85 * 4])
            H_P2_ = tf.layers.average_pooling2d(tf.expand_dims(H_P2_res, axis=-1), pool_size=(p_length, 1), strides=1,
                                                padding="VALID", name="hp2")

            scope.reuse_variables()

        with tf.variable_scope("Encoding3") as scope:
            p_length -= 4

            H_P3 = self.conv_2d(x=H_P2, d=5, w=5, di=2, name='conv3',reuse=reuse)
            H_P3_res = tf.reshape(H_P3, shape=[tf.shape(H_P3)[0], tf.shape(H_P3)[1], 381 * 2])
            H_P3_ = tf.layers.average_pooling2d(tf.expand_dims(H_P3_res, axis=-1), pool_size=(p_length, 1), strides=1,
                                                padding="VALID", name="hp3")
            H_P3_ = tf.reshape(H_P3_, shape=[tf.shape(H_P3_)[0], tf.shape(H_P3_)[2]])

            scope.reuse_variables()
            # return H_P3_
        with tf.variable_scope("Encoding") as scope:
            p_length -= 2

            H_P = self.conv_2d(x=H_P3, d=3, w=3, di=1, name='conv',reuse=reuse)
            H_P = tf.squeeze(H_P, axis=-1)
            H_P_ = tf.layers.average_pooling2d(tf.expand_dims(H_P, axis=-1), pool_size=(p_length, 1), strides=1,
                                               padding="VALID", name="hp")
            H_P_ = tf.squeeze(H_P_, axis=-1)
            H_P_ = tf.reshape(H_P_, shape=[tf.shape(H_P_)[0], tf.shape(H_P_)[2]])
            scope.reuse_variables()

        return H_P3_, H_P_

    def check_para(self):
        inp = tf.placeholder(shape=[1, 70, 100], dtype=tf.float32)
        arr = np.zeros(shape=[1, 70, 100], dtype=np.float32)
        a = self.Encoder_Conv(70, inp)
        sess = tf.Session()

        sess.run(tf.initialize_all_variables())

        ab = sess.run(a, feed_dict={inp:arr})
        print(np.array(ab).shape)