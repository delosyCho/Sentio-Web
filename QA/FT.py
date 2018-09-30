import tensorflow as tf
import Sentence_Data_Processor
import Evaluate_Data_Processor
import WIKI_QA_DataProcessor
import Combined_Data_Processor

import Sentence_Representation

class Feature_Translator:
    def get_variables_with_name(self, name, train_only=True, printable=False):
        """Get variable list by a given name scope.
        Examples
        ---------
        >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
        """
        print("  [*] geting variables with %s" % name)
        # tvar = tf.trainable_variables() if train_only else tf.all_variables()
        if train_only:
            t_vars = tf.trainable_variables()
        else:
            try:  # TF1.0
                t_vars = tf.global_variables()
            except:  # TF0.12
                t_vars = tf.all_variables()

        d_vars = [var for var in t_vars if name in var.name]
        if printable:
            for idx, v in enumerate(d_vars):
                print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
        return d_vars

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

    def __init__(self, length, output_size=762):
        self.SE = Sentence_Representation.Conv_Rep()
        self.Length = length

        #762, 379
        self.output_size = output_size
        self.dataset = Combined_Data_Processor.Model()

        self.Word_Embedding_Dimension = 10
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

    def Discriminator(self, dis_inp, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            h1 = tf.contrib.layers.fully_connected(
                inputs=dis_inp,
                num_outputs=512,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC",
                reuse=reuse
            )

            h2 = tf.contrib.layers.fully_connected(
                inputs=h1,
                num_outputs=256,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC1",
                reuse=reuse
            )

            h3 = tf.contrib.layers.fully_connected(
                inputs=h2,
                num_outputs=128,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC2",
                reuse=reuse
            )

            h4 = tf.contrib.layers.fully_connected(
                inputs=h3,
                num_outputs=32,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC3",
                reuse=reuse
            )

            h5 = tf.contrib.layers.fully_connected(
                inputs=h4,
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC4",
                reuse=reuse
            )

            scope.reuse_variables()

        return h5

    def Generator(self, gen_inp, reuse=False):
        with tf.variable_scope("generator", reuse=reuse) as scope:
            SE = Sentence_Representation.Conv_Rep()
            H_3, H = SE.Encoder_Conv(self.Length, gen_inp, reuse=reuse)

        return H_3, H

    def get_loss(self, kor_Embed, eng_Embed):
        G_3, G = self.Generator(gen_inp=kor_Embed)
        G_con = tf.concat([G_3, G], axis=1)
        G_con.set_shape((None, 1141))

        D = self.Discriminator(eng_Embed)
        D_ = self.Discriminator(G_con, reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=tf.ones_like(D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_, labels=tf.zeros_like(D_)))

        g_loss = tf.reduce_mean(tf.square(tf.add(D_, tf.ones_like(D_)))) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_, labels=tf.ones_like(D_)))

        d_loss = tf.add(d_loss_real, d_loss_fake)

        g_vars = self.get_variables_with_name('generator', True, True)
        d_vars = self.get_variables_with_name('discriminator', True, True)

        return g_loss, g_vars, d_loss, d_vars