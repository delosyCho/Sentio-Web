import tensorflow as tf
import Sentence_Data_Processor
import Evaluate_Data_Processor

class DMA_NET:

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
        if isEvaluate:
            self.dataset = Evaluate_Data_Processor.Data_holder()
        else:
            self.dataset = Sentence_Data_Processor.Data_holder()

        plength, qlength = self.dataset.set_batch()

        self.test_Batch_Size = 5
        self.Batch_Size = 100

        self.P_Length = 850
        self.Q_Length = 85
        self.Word_Embedding_Dimension = 100
        self.Contextual_Embedding_Dimension = 200

        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        self.X_P = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Q = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

        self.cell_Enc_fw = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        self.cell_Enc_fw2 = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)

        self.cell_Enc_bw = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        self.cell_Enc_bw2 = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)

        #self.cell_Q_Enc2 = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        #self.cell_P_Enc2 = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)

        #attention
        self.Generative_Weight = self.weight_variable(
            shape=[1, self.Contextual_Embedding_Dimension * 2])
        self.Generative_Bias = self.bias_variable(shape=[self.Contextual_Embedding_Dimension * 2])

        self.Output_Weight = self.weight_variable(shape=[2, 2])
        self.Output_Bias = self.bias_variable(shape=[2])

    def Model(self):
        with tf.variable_scope("Encoding_Q") as scope:
            H_Q, encoding_Qr = tf.nn.dynamic_rnn(cell=self.cell_Enc_fw, inputs=self.X_Q, dtype=tf.float32)

            scope.reuse_variables()

        with tf.variable_scope("Encoding_Q2") as scope:
            H_Q2, encoding_Qr2 = tf.nn.dynamic_rnn(cell=self.cell_Enc_fw2, inputs=H_Q, dtype=tf.float32)


            scope.reuse_variables()
        #return tf.reduce_mean(H_Q2, axis=1)
        with tf.variable_scope("Encoding_P") as scope:
            H_P, encoding_Pr = tf.nn.dynamic_rnn(cell=self.cell_Enc_fw, inputs=self.X_P, dtype=tf.float32)

            scope.reuse_variables()

        with tf.variable_scope("Encoding_P2") as scope:
            H_P2, encoding_Pr2 = tf.nn.dynamic_rnn(cell=self.cell_Enc_fw2, inputs=H_P, dtype=tf.float32)
            """
            H_P2 = tf.contrib.layers.fully_connected(
                inputs=H_P2,
                num_outputs=self.Contextual_Embedding_Dimension * 2,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )
            """
            scope.reuse_variables()
        #return H_P2#tf.reduce_mean(H_P2, axis=1)
        with tf.variable_scope("output") as scope:

            H_P = tf.layers.average_pooling2d(tf.expand_dims(H_P, axis=-1), pool_size=(70, 1), strides=1, padding="VALID", name="hp")
            H_Q = tf.layers.average_pooling2d(tf.expand_dims(H_Q, axis=-1), pool_size=(50, 1), strides=1, padding="VALID", name="hq")

            H_P2 = tf.layers.average_pooling2d(tf.expand_dims(H_P2, axis=-1), pool_size=(70, 1), strides=1, padding="VALID", name="hp2")
            H_Q2 = tf.layers.average_pooling2d(tf.expand_dims(H_Q2, axis=-1), pool_size=(50, 1), strides=1, padding="VALID", name="hq2")

            sim = tf.concat(
                [tf.expand_dims(self.cos_sim(H_P, H_Q), axis=-1), tf.expand_dims(self.cos_sim(H_P2, H_Q2), axis=-1)],
                axis=1)

            sim = tf.squeeze(sim, axis=-1)
            sim = tf.squeeze(sim, axis=-1)

            score = tf.contrib.layers.fully_connected(
                inputs=sim,
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

            return score

        return score

    def training_prediction_index(self, training_epoch, is_continue, is_Start = True):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')
            batch_paragraph, batch_question, batch_length, batch_label, _, _ = self.dataset.get_next_batch()

            score = self.Model()

            prop = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=score)
            loss = tf.reduce_mean(prop)

            Train_OP = tf.train.AdamOptimizer(0.00002).minimize(loss)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            self.dataset.Batch_Index = 0

            epo = 0
            while epo < training_epoch:
                epo += 1
                #self.dataset.Batch_Index = 0

                batch_paragraph, batch_question, batch_length, batch_label, _, _ = self.dataset.get_next_batch()
                training_feed_dict = {self.Y: batch_label, self.X_P: batch_paragraph, self.X_Q: batch_question}

                _, loss_value = sess.run([Train_OP, loss], feed_dict=training_feed_dict)
                print(epo, ',', ':', loss_value)
                #print(sess.run(A_sta[0], feed_dict=training_feed_dict))
                #print(start_indexes[0])

                if epo % 300 == 0:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')
                    print('saved!')

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

        return 0

    def check_para(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            score = self.Model()

            prop = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=score)
            loss = tf.reduce_mean(prop)

            Train_OP = tf.train.AdamOptimizer(0.00005).minimize(loss)

            sess.run(tf.initialize_all_variables())

            batch_paragraph, batch_question, batch_length, batch_label, _, _ = self.dataset.get_next_batch()
            training_feed_dict = {self.Y: batch_label, self.X_P: batch_paragraph, self.X_Q: batch_question}

            result = sess.run(score, feed_dict=training_feed_dict)
            print(',', ':', result.shape)

        return 0

    def check(self):
        with tf.Session() as sess:
            #saver = tf.train.Saver()
            #save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            batch_paragraph, batch_question, batch_label, _, _ = self.dataset.get_test_batch()
            score = self.Model()

            sess.run(tf.initialize_all_variables())

            #saver = tf.train.Saver()
            #save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

            print(sess.run(score, feed_dict=training_feed_dict))

        return 0

    def get_test_result(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            score = self.Model()
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            while True:
                batch_paragraph, batch_question, batch_label, line, q_line = self.dataset.get_next_batch()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                result = sess.run(score, feed_dict=training_feed_dict)

                print(result.shape)
                print(batch_label.shape)

                for i in range(10):
                    print(result[i], batch_label[i])

                #print(line)
                #print(q_line)
                input()


        return 0

    def get_test_result_(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            score = self.Model()
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            while True:
                batch_paragraph, batch_question, batch_label, line, q_line = self.dataset.get_test_batch()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                result = sess.run(score, feed_dict=training_feed_dict)
                print(result.shape)
                for i in range(result.shape[0]):
                    print(result[i][0:10])
                print(batch_label)
                print(line)
                print(q_line)
                input()


        return 0