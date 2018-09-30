import tensorflow as tf
import Sentence_Data_Processor
import Evaluate_Data_Processor
import WIKI_QA_DataProcessor
import Combined_Data_Processor
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
        self.dataset = Combined_Data_Processor.Model()

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

        #self.cell_Q_Enc2 = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        #self.cell_P_Enc2 = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)

        #attention
        self.Generative_Weight = self.weight_variable(
            shape=[1, self.Contextual_Embedding_Dimension * 2])
        self.Generative_Bias = self.bias_variable(shape=[self.Contextual_Embedding_Dimension * 2])

        self.Output_Weight = self.weight_variable(shape=[2, 2])
        self.Output_Bias = self.bias_variable(shape=[2])

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

    def Model(self):
        p_length = 50
        q_length = 30

        with tf.variable_scope("Contextual_Encoding") as scope:
            context_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Enc_fw,
                                                                     cell_bw=self.cell_Enc_bw,
                                                                     inputs=self.X_P,
                                                                     sequence_length=self.seq_length(self.X_P),
                                                                     dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = context_Pr
            X_Pr = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)

            context_Qr, encoding_Qr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Enc_fw,
                                                                      cell_bw=self.cell_Enc_bw,
                                                                      inputs=self.X_Q,
                                                                      sequence_length=self.seq_length(self.X_Q),
                                                                      dtype=tf.float32)

            output_Qr_fw, output_Qr_bw = context_Qr
            X_Qr = tf.concat([output_Qr_fw, output_Qr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("Encoding1") as scope:
            q_length -= 9

            H_Q1 = self.conv_2d(x=tf.expand_dims(X_Qr, axis=-1), d=10, w=10, di=8, name='conv1', reuse=False)
            scope.reuse_variables()

        with tf.variable_scope("Encoding2") as scope:
            q_length -= 6

            H_Q2 = self.conv_2d(x=H_Q1, d=7, w=7, di=4, name='conv2', reuse=False)
            #(500, 35, 85, 4)
            H_Q2_res = tf.reshape(H_Q2, shape=[tf.shape(H_Q2)[0], tf.shape(H_Q2)[1], 385 * 4])
            H_Q2_ = tf.layers.average_pooling2d(tf.expand_dims(H_Q2_res, axis=-1), pool_size=(q_length, 1), strides=1,
                                               padding="VALID", name="hq2")
            H_Q2_ = tf.reshape(H_Q2_, shape=[tf.shape(H_Q2_)[0], tf.shape(H_Q2_)[2]])

            scope.reuse_variables()

        with tf.variable_scope("Encoding3") as scope:
            q_length -= 4

            H_Q3 = self.conv_2d(x=H_Q2, d=5, w=5, di=2, name='conv3', reuse=False)
            H_Q3_res = tf.reshape(H_Q3, shape=[tf.shape(H_Q3)[0], tf.shape(H_Q3)[1], 381 * 2])
            H_Q3_ = tf.layers.average_pooling2d(tf.expand_dims(H_Q3_res, axis=-1), pool_size=(q_length, 1), strides=1,
                                               padding="VALID", name="hq3")
            H_Q3_ = tf.reshape(H_Q3_, shape=[tf.shape(H_Q3_)[0], tf.shape(H_Q3_)[2]])

            scope.reuse_variables()

        with tf.variable_scope("Encoding") as scope:
            q_length -= 2

            H_Q = self.conv_2d(x=H_Q3, d=3, w=3, di=1, name='conv', reuse=False)
            H_Q_ = tf.layers.average_pooling2d(H_Q, pool_size=(q_length, 1), strides=1,
                                              padding="VALID", name="hq")
            H_Q_ = tf.squeeze(H_Q_, axis=-1)
            H_Q_ = tf.reshape(H_Q_, shape=[tf.shape(H_Q_)[0], tf.shape(H_Q_)[2]])
            scope.reuse_variables()

        with tf.variable_scope("Encoding1") as scope:
            p_length -= 9

            H_P1 = self.conv_2d(x=tf.expand_dims(X_Pr, axis=-1), d=10, w=10, di=8, name='conv1')
            scope.reuse_variables()

        with tf.variable_scope("Encoding2") as scope:
            p_length -= 6

            H_P2 = self.conv_2d(x=H_P1, d=7, w=7, di=4, name='conv2')
            H_P2_res = tf.reshape(H_P2, shape=[tf.shape(H_P2)[0], tf.shape(H_P2)[1], 385 * 4])
            H_P2_ = tf.layers.average_pooling2d(tf.expand_dims(H_P2_res, axis=-1), pool_size=(p_length, 1), strides=1,
                                               padding="VALID", name="hp2")
            H_P2_ = tf.reshape(H_P2_, shape=[tf.shape(H_P2_)[0], tf.shape(H_P2_)[2]])

            scope.reuse_variables()

        with tf.variable_scope("Encoding3") as scope:
            p_length -= 4

            H_P3 = self.conv_2d(x=H_P2, d=5, w=5, di=2, name='conv3')
            H_P3_res = tf.reshape(H_P3, shape=[tf.shape(H_P3)[0], tf.shape(H_P3)[1], 381 * 2])
            H_P3_ = tf.layers.average_pooling2d(tf.expand_dims(H_P3_res, axis=-1), pool_size=(p_length, 1), strides=1,
                                               padding="VALID", name="hp3")
            H_P3_ = tf.reshape(H_P3_, shape=[tf.shape(H_P3_)[0], tf.shape(H_P3_)[2]])

            scope.reuse_variables()
            #return H_P3_
        with tf.variable_scope("Encoding") as scope:
            p_length -= 2

            H_P = self.conv_2d(x=H_P3, d=3, w=3, di=1, name='conv')
            H_P = tf.squeeze(H_P, axis=-1)
            H_P_ = tf.layers.average_pooling2d(tf.expand_dims(H_P, axis=-1), pool_size=(p_length, 1), strides=1,
                                              padding="VALID", name="hp")
            H_P_ = tf.squeeze(H_P_, axis=-1)
            H_P_ = tf.reshape(H_P_, shape=[tf.shape(H_P_)[0], tf.shape(H_P_)[2]])
            scope.reuse_variables()
            #return H_P_
        with tf.variable_scope("output") as scope:
            #sim2 = self.cos_sim(H_P2_, H_Q2_)
            #sim2 = tf.reshape(sim2, shape=[tf.shape(sim2)[0], 1])
            #sim3 = self.cos_sim(H_P3_, H_Q3_)
            #sim3 = tf.reshape(sim3, shape=[tf.shape(sim3)[0], 1])
            sim = self.cos_sim(H_P_, H_Q_)
            sim = tf.reshape(sim, shape=[tf.shape(sim)[0], 1])


            sim_ = tf.concat([sim], axis=1)
            sim_ = tf.reshape(sim_, shape=[tf.shape(sim_)[0], 1])

            score = tf.contrib.layers.fully_connected(
                inputs=sim_,
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        return score

    def training_prediction_index(self, training_epoch, is_continue, is_Start=True):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        path = 'D:\\se_model_cnn\\Net.ckpf'

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, path)
                batch_paragraph, batch_question, batch_label2, batch_label, batch_dump= self.dataset.get_next_batch()

            score = self.Model()

            prop = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=score)
            loss = tf.reduce_mean(prop)

            Train_OP = tf.train.AdamOptimizer(0.0001).minimize(loss)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, path)

            self.dataset.Batch_Index = 0

            epo = 0
            while epo < training_epoch:
                epo += 1
                #self.dataset.Batch_Index = 0

                batch_paragraph, batch_question, batch_label2, batch_label, batch_dump = self.dataset.get_next_batch()
                training_feed_dict = {self.Y: batch_label, self.X_P: batch_paragraph, self.X_Q: batch_question}

                _, loss_value = sess.run([Train_OP, loss], feed_dict=training_feed_dict)
                print(epo, ',', ':', loss_value)
                #print(sess.run(A_sta[0], feed_dict=training_feed_dict))
                #print(start_indexes[0])

                if epo % 300 == 0:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, path)
                    print('saved!')

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

        return 0

    def check_para(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            score = self.Model()

            #prop = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=score)
            #loss = tf.reduce_mean(prop)

            #Train_OP = tf.train.AdamOptimizer(0.00005).minimize(loss)

            sess.run(tf.initialize_all_variables())

            batch_paragraph, batch_question, batch_label2, batch_label, _ = self.dataset.get_next_batch()
            training_feed_dict = {self.Y: batch_label, self.X_P: batch_paragraph, self.X_Q: batch_question}

            result = sess.run(score, feed_dict=training_feed_dict)
            print(',', ':', result.shape)

        return 0

    def check(self):
        with tf.Session() as sess:
            #saver = tf.train.Saver()
            #save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            batch_paragraph, batch_question, batch_label, _, _ = self.dataset.get_next_batch()
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
                batch_paragraph, batch_question, batch_label, line, q_line = self.dataset.get_test_batch()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                result = sess.run(score, feed_dict=training_feed_dict)

                print(result.shape)
                print(batch_label.shape)

                for i in range(result.shape[0]):
                    print((result[i][0] - result[i][1]), batch_label[i])

                #print(line)
                #print(q_line)
                if input() == '0':
                    for j in range(self.dataset.Sentence_s_e_Index[line][0], self.dataset.Sentence_s_e_Index[line][1]):
                        print(self.dataset.Paragraphs[j])
                    print('\n\n')
                    print(self.dataset.Questions[j - 1])

#        return 0

    def get_refined_data(self):
        with tf.Session() as sess:
            file_p = open('refined_data_paragraph', 'w', encoding='utf8')
            file_q = open('refined_data_question', 'w', encoding='utf8')
            file_l = open('refined_data_label', 'w', encoding='utf8')

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            score = self.Model()
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            isRunning = True
            while isRunning:
                #print()
                batch_paragraph, batch_question, batch_label, line, cont, batch_index = self.dataset.get_sequence_batch()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                print(batch_index)

                result = sess.run(score, feed_dict=training_feed_dict)

                idx = 0

                for i in range(batch_index - cont, batch_index):

                    minimum_value = 9999
                    minimum_index = -1
                    minimum_idx = -1

                    for j in range(self.dataset.Sentence_s_e_Index[i][0], self.dataset.Sentence_s_e_Index[i][1]):
                        if minimum_value > (result[idx][0] - result[idx][1]):
                            minimum_value = (result[idx][0] - result[idx][1])
                            minimum_index = batch_label[idx]
                            minimum_idx = j

                        idx += 1

                    if minimum_index[1] == 1:
                        for j in range(self.dataset.Sentence_s_e_Index[i][0], self.dataset.Sentence_s_e_Index[i][1]):

                            line_str = ''
                            for k in range(len(self.dataset.Paragraphs[j]) - 1):
                                line_str += self.dataset.Paragraphs[j][k] + '\t'
                            line_str += self.dataset.Paragraphs[j][len(self.dataset.Paragraphs[j]) - 1]
                            file_p.write(line_str)
                            file_p.write('\a')

                            line_str = ''
                            for k in range(len(self.dataset.Questions[j]) - 1):
                                line_str += self.dataset.Questions[j][k] + '\t'
                            line_str += self.dataset.Questions[j][len(self.dataset.Questions[j]) - 1]
                            file_q.write(line_str)
                            file_q.write('\a')

                            if minimum_idx == j:
                                file_l.write('1\a')
                            else:
                                file_l.write('0\a')

                        print('writing...')
                #print(cont)

                if cont != 80:
                    file_p.close()
                    file_q.close()
                    file_l.close()
                    isRunning = False
                    print('!!')

        return 0

    def get_test_data_result_(self):
        true_case = 0
        false_case = 0

        path = 'D:\\se_model_cnn\\Net.ckpf'

        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, path)

            score = self.Model()
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, path)

            for q in range(200 - 1):
                batch_paragraph, batch_question, batch_label2, batch_label, _ = self.dataset.get_test_Batch1()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                result = sess.run(score, feed_dict=training_feed_dict)

                min_value = 9999
                min_index = -1

                for i in range(result.shape[0]):
                    if result[i][0] - result[i][1] < min_value:
                        min_index = i
                        min_value = result[i][0] - result[i][1]

                if batch_label[min_index][1] == 1:
                    true_case += 1
                else:
                    false_case += 1

                print(true_case, '/', false_case)

            for q in range(5000 - 1):
                batch_paragraph, batch_question, batch_label2, batch_label, _ = self.dataset.get_test_Batch2()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                result = sess.run(score, feed_dict=training_feed_dict)

                min_value = 9999
                min_index = -1

                for i in range(result.shape[0]):
                    if result[i][0] - result[i][1] < min_value:
                        min_index = i
                        min_value = result[i][0] - result[i][1]

                if batch_label[min_index][1] == 1:
                    true_case += 1
                else:
                    false_case += 1

                print(true_case, '/', false_case)

        print(true_case, '/', false_case)

#        return 0

