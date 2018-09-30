import tensorflow as tf
import Data_processor
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

    def __init__(self, isEvaluate=False):
        if isEvaluate:
            self.dataset = Evaluate_Data_Processor.Data_holder()
        else:
            self.dataset = Data_processor.Data_holder()

        plength, qlength = self.dataset.set_batch()

        self.test_Batch_Size = 5
        self.Batch_Size = 100

        self.P_Length = 850
        self.Q_Length = 85
        self.Word_Embedding_Dimension = 100
        self.Contextual_Embedding_Dimension = 150

        self.Y_sta = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.Y_sto = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.X_P = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Q = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

        self.cell_Q_Enc_fw = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        self.cell_Q_Enc_bw = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)

        self.cell_P_Enc_fw = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)
        self.cell_P_Enc_bw = tf.nn.rnn_cell.BasicLSTMCell(self.Contextual_Embedding_Dimension)

        self.cell1_Modelling_fw = tf.nn.rnn_cell.BasicLSTMCell(150)
        self.cell1_Modelling_bw = tf.nn.rnn_cell.BasicLSTMCell(150)

        self.cell2_Modelling_fw = tf.nn.rnn_cell.BasicLSTMCell(150)
        self.cell2_Modelling_bw = tf.nn.rnn_cell.BasicLSTMCell(150)

        self.cell3_Modelling_fw = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell3_Modelling_bw = tf.nn.rnn_cell.BasicLSTMCell(1)

        #attention
        self.Weight_Q_1 = self.weight_variable(
            shape=[1, self.Contextual_Embedding_Dimension * 2, self.Contextual_Embedding_Dimension * 2])
        self.Weight_Q_2 = self.weight_variable(
            shape=[1, self.Contextual_Embedding_Dimension * 2, self.Contextual_Embedding_Dimension * 2])
        self.Weight_P_1 = self.weight_variable(
            shape=[1, self.Contextual_Embedding_Dimension * 2, self.Contextual_Embedding_Dimension * 2])
        self.Weight_P_2 = self.weight_variable(
            shape=[1, self.Contextual_Embedding_Dimension * 2, self.Contextual_Embedding_Dimension * 2])

        self.Bias_Q_1 = self.bias_variable(shape=[self.Contextual_Embedding_Dimension * 2])
        self.Bias_Q_2 = self.bias_variable(shape=[self.Contextual_Embedding_Dimension * 2])
        self.Bias_P_1 = self.bias_variable(shape=[self.Contextual_Embedding_Dimension * 2])
        self.Bias_P_2 = self.bias_variable(shape=[self.Contextual_Embedding_Dimension * 2])

        #answer module
        self.Weight_Ans_Sta = self.weight_variable(shape=[1, 300, 1])
        self.Weight_Ans_Sto = self.weight_variable(shape=[1, 300, 1])

        self.Bias_Ans_Sta = self.bias_variable(shape=[1])
        self.Bias_Ans_Sto = self.bias_variable(shape=[1])

    def Model(self):
        tensor_batch_size = tf.shape(self.X_P)[0]

        with tf.variable_scope("Encoding_Q") as scope:
            output_Qr, encoding_Qr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Q_Enc_fw,
                                                                     cell_bw=self.cell_Q_Enc_bw,
                                                                     inputs=self.X_Q,
                                                                     sequence_length=self.seq_length(self.X_Q),
                                                                     dtype=tf.float32)
            output_Qr_fw, output_Qr_bw = output_Qr
            H_Q = tf.concat([output_Qr_fw, output_Qr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("Encoding_P") as scope:
            output_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_P_Enc_fw,
                                                                     cell_bw=self.cell_P_Enc_bw,
                                                                     inputs=self.X_P,
                                                                     sequence_length=self.seq_length(self.X_P),
                                                                     dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = output_Pr
            H_P = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("dynamic_memory_attention") as scope:
            Weight_P_1 = tf.tile(self.Weight_P_1, multiples=[tensor_batch_size, 1, 1], name='w1')
            Weight_P_2 = tf.tile(self.Weight_P_2, multiples=[tensor_batch_size, 1, 1], name='w2')

            Weight_Q_1 = tf.tile(self.Weight_Q_1, multiples=[tensor_batch_size, 1, 1], name='w3')
            Weight_Q_2 = tf.tile(self.Weight_Q_2, multiples=[tensor_batch_size, 1, 1], name='w4')

            C1 = tf.matmul(H_P, tf.transpose(tf.nn.bias_add(tf.matmul(H_Q, Weight_P_1), self.Bias_P_1), perm=[0, 2, 1])) # P X Q
            C2 = tf.matmul(tf.nn.bias_add(tf.matmul(H_P, Weight_P_2), self.Bias_P_2), tf.transpose(H_Q, perm=[0, 2, 1])) # P X Q
            ##########
            D = tf.matmul(H_P, tf.transpose(H_Q, perm=[0, 2, 1]))
            D_ = tf.matmul(H_Q, tf.transpose(H_P, perm=[0, 2, 1]))

            D1 = tf.matmul(D_, tf.nn.bias_add(tf.matmul(H_P, Weight_Q_1), self.Bias_Q_1))  # Q X D
            D2 = tf.matmul(D, tf.nn.bias_add(tf.matmul(H_Q, Weight_Q_2), self.Bias_Q_2))  # P X D

            B1 = tf.matmul(C1, D1) # P X D
            B2 = tf.matmul(C2, D1) # P X D

            M = tf.concat([H_P, D2, B1, B2], axis=2)
            M.set_shape(shape=[None, None, self.Contextual_Embedding_Dimension * 8])


        with tf.variable_scope("Modelling1") as scope:
            output_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell1_Modelling_fw,
                                                                     cell_bw=self.cell1_Modelling_bw,
                                                                     inputs=M,
                                                                     sequence_length=self.seq_length(M),
                                                                     dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = output_Pr
            A = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("answer_start") as scope:
            Weight_P_Sta = tf.tile(self.Weight_Ans_Sta, multiples=[tensor_batch_size, 1, 1], name='weight_start')

            output_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell2_Modelling_fw,
                                                                     cell_bw=self.cell2_Modelling_bw,
                                                                     inputs=A,
                                                                     sequence_length=self.seq_length(A),
                                                                     dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = output_Pr
            S = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)

            S_sta = tf.nn.bias_add(tf.matmul(S, Weight_P_Sta), self.Bias_Ans_Sta)
            prediction_sta = tf.nn.relu(tf.squeeze(S_sta, -1))

            scope.reuse_variables()

        with tf.variable_scope("answer_stop") as scope:
            Weight_P_Sto = tf.tile(self.Weight_Ans_Sto, multiples=[tensor_batch_size, 1, 1], name='weight_stop')

            output_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell2_Modelling_fw,
                                                                     cell_bw=self.cell2_Modelling_bw,
                                                                     inputs=S,
                                                                     sequence_length=self.seq_length(A),
                                                                     dtype=tf.float32)

            output_Pr_fw_, output_Pr_bw_ = output_Pr
            S_ = tf.concat([output_Pr_fw_, output_Pr_bw_], axis=2)

            S_sto = tf.nn.bias_add(tf.matmul(S_, Weight_P_Sto), self.Bias_Ans_Sto)

            prediction_sto = tf.nn.relu(tf.squeeze(S_sto, -1))
            scope.reuse_variables()

        return prediction_sta, prediction_sto
    def training_prediction_index(self, training_epoch, is_continue, is_Start = True):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            A_sta, A_sto = self.Model()

            Probability_sta = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_sta, logits=A_sta)
            Probability_sto = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_sto, logits=A_sto)

            loss = tf.reduce_mean(tf.add(Probability_sta, Probability_sto))

            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.0000000000001
            #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
            #                                           30, 0.96, staircase=True)

            train_step = tf.train.AdamOptimizer(0.00002).minimize(loss, global_step=global_step)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            self.dataset.Batch_Index = 0

            epo = 0
            while epo < training_epoch:
                epo += 1
                #self.dataset.Batch_Index = 0

                paragraphs, questions, start_indexes, stop_indexes, index_start, index_stop, _, _, _ = self.dataset.get_next_batch()

                training_feed_dict = {self.Y_sta: start_indexes, self.Y_sto: stop_indexes, self.X_P: paragraphs,
                                      self.X_Q: questions}

                _, loss_value, ep = sess.run([train_step, loss, global_step], feed_dict=training_feed_dict)
                print(self.dataset.Batch_Index, ',', ep, ':', loss_value)
                #print(sess.run(A_sta[0], feed_dict=training_feed_dict))
                #print(start_indexes[0])

                if epo % 200 == 0:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')
                    print('saved!')

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

        return 0

    def test(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

            A_sta, A_sto = self.Model()

            Probability_sta = tf.square(tf.subtract(self.Y_sta, A_sta))
            Probability_sto = tf.square(tf.subtract(self.Y_sto, A_sto))

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            case = self.Batch_Size * self.test_Batch_Size

            f = open('C:\\Users\\Administrator\\Desktop\\qadataset\\evaluate', 'w')
            f.write('{')

            while True:
                paragraphs, questions, IDs, check_continue = self.dataset.get_next_batch()
                batch_ = paragraphs.shape[0]

                training_feed_dict = {self.X_P: paragraphs, self.X_Q: questions}

                start_prop = sess.run(Probability_sta, feed_dict=training_feed_dict)
                stop_prop = sess.run(Probability_sto, feed_dict=training_feed_dict)

                for a in range(batch_):
                    max_v = -999
                    index = -1

                    max_v2 = -999
                    index2 = -1

                    for j in range(self.P_Length):
                        if max_v < start_prop[a, j, 0]:
                            max_v = start_prop[a, j, 0]
                            index = j

                    for j in range(self.P_Length):
                        if max_v2 < stop_prop[a, j, 0]:
                            max_v2 = stop_prop[a, j, 0]
                            index2 = j

                    base_index = self.Batch_Size * (self.dataset.batch_index - 1)
                    line = self.dataset.Paragraphs[base_index + a, index]

                    for i in range(index + 1, index2 + 1):
                        if self.dataset.Paragraphs[base_index + a, i] == ',':
                            line += self.dataset.Paragraphs[base_index + a, i]
                        else:
                            line += ' '
                            line += self.dataset.Paragraphs[base_index + a, i]

                    if a == self.Batch_Size - 1:
                        f.write('\"')
                        f.write(str(IDs[a]))
                        f.write('\": ')

                        f.write('\"')
                        f.write(line)
                        f.write('\"')
                    else:
                        f.write('\"')
                        f.write(str(IDs[a]))
                        f.write('\": ')

                        f.write('\"')
                        f.write(line)
                        f.write('\"')

                        f.write(', ')

                if not check_continue:
                    f.write('}')
                    break
                else:
                    f.write(', ')

        return 0

    def get_test_result(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            A_sta, A_sto = self.Model()
            A_sta_index = tf.arg_max(A_sta, dimension=1)
            A_sto_index = tf.arg_max(A_sto, dimension=1)

            acc_sta = 0
            acc_sto = 0

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            case = self.Batch_Size * self.test_Batch_Size

            f = open('C:\\Users\\Administrator\\Desktop\\qadataset\\evaluate', 'w')
            f.write('{')

            while True:
                self.dataset.Batch_Index = 0
                paragraphs, questions, start_index, stop_index, start_index_value, stop_index_value, IDs, isEnd, para_length = self.dataset.get_next_batch()

                training_feed_dict = {self.X_P: paragraphs, self.X_Q: questions}

                start_result = sess.run(A_sta, feed_dict=training_feed_dict)
                stop_result = sess.run(A_sto, feed_dict=training_feed_dict)

                for i in range(paragraphs.shape[0]):
                    max_v = -999
                    index = -1

                    max_v2 = -999
                    index2 = -1

                    for j in range(paragraphs.shape[1]):
                        if max_v < start_result[i, j]:
                            max_v = start_result[i, j]
                            index = j

                    for j in range(paragraphs.shape[1]):
                        if max_v2 < stop_result[i, j]:
                            max_v2 = stop_result[i, j]
                            index2 = j

                    if index == start_index_value[i]:
                        acc_sta += 1
                    if index2 == stop_index_value[i]:
                        acc_sto += 1

                    print(start_index_value[i], ':', index, '  ', stop_index_value[i], ':', index2)
                    #print(start_result[i])
                    #print(stop_result[i])
                    print()
                    input()


                print(acc_sta, ',', acc_sto, '/', self.dataset.Total_Batch_Size)
                input()
                if isEnd:
                    print(acc_sta, ',', acc_sto, '/', self.dataset.Total_Batch_Size)
                    break

        return 0