import tensorflow as tf
import numpy
import Data_processor

class Improved_AoA_Reader:

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_k(self, x, k):
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')

    def __init__(self, is_Para = True):
        self.dataset = Data_processor.Data_holder()
        self.dataset.set_batch()
        #self.dataset.get_POS()

        self.question = 0
        self.paragraph = 0
        self.POS_Embeddings = 0
        self.POS_Q_Embeddings = 0
        self.start_index = 0
        self.stop_index = 0
        self.attention_Label = 0

        self.embedding_size = 50
        self.POS_Embedding_Size = 128
        self.batch = 500
        self.p_length = 125
        self.q_length = 30

        self.cell_Q_Enc_fw = tf.nn.rnn_cell.BasicLSTMCell(32)
        self.cell_Q_Enc_bw = tf.nn.rnn_cell.BasicLSTMCell(32)

        self.cell_P_Enc_fw = tf.nn.rnn_cell.BasicLSTMCell(32)
        self.cell_P_Enc_bw = tf.nn.rnn_cell.BasicLSTMCell(32)

        self.cell_POS_Enc_fw = tf.nn.rnn_cell.BasicLSTMCell(32)
        self.cell_POS_Enc_bw = tf.nn.rnn_cell.BasicLSTMCell(32)

        self.cell_output_Index_fw = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_output_Index_bw = tf.nn.rnn_cell.BasicLSTMCell(1)

        self.cell_modeling_fw = tf.nn.rnn_cell.BasicLSTMCell(64)
        self.cell_modeling_bw = tf.nn.rnn_cell.BasicLSTMCell(64)

        self.cell_modeling_fw_ = tf.nn.rnn_cell.BasicLSTMCell(32)
        self.cell_modeling_bw_ = tf.nn.rnn_cell.BasicLSTMCell(32)

        self.W_conv_2 = self.weight_variable([2, 1, 1, 4])
        self.W_conv_3 = self.weight_variable([3, 1, 1, 4])
        self.W_conv_4 = self.weight_variable([4, 1, 1, 4])
        self.W_conv_5 = self.weight_variable([5, 1, 1, 4])

        self.Weight_Paragraph = self.weight_variable(shape=[1, 64, 64])
        self.Weight_POS = self.weight_variable(shape=[1, 64, 64])

        self.Bias_Paragraph = self.bias_variable(shape=[64])
        self.Bias_POS = self.bias_variable(shape=[64])

        self.Weight_Dec_Input = self.weight_variable(shape=[1, 64 * 5, 128])
        self.Bias_Dec_Input = self.bias_variable(shape=[128])

        self.Weight_D_Out = self.weight_variable(shape=[1, 128, 1])
        self.Bias_D_Out = self.weight_variable(shape=[1])

        self.Weight_D_Out2 = self.weight_variable(shape=[1, 128, 1])
        self.Bias_D_Out2 = self.weight_variable(shape=[1])

        self.Weight_D_Out_Stop = self.weight_variable(shape=[1, 128, 1])
        self.Bias_D_Out_Stop = self.weight_variable(shape=[1])



        self.W_fc = self.weight_variable([1, 16, 1])
        self.b_fc = self.bias_variable([1])

        self.output_Start = None
        self.output_Stop = None

        self.Y_sta = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        self.Y_sto = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

        self.Word_Embedding_Dimension = 100
        self.x_p_holer = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.x_q_holer = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])


    def set_batch(self, q, p, sta, sto):
        self.question = q
        self.paragraph = p
        self.start_index = sta
        self.stop_index = sto

    def model_Index_Start_End(self):
        with tf.variable_scope("Encoding_Q") as scope:
            X_Q = self.x_q_holer
            X_P = self.x_p_holer

            output_Qr, encoding_Qr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Q_Enc_fw,
                                                                     cell_bw=self.cell_Q_Enc_bw,
                                                                     inputs=X_Q,
                                                                     sequence_length=self.seq_length(X_Q),
                                                                     dtype=tf.float32)
            output_Qr_fw, output_Qr_bw = output_Qr
            H_Q = tf.concat([output_Qr_fw, output_Qr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("Encoding_P") as scope:
            output_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_P_Enc_fw,
                                                                     cell_bw=self.cell_P_Enc_bw,
                                                                     inputs=X_P,
                                                                     sequence_length=self.seq_length(X_P),
                                                                     dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = output_Pr
            H_P = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("attention") as scope:
            H_Q_T = tf.transpose(H_Q, perm=[0, 2, 1])

            M_Vector = tf.matmul(H_P, H_Q_T)
            Alpha = tf.nn.softmax(M_Vector, dim=0)
            Beta = tf.matmul(Alpha, H_Q)

            scope.reuse_variables()

        with tf.variable_scope("attention_W") as scope:
            Weight_Paragraph = tf.tile(self.Weight_Paragraph, [self.batch, 1, 1])

            W_H_P = tf.matmul(H_P, Weight_Paragraph) + self.Bias_Paragraph
            W_Alpha = tf.matmul(W_H_P, H_Q_T)
            W_Beta = tf.matmul(W_Alpha, H_Q)

            scope.reuse_variables()

        with tf.variable_scope("attention_flow") as scope:
            C_Input = tf.concat([H_P, Beta, W_Beta], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("modeling") as scope:
            output_, encoding_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_modeling_fw,
                                                                 cell_bw=self.cell_modeling_bw,
                                                                 inputs=C_Input,
                                                                 sequence_length=self.seq_length(C_Input),
                                                                 dtype=tf.float32)

            output_fw, output_bw = output_
            output = tf.concat([output_fw, output_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("decoding_Start") as scope:
            Weight_D_Out = tf.tile(self.Weight_D_Out, [self.batch, 1, 1])

            Output = tf.matmul(output, Weight_D_Out) + self.Bias_D_Out
            Prediction_ = tf.nn.relu(Output)

            Prediction_Start = tf.reshape(Prediction_, shape=[self.batch, self.p_length])
        with tf.variable_scope("decoding_Stop") as scope:
            Weight_D_Out_Stop = tf.tile(self.Weight_D_Out_Stop, [self.batch, 1, 1])

            Output_Stop = tf.matmul(output, Weight_D_Out_Stop) + self.Bias_D_Out_Stop
            Prediction_Stop_ = tf.nn.relu(Output_Stop)

            Prediction_Stop = tf.reshape(Prediction_Stop_, shape=[self.batch, self.p_length])

        return Prediction_Start, Prediction_Stop

    def model_Index(self):
        with tf.variable_scope("Encoding_Q") as scope:
            X_Q = self.x_q_holer
            X_P = self.x_p_holer

            output_Qr, encoding_Qr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Q_Enc_fw,
                                                                     cell_bw=self.cell_Q_Enc_bw,
                                                                     inputs=X_Q,
                                                                     sequence_length=self.seq_length(X_Q),
                                                                     dtype=tf.float32)
            output_Qr_fw, output_Qr_bw = output_Qr
            H_Q = tf.concat([output_Qr_fw, output_Qr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("Encoding_P") as scope:
            output_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_P_Enc_fw,
                                                                     cell_bw=self.cell_P_Enc_bw,
                                                                     inputs=X_P,
                                                                     sequence_length=self.seq_length(X_P),
                                                                     dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = output_Pr
            H_P = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("attention") as scope:
            H_Q_T = tf.transpose(H_Q, perm=[0, 2, 1])

            M_Vector = tf.matmul(H_P, H_Q_T)
            Alpha = tf.nn.softmax(M_Vector, dim=0)
            Beta = tf.matmul(Alpha, H_Q)

            scope.reuse_variables()

        with tf.variable_scope("attention_W") as scope:
            Weight_Paragraph = tf.tile(self.Weight_Paragraph, [tf.shape(self.x_p_holer)[0], 1, 1])

            W_H_P = tf.matmul(H_P, Weight_Paragraph) + self.Bias_Paragraph
            W_Alpha = tf.matmul(W_H_P, H_Q_T)
            W_Beta = tf.matmul(W_Alpha, H_Q)

            scope.reuse_variables()

        with tf.variable_scope("attention_flow") as scope:
            C_Input = tf.concat([H_P, Beta, W_Beta], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("modeling") as scope:
            output_, encoding_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_modeling_fw,
                                                                 cell_bw=self.cell_modeling_bw,
                                                                 inputs=C_Input,
                                                                 sequence_length=self.seq_length(C_Input),
                                                                 dtype=tf.float32)

            output_fw, output_bw = output_
            output = tf.concat([output_fw, output_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("decoding") as scope:
            Weight_D_Out = tf.tile(self.Weight_D_Out, [tf.shape(self.x_p_holer)[0], 1, 1])

            Output = tf.matmul(output, Weight_D_Out) + self.Bias_D_Out
            Prediction = tf.nn.tanh(Output)

        with tf.variable_scope("decoding_") as scope:
            Weight_D_Out2 = tf.tile(self.Weight_D_Out2, [tf.shape(self.x_p_holer)[0], 1, 1])

            Output = tf.matmul(output, Weight_D_Out2) + self.Bias_D_Out2
            Prediction2 = tf.nn.tanh(Output)

        return Prediction, Prediction2

    def training_prediction_index(self, training_epoch, is_continue, is_Start = True):
        with tf.Session() as sess:
            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            output, output2 = self.model_Index()

            #Probability = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_sta, logits=output)
            #Probability2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_sto, logits=output2)

            Probability = tf.square(tf.subtract(self.Y_sta, output))
            Probability2 = tf.square(tf.subtract(self.Y_sto, output2))

            loss = tf.reduce_mean(Probability) + tf.reduce_mean(Probability2)

            train_step = tf.train.AdamOptimizer(0.05).minimize(loss)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            ep = 0

            while True:
                self.dataset.Batch_Index = 2445
                paragraphs, questions, start_indexes, stop_indexes, _, _, _, _, _ = self.dataset.get_next_batch()

                training_feed_dict = {self.Y_sta: start_indexes, self.Y_sto: stop_indexes, self.x_p_holer: paragraphs,
                                      self.x_q_holer: questions}
                print(self.dataset.Batch_Index, sess.run(loss, feed_dict=training_feed_dict))
                # print(self.attention_Label)

                _, l = sess.run([train_step, loss], feed_dict=training_feed_dict)
                #print(self.dataset.Batch_Index, l)

                saver = tf.train.Saver()
                save_path = saver.save(sess,
                                       'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')
                #print(self.dataset.whole_batch_index, sess.run(loss, feed_dict=training_feed_dict))
                #print(self.attention_Label)

                if ep % 30 == 0:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')
                    print('saved!')
                ep += 1
        return 0

    def get_test_result(self):
        with tf.Session() as sess:
            #saver = tf.train.Saver()
            #save_path = saver.restore(sess,
            #                          'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            A_sta, A_sto = self.model_Index()
            A_sta_index = tf.arg_max(A_sta, dimension=1)
            A_sto_index = tf.arg_max(A_sto, dimension=1)

            acc_sta = 0
            acc_sto = 0

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            f = open('C:\\Users\\Administrator\\Desktop\\qadataset\\evaluate', 'w')
            f.write('{')

            while True:
                self.dataset.Batch_Index = 0
                paragraphs, questions, start_index, stop_index, start_index_value, stop_index_value, IDs, isEnd, para_length = self.dataset.get_next_batch()

                training_feed_dict = {self.x_p_holer: paragraphs, self.x_q_holer: questions}

                start_result = sess.run(A_sta, feed_dict=training_feed_dict)
                stop_result = sess.run(A_sto, feed_dict=training_feed_dict)

                for i in range(paragraphs.shape[0]):
                    max_v = -999
                    index = -1

                    max_v2 = -999
                    index2 = -1

                    for j in range(paragraphs.shape[1]):
                        if max_v < start_result[i, j, 0]:
                            max_v = start_result[i, j, 0]
                            index = j

                    for j in range(paragraphs.shape[1]):
                        if max_v2 < stop_result[i, j, 0]:
                            max_v2 = stop_result[i, j, 0]
                            index2 = j

                    if index == start_index_value[i]:
                        acc_sta += 1
                    if index2 == stop_index_value[i]:
                        acc_sto += 1

                    print(index, index2)
                    print(start_index_value[i], stop_index_value[i])
                    print(start_result[i])
                    print(stop_result[i])
                    print()
                    input()

                print(acc_sta, ',', acc_sto, '/', self.dataset.Total_Batch_Size)
                input()
                if isEnd:
                    print(acc_sta, ',', acc_sto, '/', self.dataset.Total_Batch_Size)
                    break

        return 0

    def training_classification(self, training_epoch):
        with tf.Session() as sess:
            self.paragraph, self.question, self.start_index, self.stop_index, self.attention_Label = self.dataset.get_next_batch()

            Att_L = tf.placeholder(tf.float32, shape=[self.batch, 1], name='Attention_Label')

            S_Vector = self.model()

            with tf.variable_scope("classification") as scope:
                S_Vector_ = tf.expand_dims(S_Vector, axis=3)

                conv2 = self.conv2d(S_Vector_, self.W_conv_2)
                conv3 = self.conv2d(S_Vector_, self.W_conv_3)
                conv4 = self.conv2d(S_Vector_, self.W_conv_4)
                conv5 = self.conv2d(S_Vector_, self.W_conv_5)
                print("Conv", conv2, conv3, conv4, conv5)
                output2 = self.max_pool_k(conv2, self.P_Length - 1)
                output3 = self.max_pool_k(conv3, self.P_Length - 2)
                output4 = self.max_pool_k(conv4, self.P_Length - 3)
                output5 = self.max_pool_k(conv5, self.P_Length - 4)

                output2 = tf.reshape(output2, shape=[self.batch, -1])
                output3 = tf.reshape(output3, shape=[self.batch, -1])
                output4 = tf.reshape(output4, shape=[self.batch, -1])
                output5 = tf.reshape(output5, shape=[self.batch, -1])
                print("Shape Conv:", output2, output3, output4, output5)
                flat_output = tf.concat([output2, output3, output4, output5], axis=1)
                output = tf.nn.relu(tf.matmul(flat_output, self.W_fc) + self.b_fc)

                scope.reuse_variables()

            Probability_Attention = (output - Att_L) * (output - Att_L)
            loss = tf.reduce_sum(Probability_Attention)

            train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

            sess.run(tf.initialize_all_variables())

            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index, self.attention_Label = self.dataset.get_next_batch()

                sess.run(train_step, feed_dict={self.x_q_holer: self.question, self.x_p_holer: self.paragraph,
                                                Att_L: self.attention_Label})

                if self.dataset.batch_index == self.batch:
                    print(self.dataset.whole_batch_index, sess.run(loss, feed_dict={self.x_q_holer: self.question,
                                                            self.x_p_holer: self.paragraph, Att_L: self.attention_Label}))
                    #print(self.attention_Label)
            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/aoa_Reader.ckpf')

        return 0

    def training_classification_continue(self, training_epoch):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/aoa_Reader.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index, self.attention_Label = self.dataset.get_next_batch()

            Att_L = tf.placeholder(tf.float32, shape=[self.batch, 1], name='Attention_Label')

            S_Vector = self.model()

            with tf.variable_scope("classification") as scope:
                S_Vector_ = tf.expand_dims(S_Vector, axis=3)

                conv2 = self.conv2d(S_Vector_, self.W_conv_2)
                conv3 = self.conv2d(S_Vector_, self.W_conv_3)
                conv4 = self.conv2d(S_Vector_, self.W_conv_4)
                conv5 = self.conv2d(S_Vector_, self.W_conv_5)
                print("Conv", conv2, conv3, conv4, conv5)
                output2 = self.max_pool_k(conv2, self.P_Length - 1)
                output3 = self.max_pool_k(conv3, self.P_Length - 2)
                output4 = self.max_pool_k(conv4, self.P_Length - 3)
                output5 = self.max_pool_k(conv5, self.P_Length - 4)

                output2 = tf.reshape(output2, shape=[self.batch, -1])
                output3 = tf.reshape(output3, shape=[self.batch, -1])
                output4 = tf.reshape(output4, shape=[self.batch, -1])
                output5 = tf.reshape(output5, shape=[self.batch, -1])
                print("Shape Conv:", output2, output3, output4, output5)
                flat_output = tf.concat([output2, output3, output4, output5], axis=1)
                output = tf.nn.relu(tf.matmul(flat_output, self.W_fc) + self.b_fc)

                scope.reuse_variables()

            Probability_Attention = (output - Att_L) * (output - Att_L)
            loss = tf.reduce_sum(Probability_Attention)

            train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/aoa_Reader.ckpf')

            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index, self.attention_Label = self.dataset.get_next_batch()

                sess.run(train_step, feed_dict={self.x_q_holer: self.question, self.x_p_holer: self.paragraph,
                                                Att_L: self.attention_Label})

                if self.dataset.batch_index == self.batch:
                    print(self.dataset.whole_batch_index, sess.run(loss, feed_dict={self.x_q_holer: self.question,
                                                            self.x_p_holer: self.paragraph, Att_L: self.attention_Label}))
                    #print(self.attention_Label)
            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/aoa_Reader.ckpf')

        return 0

    def test_classification(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/aoa_Reader.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index, self.attention_Label = self.dataset.get_next_batch()

            Att_L = tf.placeholder(tf.float32, shape=[self.batch, 1], name='Attention_Label')

            S_Vector = self.model()

            with tf.variable_scope("classification") as scope:
                S_Vector_ = tf.expand_dims(S_Vector, axis=3)

                conv2 = self.conv2d(S_Vector_, self.W_conv_2)
                conv3 = self.conv2d(S_Vector_, self.W_conv_3)
                conv4 = self.conv2d(S_Vector_, self.W_conv_4)
                conv5 = self.conv2d(S_Vector_, self.W_conv_5)
                print("Conv", conv2, conv3, conv4, conv5)
                output2 = self.max_pool_k(conv2, self.P_Length - 1)
                output3 = self.max_pool_k(conv3, self.P_Length - 2)
                output4 = self.max_pool_k(conv4, self.P_Length - 3)
                output5 = self.max_pool_k(conv5, self.P_Length - 4)

                output2 = tf.reshape(output2, shape=[self.batch, -1])
                output3 = tf.reshape(output3, shape=[self.batch, -1])
                output4 = tf.reshape(output4, shape=[self.batch, -1])
                output5 = tf.reshape(output5, shape=[self.batch, -1])
                print("Shape Conv:", output2, output3, output4, output5)
                flat_output = tf.concat([output2, output3, output4, output5], axis=1)
                output = tf.nn.relu(tf.matmul(flat_output, self.W_fc) + self.b_fc)

                scope.reuse_variables()

            Probability_Attention = (output - Att_L) * (output - Att_L)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/aoa_Reader.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index, self.attention_Label, self.POS_Embeddings\
                = self.dataset.get_test_batch()

            test_result = sess.run(Probability_Attention, feed_dict={self.x_q_holer: self.question, self.x_p_holer: self.paragraph,
                                            Att_L: self.attention_Label})

            output_result = sess.run(output,
                                   feed_dict={self.x_q_holer: self.question, self.x_p_holer: self.paragraph,
                                              Att_L: self.attention_Label})

            check = 0
            check_ = 0
            check2 = 0
            check2_ = 0

            for i in range(self.batch):
                if self.attention_Label[i] == 1:
                    if output_result[i] > 0.5:
                        #print("Right: ", self.dataset.pa,self.question)
                        check = check + 1
                    else:
                        check_ = check_ + 1
                        #print("No attention: ", self.paragraph, self.question)
                else:
                    if output_result[i] < 0.5:
                        check2 = check2 + 1
                        #print("Right: ", self.paragraph, self.question)
                    else:
                        check2_ = check2_ + 1
                        #print("Wrong Attention: ", self.paragraph, self.question)

            check3 = check + check2

            print(check3, "/", self.batch, " ", check, check2, check_, check2_)

        return 0

    def test_index(self, is_Start = True):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index, self.POS_Embeddings = self.dataset.get_next_batch()

            if is_Start:
                label = self.start_index
            else:
                label = self.stop_index

            tensor_index = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Attention_Label')

            tensor_Label = tf.one_hot(tensor_index, self.p_length, 1, 0)
            tensor_Label_ = tf.cast(tensor_Label, tf.float32)
            tensor_Label_ = tf.reshape(tensor_Label_, [self.batch, self.p_length])

            output = self.model_Index()
            output = tf.reshape(output, shape=[self.batch, self.p_length])

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            training_feed_dict = {tensor_index: label, self.x_pos_holder: self.POS_Embeddings,
                                  self.x_p_holer: self.paragraph, self.x_q_holer: self.question}

            if is_Start:
                label = self.start_index
            else:
                label = self.stop_index

            result_Prop = sess.run(output, feed_dict=training_feed_dict)

            for a in range(self.batch):
                max_v = -999
                index = -1

                for j in range(self.p_length):
                    if max_v < result_Prop[a, j]:
                        max_v = result_Prop[a, j]
                        index = j

                #a = int(input())

                #i = self.dataset.paragraph_index[a]

                print(label[a], ':', index)

                st = ''
                for i in range(25):
                    #st = st + str(self.question[a, i]) + ' '
                    st = st + self.dataset.question_batch[a, i] + ','
                print(st)
                st = ''
                for i in range(25):
                    # st = st + str(self.question[a, i]) + ' '
                    st = st + self.dataset.paragraph_arr[a, i] + ','
                print(st)


        return 0

    def getPropResult(self, paragraph_, question_):
        batch_paragraph = numpy.zeros((self.batch, self.p_length, self.embedding_size), dtype='f')
        batch_question = numpy.zeros((self.batch, self.q_length, self.embedding_size), dtype='f')

        batch_paragraph[0] = self.dataset.get_glove_sequence(self.p_length, paragraph_)
        batch_question[0] = self.dataset.get_glove_sequence(self.q_length, question_)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            output = self.model_Index()
            output = tf.reshape(output, shape=[self.batch, self.p_length])

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Index/aoa_Reader_Index.ckpf')

            training_feed_dict = {self.x_p_holer: batch_paragraph, self.x_q_holer: batch_question}
            result_Prop = sess.run(output, feed_dict=training_feed_dict)

            max_v = -999
            max_index = -1

            for i in range(self.p_length):
                if max_v < result_Prop[0, i]:
                    max_v = result_Prop[0, i]
                    max_index = i

        return max_index

    def training_prediction_start_stop(self, training_epoch, is_continue):
        with tf.Session() as sess:
            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index, self.POS_Embeddings = self.dataset.get_next_batch()

            tensor_index_Start = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Attention_Label')

            tensor_Label_Start = tf.one_hot(tensor_index_Start, self.p_length, 1, 0)
            tensor_Label__Start = tf.cast(tensor_Label_Start, tf.float32)
            tensor_Label__Start = tf.reshape(tensor_Label__Start, [self.batch, self.p_length])

            tensor_index_Stop = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Attention_Label')

            tensor_Label_Stop = tf.one_hot(tensor_index_Stop, self.p_length, 1, 0)
            tensor_Label__Stop = tf.cast(tensor_Label_Stop, tf.float32)
            tensor_Label__Stop = tf.reshape(tensor_Label__Stop, [self.batch, self.p_length])

            output_Start, output_Stop = self.model_Index_Start_End()
            output_Start = tf.reshape(output_Start, shape=[self.batch, self.p_length])
            output_Stop = tf.reshape(output_Stop, shape=[self.batch, self.p_length])

            Probability_Start = tf.nn.softmax_cross_entropy_with_logits(labels=tensor_Label__Start, logits=output_Start)
            Probability_Stop = tf.nn.softmax_cross_entropy_with_logits(labels=tensor_Label__Stop, logits=output_Stop)

            loss = tf.reduce_mean(tf.add(tf.reduce_mean(Probability_Start), tf.reduce_mean(Probability_Stop)))

            train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index, self.POS_Embeddings = self.dataset.get_next_batch()

                training_feed_dict = {tensor_index_Start: self.start_index, tensor_index_Stop: self.stop_index,
                                      self.x_p_holer: self.paragraph, self.x_q_holer: self.question}

                sess.run(train_step, feed_dict=training_feed_dict)

                if self.dataset.batch_index % 30 == 0:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

                print(self.dataset.batch_index, '/', self.dataset.whole_batch_index, sess.run(loss, feed_dict=training_feed_dict))

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

        return 0

    def test_Start_Stop_Index(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index, self.POS_Embeddings = self.dataset.get_test_batch()

            output_Start, output_Stop = self.model_Index_Start_End()
            output_Start = tf.reshape(output_Start, shape=[self.batch, self.p_length])
            output_Stop = tf.reshape(output_Stop, shape=[self.batch, self.p_length])

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

            training_feed_dict = {self.x_p_holer: self.paragraph, self.x_q_holer: self.question}

            prop_Start =  sess.run(output_Start, feed_dict=training_feed_dict)
            prop_Stop = sess.run(output_Start, feed_dict=training_feed_dict)

            for i in range(self.batch):
                start_index = 0
                max_v = -999

                for j in range(self.p_length):
                    if prop_Start[i, j] > max_v:
                        max_v = prop_Start[i, j]
                        start_index = j

                stop_index = 0
                max_v = -999

                for j in range(self.p_length):
                    if prop_Stop[i, j] > max_v:
                        max_v = prop_Stop[i, j]
                        stop_index = j

                print(self.start_index[i], start_index, ' ', self.stop_index[i], stop_index)

        return 0

    def getProp_Start_Stop_Index(self, paragraph_, question_, is_Test=True):
        batch_paragraph = numpy.zeros((self.batch, self.p_length, self.embedding_size), dtype='f')
        batch_question = numpy.zeros((self.batch, self.q_length, self.embedding_size), dtype='f')

        batch_paragraph[0] = self.dataset.get_glove_sequence(self.p_length, paragraph_)
        batch_question[0] = self.dataset.get_glove_sequence(self.q_length, question_)

        if is_Test:
            batch_paragraph, batch_question, self.start_index, self.stop_index, self.POS_Embeddings = self.dataset.get_test_batch()


        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

            output_start, output_stop = self.model_Index_Start_End()
            output_start = tf.reshape(output_start, shape=[self.batch, self.p_length])
            output_stop = tf.reshape(output_stop, shape=[self.batch, self.p_length])

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/Start_Stop_index/aoa_Reader_Index.ckpf')

            training_feed_dict = {self.x_p_holer: batch_paragraph, self.x_q_holer: batch_question}
            result_Prop_start = sess.run(output_start, feed_dict=training_feed_dict)
            result_Prop_stop = sess.run(output_stop, feed_dict=training_feed_dict)

            max_v = -999
            max_index_Start = -1

            for i in range(self.p_length):
                if max_v < result_Prop_start[0, i]:
                    max_v = result_Prop_start[0, i]
                    max_index_Start = i

            max_v = -999
            max_index_Stop = -1

            for i in range(self.p_length):
                if max_v < result_Prop_stop[0, i]:
                    max_v = result_Prop_stop[0, i]
                    max_index_Stop = i

        return max_index_Start, max_index_Stop