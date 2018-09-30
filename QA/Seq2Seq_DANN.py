import tensorflow as tf
import Combined_Data_Processor
import BiCorpus_Data_Processor

import FT
import Sentence_Representation
import numpy

from tensorflow.python.layers import core as layers_core
from tensorflow.python.framework import ops
from flip_gradient import flip_gradient


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


class One_Model:

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

    def __init__(self):
        self.Vocabulary_Size = 0

        self.l = tf.placeholder(tf.float32, [], name='l')  # Gradient reversal scaler

        self.dataset = Combined_Data_Processor.Model()
        self.bi_dataset = BiCorpus_Data_Processor.Data_holder()

        self.SE = Sentence_Representation.Conv_Rep()
        self.Fea_GEN = FT.Feature_Translator(length=50)

        self.Word_Embedding_Dimension = 100

        self.Y_ = tf.placeholder(dtype=tf.int32, shape=[None])
        self.Y_2 = tf.placeholder(dtype=tf.int32, shape=[None])

        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.X_P = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Q = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

        self.X_Eng = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Kor = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

        return dot_products / (norm1 * norm2)

    def similiary_score(self, H_P_, H_Q_):
        with tf.variable_scope("output") as scope:
            sim = self.cos_sim(H_P_, H_Q_)
            sim = tf.reshape(sim, shape=[tf.shape(sim)[0], 1])

            score = tf.contrib.layers.fully_connected(
                inputs=sim,
                num_outputs=1,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

            return score

    def Fully_Connected(self, inp, name, reuse=False):
        h = tf.contrib.layers.fully_connected(
            inputs=inp,
            num_outputs=512,
            activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
            biases_initializer=tf.constant_initializer(1e-04),
            scope=name,
            reuse=reuse
        )

        h = tf.contrib.layers.fully_connected(
            inputs=h,
            num_outputs=1024,
            activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
            biases_initializer=tf.constant_initializer(1e-04),
            scope=name + '2',
            reuse=reuse
        )

        return h

    def Discriminator(self, dis_inp, name, reuse=False):
        with tf.variable_scope(name) as scope:
            h1 = tf.contrib.layers.fully_connected(
                inputs=dis_inp,
                num_outputs=256,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC1",
                reuse=reuse
            )

            h2 = tf.contrib.layers.fully_connected(
                inputs=h1,
                num_outputs=512,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC2",
                reuse=reuse
            )

            h3 = tf.contrib.layers.fully_connected(
                inputs=h2,
                num_outputs=1024,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC3",
                reuse=reuse
            )

            h = tf.contrib.layers.fully_connected(
                inputs=h3,
                num_outputs=1,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC_",
                reuse=reuse
            )

            return h

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def seq2seq(self):
        with tf.variable_scope('encoder') as scope:
            # Build RNN cell
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(400)

            # Run Dynamic RNN
            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_emb_inp,
                sequence_length=self.seq_length(encoder_emb_inp), time_major=True)

        with tf.variable_scope('encoder') as scope:
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            projection_layer = layers_core.Dense(
                tgt_vocab_size, use_bias=False)

            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, decoder_lengths, time_major=True)
            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, encoder_state,
                output_layer=projection_layer)
            # Dynamic decoding

            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.Y_Seq2Seq, logits=logits)
        train_loss = (tf.reduce_mean(crossent))

    def training_prediction_index(self, training_epoch, is_continue):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            batch_paragraph, batch_question, batch_label, _, _ = self.dataset.get_next_batch()

            with tf.variable_scope('similarity') as scope:
                H_P = self.SE.Encoder_Conv(70, self.X_P)
                H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)

                H_P.set_shape((None, 379))
                H_Q.set_shape((None, 379))

                score = self.similiary_score(H_P, H_Q)

            prop = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=score)
            pred_loss = tf.reduce_mean(prop)

            with tf.variable_scope('generator') as scope:
                H_E = self.SE.Encoder_Conv(50, self.X_Eng, reuse=False)
                H_K = self.SE.Encoder_Conv(50, self.X_Kor, reuse=True)

                H_K.set_shape((None, 379))
                H_E.set_shape((None, 379))

            H_K_dis = self.Discriminator(H_K, name='Discriminator')
            H_E_dis = self.Discriminator(H_E, name='Discriminator', reuse=True)

            d_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(H_E_dis), logits=H_E_dis)
            d_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(H_K_dis), logits=H_K_dis)
            d_loss = d_loss1 + d_loss2
            d_loss = tf.reduce_mean(d_loss)

            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(H_K_dis), logits=H_K_dis))
            l1_loss = tf.reduce_mean(tf.abs(H_E_dis - H_K_dis))

            g_loss = g_loss + 100 * l1_loss

            dann_loss = g_loss + pred_loss

            d_vars = self.get_variables_with_name('Discriminator', True, True)
            g_vars = self.get_variables_with_name('generator', True, True)

            optimizer = tf.train.AdamOptimizer(0.0000004)
            optimizer2 = tf.train.AdamOptimizer(0.00012)
            optimizer3 = tf.train.AdamOptimizer(0.0002)

            d_trainer = optimizer.minimize(d_loss, var_list=d_vars)
            dann_trainer = optimizer2.minimize(dann_loss, var_list=g_vars)
            pred_trainer = optimizer3.minimize(pred_loss)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            self.dataset.Batch_Index = 0

            epo = 0

            while epo < training_epoch:
                epo += 1
                # self.dataset.Batch_Index = 0

                batch_kor, batch_eng, batch_dump = self.bi_dataset.get_next_batch()
                batch_paragraph, batch_question, batch_label, _, _ = self.dataset.get_next_batch()
                training_feed_dict = {self.Y: batch_label, self.X_P: batch_paragraph, self.X_Q: batch_question,
                                      self.X_Kor: batch_kor, self.X_Eng: batch_eng}

                _, loss_value = sess.run([pred_trainer, pred_loss], feed_dict=training_feed_dict)
                print(epo, ',', ':', loss_value)

                if epo % 300 == 0:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')
                    print('saved!')

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

        return 0

    def get_test_data_result_(self):
        true_case = 0
        false_case = 0

        with tf.Session() as sess:
            with tf.variable_scope('similarity') as scope:
                H_P = self.SE.Encoder_Conv(70, self.X_P)
                H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)

                H_P.set_shape((None, 379))
                H_Q.set_shape((None, 379))

                score = self.similiary_score(H_P, H_Q)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            for q in range(200 - 1):
                batch_paragraph, batch_question, batch_label, line, q_line = self.dataset.get_test_Batch1()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                result = sess.run(score, feed_dict=training_feed_dict)

                max_value = -9999
                max_index = -1

                for i in range(result.shape[0]):
                    if result[i][0] > max_value:
                        max_index = i
                        max_value = result[i][0]

                if batch_label[max_index][0] == 1:
                    true_case += 1
                else:
                    false_case += 1

                print(true_case, '/', false_case)

            for q in range(5000 - 1):
                batch_paragraph, batch_question, batch_label, line, q_line = self.dataset.get_test_Batch2()
                training_feed_dict = {self.X_P: batch_paragraph, self.X_Q: batch_question}

                result = sess.run(score, feed_dict=training_feed_dict)

                max_value = 9999
                max_index = -1

                for i in range(result.shape[0]):
                    if result[i][0] > max_value:
                        max_index = i
                        max_value = result[i][0]

                if batch_label[max_index][0] == 1:
                    true_case += 1
                else:
                    false_case += 1

                print(true_case, '/', false_case)

        print(true_case, '/', false_case)

    def get_test_gan_result(self):
        true_case = 0
        false_case = 0

        with tf.Session() as sess:
            H_P = self.SE.Encoder_Conv(70, self.X_P)
            H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)
            score = self.similiary_score(H_P, H_Q)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            string_inp = []
            string_inp.append("한성 출신 자 시백 호 석파 본관 전주".split())
            string_inp.append("1907 10 1 대원왕 추봉 헌의 시호 받 흥선헌의대원왕".split())
            string_inp.append("그 본명 이하응".split())
            string_inp.append("서원 철폐 정리 양반 기득권 토호들 민폐 노론 일당독재".split())

            q_string = "흥선대원군 시호".split()
            question_inp = []
            for i in range(len(string_inp)):
                question_inp.append(q_string)

            cur_length = 50
            batch_kor_sentences = numpy.zeros((len(string_inp), cur_length, self.Word_Embedding_Dimension), dtype='f')
            batch_eng_sentences = numpy.zeros((len(string_inp), cur_length, self.Word_Embedding_Dimension), dtype='f')

            for i in range(len(string_inp)):
                batch_kor_sentences[i] = self.bi_dataset.get_glove_sequence_kor(length=cur_length, tokens=string_inp[i])
                batch_eng_sentences[i] = self.bi_dataset.get_glove_sequence_kor(length=cur_length, tokens=question_inp[i])

            training_feed_dict = {self.X_P: batch_kor_sentences, self.X_Q: batch_eng_sentences}

            result = sess.run(score, feed_dict=training_feed_dict)

            for i in range(len(string_inp)):
                print(result[i][0] - result[i][1])

    def get_test_gan_result_Exobrain_Dataset(self):
        true_case = 0
        false_case = 0

        with tf.Session() as sess:
            H_P = self.SE.Encoder_Conv(50, self.X_P)
            H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)
            score = self.similiary_score(H_P, H_Q)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            for a in range(len(self.bi_dataset.Labels_index) - 1):
                batch_sentence, batch_question, batch_label, sen, que, batch_dump = self.bi_dataset.QA_Test_Batch()

                training_feed_dict = {self.X_P: batch_sentence, self.X_Q: batch_question, self.Y_: batch_dump, self.l: -1.0}

                result = sess.run(score, feed_dict=training_feed_dict)

                min_index = -1
                min_value = 999

                for i in range(batch_sentence.shape[0]):
                    print(i, ':', result[i][0] - result[i][1], batch_label[i])
                    print(sen[i], '\n', que[i])
                    #print(batch_sentence[i])
                    #print(batch_question[i])
                    #print()
                    #input()

                    if result[i][0] - result[i][1] < min_value:
                        min_value = result[i][0] - result[i][1]
                        min_index = i
                print('---')
                print('score: ', true_case, '/', false_case)
                print()
                input( )

                if min_index != -1:
                    if batch_label[min_index][1] == 1:
                        true_case += 1
                    else:
                        false_case += 1

            print('score: ', true_case, '/', false_case)
            print()
