import tensorflow as tf
import Combined_Data_Processor
import BiCorpus_Data_Processor

import FT
import Sentence_Representation

import numpy

class One_Model:

    def __init__(self):
        self.dataset = Combined_Data_Processor.Model()
        self.bi_dataset = BiCorpus_Data_Processor.Data_holder()

        self.SE = Sentence_Representation.Conv_Rep()
        self.Fea_GEN = FT.Feature_Translator(length=50)

        self.Word_Embedding_Dimension = 100

        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.X_P = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Q = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

        self.X_Eng = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Kor = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

        return dot_products / (norm1 * norm2)

    def similiary_score(self, H_P3_, H_Q3_, H_P_, H_Q_):
        with tf.variable_scope("output") as scope:
            sim3 = self.cos_sim(H_P3_, H_Q3_)
            sim3 = tf.reshape(sim3, shape=[tf.shape(sim3)[0], 1])
            sim = self.cos_sim(H_P_, H_Q_)
            sim = tf.reshape(sim, shape=[tf.shape(sim)[0], 1])

            sim_ = tf.concat([sim3, sim], axis=1)
            sim_ = tf.reshape(sim_, shape=[tf.shape(sim_)[0], 2])

            score = tf.contrib.layers.fully_connected(
                inputs=sim_,
                num_outputs=2,
                activation_fn=tf.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

            return score

    def training_prediction_index(self, training_epoch, is_continue, is_SE=True):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            batch_paragraph, batch_question, batch_label, _, _ = self.dataset.get_next_batch()

            #input: English Embedding
            #input: Korean Embedding
            H_P3, H_P = self.SE.Encoder_Conv(70, self.X_P)
            H_Q3, H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)
            score = self.similiary_score(H_P3, H_Q3, H_P, H_Q)

            prop = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=score)
            loss = tf.reduce_mean(prop)
            Train_OP = tf.train.AdamOptimizer(0.0005).minimize(loss)

            H_E3, H_E = self.SE.Encoder_Conv(50, self.X_Eng, reuse=True)
            H_E_con = tf.concat([H_E3, H_E], axis=1)
            H_E_con.set_shape((None, 1141))
            #762 + 379

            g_loss, g_vars, d_loss, d_vars = self.Fea_GEN.get_loss(kor_Embed=self.X_Kor, eng_Embed=H_E_con)

            d_optim = tf.train.AdamOptimizer(0.00005, beta1=0.5) \
                .minimize(d_loss, var_list=d_vars)
            g_optim = tf.train.AdamOptimizer(0.000055, beta1=0.5) \
                .minimize(g_loss, var_list=g_vars)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            if True:
                tf.initialize_variables(
                    g_vars,
                    name='init'
                )

                tf.initialize_variables(
                    d_vars,
                    name='init'
                )

            self.dataset.Batch_Index = 0

            epo = 0

            if is_SE:
                while epo < training_epoch:
                    epo += 1
                    # self.dataset.Batch_Index = 0

                    batch_paragraph, batch_question, batch_label, _, _ = self.dataset.get_next_batch()
                    training_feed_dict = {self.Y: batch_label, self.X_P: batch_paragraph, self.X_Q: batch_question}

                    _, loss_value = sess.run([Train_OP, loss], feed_dict=training_feed_dict)
                    print(epo, ',', ':', loss_value)

                    if epo % 300 == 0:
                        saver = tf.train.Saver()
                        save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')
                        print('saved!')

                saver = tf.train.Saver()
                save_path = saver.save(sess, 'D:\qa_data/Index/DMA_Net.ckpf')
            else:
                if False:
                    init_new_vars_op = tf.initialize_variables(d_vars)
                    init_new_vars_op2 = tf.initialize_variables(g_vars)

                    sess.run(init_new_vars_op)
                    sess.run(init_new_vars_op2)

                while epo < training_epoch:
                    epo += 1

                    batch_kor, batch_eng = self.bi_dataset.get_next_batch()
                    training_feed_dict = {self.X_Kor: batch_kor, self.X_Eng: batch_eng}

                    _, D_loss = sess.run(
                        [d_optim, d_loss],
                        feed_dict=training_feed_dict
                    )

                    _, G_loss = sess.run(
                        [g_optim, g_loss],
                        feed_dict=training_feed_dict
                    )

                    print(epo, ',', ':', D_loss, ' , ', G_loss)

                    if epo % 500 == 0:
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
            H_P3, H_P = self.SE.Encoder_Conv(70, self.X_P)
            H_Q3, H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)
            score = self.similiary_score(H_P3, H_Q3, H_P, H_Q)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            for q in range(200 - 1):
                batch_paragraph, batch_question, batch_label, line, q_line = self.dataset.get_test_Batch1()
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

            for q in range(5000 - 1):
                batch_paragraph, batch_question, batch_label, line, q_line = self.dataset.get_test_Batch2()
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

    def get_test_gan_result(self):
        true_case = 0
        false_case = 0

        with tf.Session() as sess:
            H_P3, H_P = self.Fea_GEN.Generator(self.X_P)
            H_Q3, H_Q = self.Fea_GEN.Generator(self.X_Q, reuse=True)
            score = self.similiary_score(H_P3, H_Q3, H_P, H_Q)

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
            H_P3, H_P = self.Fea_GEN.Generator(self.X_P)
            H_Q3, H_Q = self.Fea_GEN.Generator(self.X_Q, reuse=True)
            score = self.similiary_score(H_P3, H_Q3, H_P, H_Q)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

            for a in range(len(self.bi_dataset.Labels_index) - 1):
                batch_sentence, batch_question, batch_label, sen, que = self.bi_dataset.QA_Test_Batch()

                training_feed_dict = {self.X_P: batch_sentence, self.X_Q: batch_question}

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
