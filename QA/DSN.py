import tensorflow as tf
import Combined_Data_Processor
import BiCorpus_Data_Processor

import FT
import Sentence_Representation
import numpy

from utils import *

from tensorflow.python.framework import ops
from flip_gradient import flip_gradient

import codecs

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
        """
        추가해야 할 것:
        word2vec 읽어와서 tensor로 변환하여 넘겨주기
        """

        """
        english glove
        """
        in_path_glove = "C:\\Users\\Administrator\\Desktop\\qadataset\\glove6B100d.txt"
        glove_f = codecs.open(in_path_glove, 'r', 'utf-8')

        self.words = []
        self.vectors = []

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = -1

            arr.append(0.002 * pm * i)
        self.words.append('#END')
        self.vectors.append(arr)

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = 0.1
            elif i % 3 == 0:
                pm = -1

            arr.append(0.1 * pm)
        self.words.append('#START')
        self.vectors.append(arr)

        for line in glove_f:
            tokens = line.split(' ')
            self.words.append(tokens.pop(0))
            self.vectors.append(tokens)

        self.vectors = numpy.array((self.vectors), 'f').reshape((-1, self.Word_Embedding_Dimension))

        self.dictionary = numpy.array(self.words)
        self.glove_arg_index = self.dictionary.argsort()
        self.dictionary.sort()
        ###############

        self.word_embedding_eng_tensor = tf.convert_to_tensor(self.vectors, dtype=tf.float32, name='eng_embedding')
        del self.vectors

        """
        korean embedding
        """
        word2vec_kor = codecs.open('C:\\Users\\Administrator\\Desktop\\qadataset\\kor_word2vec_100d', 'r', 'utf-8')
        self.kor_words = []
        self.kor_vectors = []

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = -1

            arr.append(0.002 * pm * i)
        self.kor_words.append('#END')
        self.kor_vectors.append(arr)

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = 0.1
            elif i % 3 == 0:
                pm = -1

            arr.append(0.1 * pm)
        self.kor_words.append('#START')
        self.kor_vectors.append(arr)

        for line in word2vec_kor:
            tokens = line.split('\t')
            self.kor_words.append(tokens.pop(0))
            self.kor_vectors.append(tokens)

        print(self.kor_words[0])
        print(self.kor_words[1])

        self.kor_dictionary = numpy.array(self.kor_words)
        self.word2vec_arg_index = self.kor_dictionary.argsort()
        self.kor_dictionary.sort()
        ###################

        self.word_embedding_kor_tensor = tf.convert_to_tensor(self.kor_vectors, dtype=tf.float32, name='kor_embedding')
        del self.kor_vectors

        self.l = tf.placeholder(tf.float32, [], name='l')  # Gradient reversal scaler

        self.dataset = Combined_Data_Processor.Model()
        self.bi_dataset = BiCorpus_Data_Processor.Data_holder()

        self.SE = Sentence_Representation.Conv_Rep()
        self.Fea_GEN = FT.Feature_Translator(length=50)

        self.Word_Embedding_Dimension = 100

        self.Y_ = tf.placeholder(dtype=tf.int32, shape=[None])
        self.Y_2 = tf.placeholder(dtype=tf.int32, shape=[None])

        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.X_P = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Q = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

        self.X_Eng = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])
        self.X_Kor = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Word_Embedding_Dimension])

        self.eng_vocab_size = self.dictionary.shape[0]
        self.kor_vocab_size = self.kor_dictionary.shape[0]

        self.eng_start_token = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.eng_end_token = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.kor_start_token = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.kor_end_token = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        self.encoder_inputs_eng = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.encoder_inputs_eng_q = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.encoder_inputs_kor = tf.placeholder(dtype=tf.int32, shape=[None, None])

        self.shared_inputs_eng = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.shared_inputs_kor = tf.placeholder(dtype=tf.int32, shape=[None, None])

        self.decoder_inputs_eng = tf.concat([self.eng_start_token, enco])
        self.decoder_inputs_kor = tf.placeholder(dtype=tf.int32, shape=[None, None])

        self.class_label = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.domain_label = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.hidden_size = 200
        self.keep_prob = 0.8

        self.encoder_eng_length = seq_length(self.encoder_inputs_eng)
        self.encoder_kor_length = seq_length(self.encoder_inputs_kor)

        self.shared_length_eng = seq_length(self.shared_inputs_eng)
        self.shared_length_kor = seq_length(self.shared_inputs_kor)

        self.attention_hidden_size = 400
        self.batch_size = 64
        self.max_decoder_length = 50

        self.embedding_size = 100

        self.word_embedding_eng = tf.get_variable("encoder_embeddings",
                                                  shape=[self.eng_vocab_size, self.embedding_size],
                                                  dtype=tf.float32, trainable=True,
                                                  initializer=tf.constant_initializer(self.word_embedding_eng_tensor))
        self.word_embedding_kor = tf.get_variable("decoder_embeddings",
                                                  shape=[self.kor_vocab_size, self.embedding_size],
                                                  dtype=tf.float32, trainable=True,
                                                  initializer=tf.constant_initializer(self.word_embedding_kor_tensor))

    def Private_Encoder_Source(self, inp):
        with tf.variable_scope("source_private"):
            encoder_lookup_inputs = tf.nn.embedding_lookup(self.word_embedding_eng, inp)
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=encoder_lookup_inputs,
                sequence_length=seq_length(inp), dtype=tf.float32, time_major=False)

            private_source = tf.concat([fw_outputs, bw_outputs], -1)
            private_source_state = tf.concat([fw_state[-1], bw_state[-1]], -1)

            return private_source, private_source_state

    def Private_Encoder_Target(self):
        with tf.variable_scope("source_private"):
            encoder_lookup_inputs = tf.nn.embedding_lookup(self.word_embedding_kor, self.encoder_inputs_kor)
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=encoder_lookup_inputs,
                sequence_length=self.encoder_inputs_kor, dtype=tf.float32, time_major=False)

            private_target = tf.concat([fw_outputs, bw_outputs], -1)
            private_target_state = tf.concat([fw_state[-1], bw_state[-1]], -1)

            return private_target, private_target_state

    def Shared_Encoder(self, encoder_input, word_embedding):
        with tf.variable_scope("shared_encoder"):
            encoder_lookup_inputs = tf.nn.embedding_lookup(word_embedding, encoder_input)
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=encoder_lookup_inputs,
                sequence_length=seq_length(encoder_input), dtype=tf.float32, time_major=False)

            output = tf.concat([fw_outputs, bw_outputs], -1)
            output_state = tf.concat([fw_state[-1], bw_state[-1]], -1)

            return output, output_state

    def attention_init(self, encoder_input):
        with tf.variable_scope("attention_layer"):
            mechanism = tf.contrib.seq2seq.LuongAttention(self.attention_hidden_size,
                                                          encoder_input,
                                                          memory_sequence_length=seq_length(encoder_input))
            return mechanism

    def Shared_Decoder(self, output_state, decoder_input, decoder_embedding, mechanism):
        vocab_size = self.dictionary.shape[0]
        if self.kor_dictionary.shape[0] > vocab_size:
            vocab_size = self.kor_dictionary.shape[0]

        with tf.variable_scope("decode_layer"):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size * 2)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            cell = tf.contrib.seq2seq.AttentionWrapper(cell, mechanism,
                                                       attention_layer_size=self.attention_hidden_size,
                                                       alignment_history=False)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, vocab_size)

            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            init_state.clone(cell_state=output_state)

            decoder_lookup_inputs = tf.nn.embedding_lookup(decoder_embedding, decoder_input)
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_lookup_inputs,
                                                       sequence_length=seq_length(decoder_input))

            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, init_state)

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                              maximum_iterations=self.max_decoder_length)
            logits = outputs.rnn_output

            return logits

    def training_prediction_index(self, training_epoch, is_continue):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            batch_paragraph, batch_question, batch_label, _, _ = self.dataset.get_next_batch()

            private_source, private_source_state = self.Private_Encoder_Source(inp=self.encoder_inputs_eng)
            private_source_q, private_source_state_q = self.Private_Encoder_Source(inp=self.encoder_inputs_eng_q)
            private_target, private_target_state = self.Private_Encoder_Target()
            shared_source, shared_source_state = self.Shared_Encoder(self.shared_inputs_eng, self.word_embedding_eng)
            shared_target, shared_target_state = self.Shared_Encoder(self.shared_inputs_kor, self.word_embedding_kor)

            source_mechanism = self.attention_init(shared_source)
            target_mechanism = self.attention_init(shared_target)

            source_logits = self.Shared_Decoder(shared_source_state, self.decoder_inputs_eng, self.word_embedding_eng,
                                                source_mechanism)
            target_logits = self.Shared_Decoder(shared_target_state, self.decoder_inputs_kor, self.word_embedding_kor,
                                                target_mechanism)



            d_vars = self.get_variables_with_name('Discriminator', True, True)
            g_vars = self.get_variables_with_name('generator', True, True)

            optimizer = tf.train.AdamOptimizer(0.0000004)
            optimizer2 = tf.train.AdamOptimizer(0.00012)
            optimizer3 = tf.train.AdamOptimizer(0.0002)

            d_trainer = optimizer.minimize(d_loss, var_list=d_vars)
            dann_trainer = optimizer2.minimize(dann_loss, var_list=g_vars)
            #pred_trainer = optimizer3.minimize(pred_loss)

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
                batch_paragraph, batch_question, batch_label2, batch_label, batch_dump = self.dataset.get_next_batch()
                training_feed_dict = {self.Y: batch_label, self.X_P: batch_paragraph, self.X_Q: batch_question,
                                      self.X_Kor: batch_kor, self.X_Eng: batch_eng}

                _, _, loss_value = sess.run([d_trainer, dann_trainer, dann_loss], feed_dict=training_feed_dict)
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
            with tf.variable_scope('generator') as scope:
                H_P3, H_P = self.SE.Encoder_Conv(50, self.X_P)
                H_Q3, H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)

                # H_P = tf.concat([H_P3, H_P], axis=1)
                # H_Q = tf.concat([H_Q3, H_Q], axis=1)

                H_P3.set_shape((None, 762))
                H_Q3.set_shape((None, 762))
                H_P.set_shape((None, 379))
                H_Q.set_shape((None, 379))

                score = self.similiary_score(H_P, H_Q, H_P3, H_Q3)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'D:\qa_data/Index/DMA_Net.ckpf')

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
            with tf.variable_scope('generator') as scope:
                H_P3, H_P = self.SE.Encoder_Conv(50, self.X_P)
                H_Q3, H_Q = self.SE.Encoder_Conv(50, self.X_Q, reuse=True)

                # H_P = tf.concat([H_P3, H_P], axis=1)
                # H_Q = tf.concat([H_Q3, H_Q], axis=1)

                H_P3.set_shape((None, 762))
                H_Q3.set_shape((None, 762))
                H_P.set_shape((None, 379))
                H_Q.set_shape((None, 379))

                score = self.similiary_score(H_P, H_Q, H_P3, H_Q3)

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
                    print(sen[i], '\n--\n', que[i])
                    print('--')
                    #print(batch_label[i], result[i][0] - result[i][1])
                    #input()

                    if result[i][0] - result[i][1] < min_value:
                        min_value = result[i][0] - result[i][1]
                        min_index = i
                print('---')
                print('score: ', true_case, '/', false_case)
                print()
                #input()

                if min_index != -1:
                    if batch_label[min_index][1] == 1:
                        true_case += 1
                    else:
                        false_case += 1

            print('score: ', true_case, '/', false_case)
            print()
