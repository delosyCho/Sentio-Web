import hgtk
import socket
import numpy as np
import collections
import tensorflow as tf
from  layers import *

def preprocess_stemmer(string):
    string = str(string)
    for i in range(3):
        string = string.replace('..', '.')
    string = string.replace('  ', ' ').replace(' .', '').strip()
    string = str(string[0:len(string) - 1])
    return string


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
        unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def check_the_number(string):
    for i in range(len(string) - 1):
        if not ('0' <= string[i] <= '9'):
            return False

    return True


def check_time(string):
    if len(string) < 2:
        return False

    dics = ['년', '월', '일', '시', '분', '초']

    is_Dic = False

    if len(string) > 2:
        for dic in dics:
            if string[len(string) - 2] == dic:
                is_Dic = True

    if is_Dic is True:
        for i in range(len(string) - 2):
            if not ('0' <= string[i] <= '9'):
                return False

    if is_Dic is True:
        return True

    for dic in dics:
        if string[len(string) - 1] == dic:
            is_Dic = True

    if is_Dic is False:
        return False

    if is_Dic is True:
        for i in range(len(string) - 1):
            if not ('0' <= string[i] <= '9'):
                return False

    return True


def check_number(string):
    if len(string) < 1:
        return False

    for i in range(len(string)):
        if not ('0' <= string[i] <= '9'):
            return False
    return True


def check_univ(string, dic):
    string = str(string)

    for i in range(len(dic)):
        if len(dic[i]) > 1:
            idx = string.find(dic[i])
            if idx != -1:
                return True


def check_dic(string, dic):
    idx = dic.searchsorted(string)
    if idx < dic.shape[0]:
        if dic[idx] == string:
            return True
        else:
            return False
    else:
        return False


class N_Gram_classifier:
    def __init__(self, use_dic=False):
        if use_dic is True:
            f = open('n_dictionary', 'w', encoding='utf-8')
            words = f.read().split('\n')
            self.Dictionary = np.array(words)

        self.use_dic = use_dic

    def get_dictionary(self, string):
        if self.use_dic is True:
            return self.Dictionary
        else:
            words = []
            for i in range(len(string) - 1):
                words.append(str(string[i] + string[i + 1]))

            data, count, dictionary, reverse_dictionary = build_dataset(words, 50000)

            dics = []
            for i in range(len(reverse_dictionary)):
                dics.append(reverse_dictionary[i])

            result = np.array(dics)
            result.sort()

            return result

    def get_n_gram_vector(self, string):
        Dictionary = self.get_dictionary(string)

        sentences = string.split('.')

        n_gram_vector = np.zeros(shape=[len(sentences), Dictionary.shape[0]])

        for i in range(len(sentences)):
            for j in range(len(sentences[i]) - 1):
                word = str(sentences[i][j] + sentences[i][j + 1])

                idx = Dictionary.searchsorted(word)
                if idx < Dictionary.shape[0]:
                    if word == Dictionary[idx]:
                        n_gram_vector[i, idx] += 1
            if np.sum(n_gram_vector[i]) > 0:
                n_gram_vector[i] = n_gram_vector[i] / np.sum(n_gram_vector[i])

        return n_gram_vector


class Rule_Based_QA:
    def __init__(self):
        file = open('hw_names', 'r', encoding='utf-8')
        hw_names = file.read().replace(' ', '\n').replace('(', '').replace(')', '')
        hw_names = hw_names[0:len(hw_names) - 1]
        self.hw_names = np.array(hw_names.split('\n'))
        self.hw_names.sort()
        file.close()

        file = open('hw_places', 'r', encoding='utf-8')
        hw_places = file.read().replace(' ', '\n').replace('(', '').replace(')', '')
        hw_places = hw_places[0:len(hw_places) - 1]
        self.hw_places = np.array(hw_places.split('\n'))
        self.hw_places.sort()
        file.close()

        file = open('hw_univ_', 'r', encoding='utf-8')
        hw_univ = file.read().replace(' ', '\n').replace('(', '').replace(')', '')
        hw_univ = hw_univ[0:len(hw_univ) - 1]
        self.hw_univ = np.array(hw_univ.split('\n'))
        self.hw_univ.sort()
        file.close()

        file = open('question_dic_times', 'r', encoding='utf-8')
        question_dic_times = file.read()
        question_dic_times = question_dic_times[0:len(question_dic_times) - 1]
        self.question_dic_times = question_dic_times.split('\n')
        file.close()

        file = open('question_dic_the_numbers', 'r', encoding='utf-8')
        question_dic_the_numbers = file.read()
        question_dic_the_numbers = question_dic_the_numbers[0:len(question_dic_the_numbers) - 1]
        self.question_dic_the_numbers = question_dic_the_numbers.split('\n')
        file.close()

        file = open('question_dic_numbers', 'r', encoding='utf-8')
        question_dic_numbers = file.read()
        question_dic_numbers = question_dic_numbers[0:len(question_dic_numbers) - 1]
        self.question_dic_numbers = question_dic_numbers.split('\n')
        file.close()

        file = open('question_dic_names', 'r', encoding='utf-8')
        question_dic_names = file.read()
        question_dic_names = question_dic_names[0:len(question_dic_names) - 1]
        self.question_dic_names = question_dic_names.split('\n')
        file.close()

        file = open('question_dic_places', 'r', encoding='utf-8')
        question_dic_places = file.read()
        question_dic_places = question_dic_places[0:len(question_dic_places) - 1]
        self.question_dic_places = question_dic_places.split('\n')
        file.close()

        file = open('question_dic_univs', 'r', encoding='utf-8')
        question_dic_univs = file.read()
        question_dic_univs = question_dic_univs[0:len(question_dic_univs) - 1]
        self.question_dic_univs = question_dic_univs.split('\n')
        file.close()

    def classify_question(self, question):
        score = np.zeros(shape=[6], dtype=np.int32)
        question = str(question)

        for i in range(len(self.question_dic_times)):
            idx = question.find(self.question_dic_times[i])
            if idx != -1:
                score[0] += 1
        for i in range(len(self.question_dic_the_numbers)):
            idx = question.find(self.question_dic_the_numbers[i])
            if idx != -1:
                score[1] += 1
        for i in range(len(self.question_dic_numbers)):
            idx = question.find(self.question_dic_numbers[i])
            if idx != -1:
                score[2] += 1
        for i in range(len(self.question_dic_names)):
            idx = question.find(self.question_dic_names[i])
            if idx != -1:
                score[3] += 1
        for i in range(len(self.question_dic_places)):
            idx = question.find(self.question_dic_places[i])
            if idx != -1:
                score[4] += 1
        for i in range(len(self.question_dic_univs)):
            idx = question.find(self.question_dic_univs[i])
            if idx != -1:
                score[5] += 1

        max_value = 0
        max_index = -1
        for i in range(6):
            if max_value <= score[i]:
                max_value = score[i]
                max_index = i
        print(score)
        return max_index

    def Weight_Sentence(self, String, idx):
        Weight = np.ones(shape=[len(String)], dtype=np.float32)

        functions = []
        functions.append(check_time)
        functions.append(check_the_number)
        functions.append(check_number)
        functions.append(check_dic)
        functions.append(check_dic)
        functions.append(check_univ)

        Dics = []
        Dics.append(0)
        Dics.append(0)
        Dics.append(0)
        Dics.append(self.hw_names)
        Dics.append(self.hw_places)
        Dics.append(self.hw_univ)

        for i in range(len(String)):
            if idx <= 2:
                check = functions[idx](String[i])
            else:
                check = functions[idx](String[i], Dics[idx])

            if check is True:
                Weight[i] = 1.5

        return Weight

    def POS_Weight(self, Word_TK, POS_TK):
        Weight_POS = np.ones(shape=[len(Word_TK)], dtype=np.float32)
        POS_dic = ['OL', 'NN', 'UN', 'NR']

        #print(POS_TK)
        #print(len(POS_TK), len(Word_TK))

        for i in range(len(Word_TK)):
            POS_word = POS_TK[i][0] + POS_TK[i][1]
            is_NN = False

            for j in range(len(POS_dic)):
                if POS_dic[j] == POS_word:
                    is_NN = True

            if is_NN is False:
                Weight_POS[i] = 0

        return Weight_POS


class Stemmer:
    def __init__(self):
        self.HOST = 'localhost'  # 호스트를 지정하지 않으면 가능한 모든 인터페이스를 의미한다.
        self.PORT = 7979  # 포트지정
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.HOST, self.PORT))
        print('접속 대기')
        self.s.listen(10)  # 접속이 있을때까지 기다림
        self.conn, addr = self.s.accept()  # 접속 승인
        print('ready')

    def stemming(self, text):
        TK = text.replace('!', '.').replace('?', '.').split('.')

        result = ''
        POS_inform = ''

        for i in range(len(TK)):
            if len(TK[i]) > 5:
                #print('TK:', TK[i])

                self.conn.send("".join(TK[i]).encode(encoding='utf-8'))  # 받은 데이터를 그대로 클라이언트에 전송
                data = self.conn.recv(1024)

                qa_processed = str(data, encoding='utf-8').replace('\n', '').replace('  ', ' ').replace('###', '').strip()
                result += qa_processed + '. '

                data = self.conn.recv(1024)

                POS_processed = str(data, encoding='utf-8').replace('\n', '').replace('  ', ' ').replace('###', '').strip()
                POS_inform += POS_processed + '. '

        return result, POS_inform


def decompose_hangeul(string, max_length=20):
    #utf-8 hanguel start index: 44032, end index: 55203
    utf_start_hanguel = 44032
    utf_end_hanguel = 55203

    result = ''

    for w in range(len(string)):
        if utf_start_hanguel <= ord(string[w]) <= utf_end_hanguel:
            letters = hgtk.letter.decompose(string[w])

            length = len(letters)

            for i in range(length):
                if letters[i] != '':
                    result += letters[i]

        else:
            result += string[w]

    return result

def check_dic(word, dic):
    for i in range(len(dic)):
        if word == dic[i]:
            return True
    return False

def distance_of_words(word1, word2):
    #dictionaty variable
    consonant_dic = ['ㅂ', 'ㅃ', 'ㅍ', 'ㅁ', 'ㄷ', 'ㄸ', 'ㅌ', 'ㄱ', 'ㄲ', 'ㅋ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅅ',
                     'ㅆ', 'ㄴ', 'ㄹ', 'ㅇ', 'ㅎ']
    consonant_dic1 = ['ㅂ', 'ㅃ', 'ㅍ', 'ㅁ']
    consonant_dic2 = ['ㄷ', 'ㄸ', 'ㅌ']
    consonant_dic3 = ['ㄱ', 'ㄲ', 'ㅋ']
    consonant_dic4 = ['ㅈ', 'ㅉ', 'ㅊ']
    consonant_dic5 = ['ㅅ', 'ㅆ', 'ㄴ', 'ㄹ']
    consonant_dic6 = ['ㅇ', 'ㅎ']

    consonant_dics = []
    consonant_dics.append(consonant_dic1)
    consonant_dics.append(consonant_dic2)
    consonant_dics.append(consonant_dic3)
    consonant_dics.append(consonant_dic4)
    consonant_dics.append(consonant_dic5)
    consonant_dics.append(consonant_dic6)

    vowel_dic = ['ㅏ', 'ㅑ', 'ㅘ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅝ', 'ㅜ', 'ㅠ', 'ㅐ', 'ㅔ', 'ㅒ', 'ㅖ',
                 'ㅡ', 'ㅣ', 'ㅢ', 'ㅟ', 'ㅚ', 'ㅙ', 'ㅞ']
    vowel_dic1 = ['ㅏ', 'ㅑ', 'ㅘ']
    vowel_dic2 = ['ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅝ']
    vowel_dic3 = ['ㅜ', 'ㅠ']
    vowel_dic4 = ['ㅐ', 'ㅔ', 'ㅒ', 'ㅖ']
    vowel_dic5 = ['ㅡ', 'ㅣ', 'ㅢ', 'ㅟ']
    vowel_dic6 = ['ㅚ', 'ㅙ', 'ㅞ']

    vowel_dics = []
    vowel_dics.append(vowel_dic1)
    vowel_dics.append(vowel_dic2)
    vowel_dics.append(vowel_dic3)
    vowel_dics.append(vowel_dic4)
    vowel_dics.append(vowel_dic5)
    vowel_dics.append(vowel_dic6)
    ################

    string1 = decompose_hangeul(word1)
    string2 = decompose_hangeul(word2)

    distance = 0

    if len(string1) != len(string2):
        return 999
    else:
        for i in range(len(string1)):
            if string1[i] != string2[i]:
                idx = -1
                idx2 = -1

                ##word1
                if check_dic(string1[i], vowel_dic) is True:
                    #number of dics is 6
                    for j in range(6):
                        if check_dic(string1[i], vowel_dics[j]) is True:
                            idx = j

                if check_dic(string1[i], consonant_dic) is True:
                    # number of dics is 6
                    for j in range(6):
                        if check_dic(string1[i], consonant_dics[j]) is True:
                            idx = j
                ##

                ##word2
                if check_dic(string2[i], vowel_dic) is True:
                    #number of dics is 6
                    for j in range(6):
                        if check_dic(string2[i], vowel_dics[j]) is True:
                            idx2 = j

                if check_dic(string2[i], consonant_dic) is True:
                    # number of dics is 6
                    for j in range(6):
                        if check_dic(string2[i], consonant_dics[j]) is True:
                            idx2 = j
                ##
                #print(idx, idx2)

                if idx != -1 and idx2 != -1:
                    if idx == idx2:
                        distance += 0.5
                    else:
                        distance += 3
                else:
                    distance += 3

    return distance


def structured_sa_embedding(H, sequence_length, hidden_size, d_a_size, r_size, p_coef, name, fc_size=None, projection=False, reuse=False):
    if fc_size is None:
        fc_size = hidden_size * 2

    with tf.variable_scope(name, reuse=reuse):
        with tf.name_scope("self-attention"):
            H_reshape = tf.reshape(H, [-1, 2 * hidden_size])

            W_s1 = tf.get_variable("W_s1", shape=[2 * hidden_size, d_a_size],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, W_s1))
            W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            _H_s2 = tf.matmul(_H_s1, W_s2)
            print(_H_s2)
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, sequence_length, r_size]), [0, 2, 1])
            A = tf.nn.softmax(_H_s2_reshape, name="attention")

        with tf.name_scope("sentence-embedding"):
            M = tf.matmul(A, H)

        if projection is True:
            M_flat = tf.reshape(M, shape=[-1, 2 * hidden_size * r_size])
            fc = Fully_Connected(M_flat, fc_size, 'fc_layer', activation=tf.nn.tanh, reuse=reuse)

            M_flat = fc
        else:
            M_flat = tf.reshape(M, shape=[-1, 2 * hidden_size * r_size])

        with tf.name_scope("penalization"):
            AA_T = tf.matmul(A, tf.transpose(A, [0, 2, 1]))
            I = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(A)[0], 1]), [-1, r_size, r_size])
            P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))

        with tf.name_scope("penalization_loss"):
            loss_P = tf.reduce_mean(P * p_coef)

    return M_flat, loss_P

def self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.05, sublayers = (1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        # Self attention
        outputs = norm_fn(inputs, scope = "layer_norm_1", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
            num_heads = num_heads, seq_len = seq_len, reuse = reuse,
            mask = mask, is_training = is_training, bias = bias, dropout = dropout)

        residual = outputs + inputs

        # Feed-forward
        outputs = norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
        outputs = conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
        outputs = residual + outputs

        return outputs

def multihead_attention(queries, units, num_heads,
                        memory = None,
                        seq_len = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.05):
    with tf.variable_scope(scope, reuse = reuse):
        # Self attention
        if memory is None:
            memory = queries

        memory = conv(memory, 2 * units, name = "memory_projection", reuse=reuse)
        #print(memory)
        query = conv(queries, units, name = "query_projection", reuse=reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory,2,axis = 2)]

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5
        x = dot_product_attention(Q,K,V,
                                  bias = bias,
                                  seq_len = seq_len,
                                  mask = mask,
                                  is_training = is_training,
                                  scope = "dot_product_attention",
                                  reuse = reuse, dropout = dropout)
        #print(x)
        return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
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


def convolution_block(inputs, reuse, num_conv_layers, dropout, num_filters, kernel_size=1, is_training=True):
    outputs = tf.expand_dims(inputs, 2)
    for i in range(num_conv_layers):
        residual = outputs
        outputs = norm_fn(outputs, scope="layer_norm_%d" % i, reuse=reuse)
        if (i) % 2 == 0:
            outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = depthwise_separable_convolution(outputs,
                                                  kernel_size=(kernel_size, 1), num_filters=num_filters,
                                                  scope="depthwise_conv_layers_%d" % i, is_training=is_training,
                                                  reuse=reuse)
        d_bn = batch_norm(name='d_bn' + str(i))

        #print(outputs)
        #print(residual)

        outputs = tf.nn.leaky_relu(d_bn(outputs)) + residual

    return tf.squeeze(outputs,2)

def residual_conv_block(inputs, num_blocks, num_conv_layer, kernel_size, num_filters, input_projection,
                        seq_len, num_heads, name, is_training=True, dropout=0.0, bias=True, reuse=False):
    with tf.variable_scope(name, reuse = reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name = "input_projection", reuse = reuse)
        outputs = inputs

        for i in range(num_blocks):
            with tf.variable_scope(name + str(i), reuse=reuse):
            #outputs = add_timing_signal_1d(outputs)
                #print('check', reuse)
                outputs = convolution_block(outputs, reuse, num_conv_layer, dropout, num_filters, kernel_size, is_training)
                #print('filter', num_filters)
                #print(outputs)
                outputs = self_attention_block(outputs, num_filters, seq_len, mask=None, num_heads=num_heads,
                                                         scope="self_attention_layers%d" % i, reuse=reuse,
                                                         is_training=is_training,
                                                         bias=bias, dropout=dropout)

        return outputs


def Highway_Network_Fullyconnceted(x, dropout, name, padding, size, activation=tf.sigmoid, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        if padding is True:
            x = Fully_Connected(x, size, name='padding', activation=activation)

        T = Fully_Connected(x, size, 'transform_gate', tf.sigmoid, reuse)
        H = Fully_Connected(x, size, 'activation', activation, reuse)
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def cos_sim(v1, v2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
    dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

    return dot_products / (norm1 * norm2)


def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)
  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)
  correlation_matrix = tf.matmul( private_samples, shared_samples, transpose_a=True)
  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')
  #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
     tf.losses.add_loss(cost)
  return cost


def similiary_score(H_P, H_Q):
    with tf.variable_scope("output") as scope:
        sim = cos_sim(H_P, H_Q)

        score = tf.contrib.layers.fully_connected(
            inputs=sim,
            num_outputs=2,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
            biases_initializer=tf.constant_initializer(1e-04),
            scope="FC"
        )

        return score


def Fully_Connected(inp, output, name, activation, reuse=False):
    h = tf.contrib.layers.fully_connected(
        inputs=inp,
        num_outputs=output,
        activation_fn=activation,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=3e-7),
        biases_initializer=tf.constant_initializer(3e-7),
        scope=name,
        reuse=reuse
    )

    return h


def class_pred_net(feat, name='class_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = Fully_Connected(feat, 256, 'fc1', tf.nn.tanh)
        net = Fully_Connected(net, 128, 'fc2', tf.nn.tanh)
        net = Fully_Connected(net, 64, 'fc2', tf.nn.tanh)

        net = Fully_Connected(net, 2, 'out', None)
    return net


# DOMAIN PREDICTION
def domain_pred_net(feat, flip_gradient, name='domain_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        feat = flip_gradient(feat, -1.0) # GRADIENT REVERSAL
        net = Fully_Connected(feat, 100, 'fc1', tf.nn.tanh)
        net = Fully_Connected(net, 100, 'fc2', tf.nn.tanh)
        net = Fully_Connected(net, 2, 'out', None)
    return net
