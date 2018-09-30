import Model
import Data_processor
import tensorflow as tf
import numpy as np
import test
import IAR

div = -4

if div == -4:
    dataset = Data_processor.Data_holder()
    dataset.set_batch()

    while True:
        dataset.get_next_batch()
        print('!!!')
if div == -3:
    net = IAR.Improved_AoA_Reader()
    # net.Batch_Size = 872
    net.get_test_result()
if div == -2:
    net = IAR.Improved_AoA_Reader()
    # net.Batch_Size = 872
    net.training_prediction_index(training_epoch=1000, is_continue=False)
if div == -1:
    dataset = Data_processor.Data_holder()
    dataset.set_batch()

    batch_paragraph, batch_question, batch_start_index, batch_stop_index, batch_start_index_value, batch_stop_index_value, batch_length, \
    batch_paragraph_str, batch_question_str = dataset.get_next_batch()

    for i in range(1000):
        print(batch_question_str[i])
        print(batch_paragraph_str[i])
        answer = ''
        for j in range(batch_start_index_value[i], batch_stop_index_value[i] + 1):
            answer += batch_paragraph_str[i][j] + ' '
        print('answer : ', answer)

        print(batch_paragraph[i][0])
        print(dataset.get_glove(batch_paragraph_str[i][0]))

        input()
if div == 0:
    dataset = Data_processor.Data_holder()
    dataset.set_batch()

    a = np.zeros(shape=[1,], dtype='<U20')
    print(dataset.get_glove('asdasdaddasd'))

if div == 1:
    dataset = Data_processor.Data_holder()
    dataset.set_batch()

    for i in range(1000):
        dataset.Batch_Index = 5000
        batch_paragraph, batch_question, batch_start_index, batch_stop_index, batch_start_index_value, batch_stop_index_value, batch_length = dataset.get_next_batch()
        for j in range(10):
            print(dataset.Paragraphs[dataset.argsort_length[5000 + j]])
            print(dataset.Questions[dataset.argsort_length[5000 + j]])
            print(dataset.Paragraphs[j][int(batch_start_index_value[j]):int(batch_stop_index_value[j]) + 1])
            print(batch_start_index_value[j], batch_stop_index_value[j])
            print(dataset.argsort_length[5000 + j])
            print('-')
        input()

    while True:
        i = int(input())
        print(dataset.Paragraphs[i])
        print(dataset.Questions[i])
        print(dataset.Paragraphs[i][dataset.start_index_batch[i]:dataset.stop_index_batch[i] + 1])
        print(dataset.start_index_batch[i], dataset.stop_index_batch[i])
        print('-')
        input()

elif div == 2:
    net = Model.DMA_NET()
    net.training_prediction_index(training_epoch=5000, is_continue=True  )

elif div == 3:
    net = Model.DMA_NET(isEvaluate=True)
    net.get_test_result()
elif div == 4:
    net = Model.DMA_NET()

    with tf.Session() as sess:
        m = net.Model()
        sess.run(tf.initialize_all_variables())

        paragraphs, questions, start_indexes, stop_indexes, _, _, _, _, _ = net.dataset.get_next_batch()

        training_feed_dict = {net.Y_sta: start_indexes, net.Y_sto: stop_indexes, net.X_P: paragraphs,
                              net.X_Q: questions}
        print(sess.run(m, feed_dict=training_feed_dict)[0])
        print(np.array(sess.run(m, feed_dict=training_feed_dict)).shape)


