# -*- coding: utf-8 -*-

# 절대 임포트 설정
from __future__ import absolute_import
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf

vocabulary_size = 200000 * 2
vocab = 200000

batch_size = 512
embedding_size = 300  # embedding vector의 크기.
skip_window = 1       # 윈도우 크기 : 왼쪽과 오른쪽으로 얼마나 많은 단어를 고려할지를 결정.
num_skips = 2         # 레이블(label)을 생성하기 위해 인풋을 얼마나 많이 재사용 할 것인지를 결정.

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

file = open('vec_input', 'r', encoding='utf-8')
inputs = file.read().split('#!#')
file.close()

file = open('vec_target', 'r', encoding='utf-8')
target = file.read().split('#!#')
file.close()

print('read!', len(target), len(inputs))

dictionary_arr = np.zeros(shape=[vocabulary_size], dtype='<U30')

data, count, dictionary, reverse_dictionary = build_dataset(inputs, vocab)
for i in range(len(reverse_dictionary)):
    dictionary_arr[i] = reverse_dictionary[i]
print('1:', len(reverse_dictionary))

data, count, dictionary, reverse_dictionary = build_dataset(target, vocab)
for i in range(len(reverse_dictionary)):
    dictionary_arr[i + vocab] = reverse_dictionary[i]
print('2:', len(reverse_dictionary))

print('dic', reverse_dictionary[0], dictionary_arr[0])
print('dic', reverse_dictionary[10], dictionary_arr[10])
print('dic', reverse_dictionary[20], dictionary_arr[20])
#input()
dictionary_arr.sort()
"""
while True:
    idx = dictionary_arr.searchsorted(input())
    print(dictionary_arr[idx])
"""
# Step 3: skip-gram model을 위한 트레이닝 데이터(batch)를 생성하기 위한 함수.
def generate_batch(batch_size, num_skips, skip_window):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    np_range = np.arange(0, vocabulary_size)
    np.random.shuffle(np_range)

    index = 0
    index2 = 0

    while True:
        #print(index, '/', batch_size)
        idx = np_range[index2]

        w1 = dictionary_arr.searchsorted(inputs[idx])
        w2 = dictionary_arr.searchsorted(target[idx])

        if w1 < dictionary_arr.shape[0] and w2 < dictionary_arr.shape[0]:
            if dictionary_arr[w1] == inputs[idx] and dictionary_arr[w2] == target[idx]:
                batch[index] = w1
                labels[index] = w2

                index += 1

        if index == batch_size:
            break

        index2 += 1

    return batch, labels

print('batch...')
batch, labels = generate_batch(batch_size=batch_size, num_skips=2, skip_window=1)
print('batch!')

# sample에 대한 validation set은 원래 랜덤하게 선택해야한다. 하지만 여기서는 validation samples을
# 가장 자주 생성되고 낮은 숫자의 ID를 가진 단어로 제한한다.
valid_size = 16     # validation 사이즈.
valid_window = 100  # 분포의 앞부분(head of the distribution)에서만 validation sample을 선택한다.
num_sampled = 64    # sample에 대한 negative examples의 개수.

graph = tf.Graph()

with graph.as_default():
    # 트레이닝을 위한 인풋 데이터들
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_examples = np.random.choice(vocabulary_size, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation

    # embedding vectors 행렬을 랜덤값으로 초기화
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embedding')
    # 행렬에 트레이닝 데이터를 지정
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # NCE loss를 위한 변수들을 선언
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # batch의 average NCE loss를 계산한다.
    # tf.nce_loss 함수는 loss를 평가(evaluate)할 때마다 negative labels을 가진 새로운 샘플을 자동적으로 생성한다.
    loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))

    # SGD optimizer를 생성한다.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # minibatch examples과 모든 embeddings에 대해 cosine similarity를 계산한다.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    """
    여기서 부터 시작하면 됨
    valid embedding 저거를 sess.run 해서 embedding 뽑아보면 됨

    valid_dataset에 index를 넣어주면 됨

    valid_dataset: 저장할 단어 index
    """

# Step 5: 트레이닝을 시작한다.
num_steps = 1000001

with tf.Session(graph=graph) as session:
    # 트레이닝을 시작하기 전에 모든 변수들을 초기화한다.
    tf.initialize_all_variables().run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # optimizer op을 평가(evaluating)하면서 한 스텝 업데이트를 진행한다.
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 100 == 0:
            if step > 0:
                average_loss /= 200
        # 평균 손실(average loss)은 지난 2000 배치의 손실(loss)로부터 측정된다.
                print("Average loss at step ", step, ": ", average_loss)
        average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 200 == 0:
            valid_examples = np.random.choice(vocabulary_size, valid_size, replace=False)
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            saver = tf.train.Saver()
            save_path = saver.save(session, 'D:\\word2vec\\word2vec_kor.ckpf')

            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = dictionary_arr[valid_examples[i]]
                top_k = 8 # nearest neighbors의 개수
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to " + valid_word + ': '
                for k in range(top_k):
                    close_word = dictionary_arr[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

embedding_arr = np.array(final_embeddings, dtype=np.float)
print(embedding_arr.shape)

embedding_file = open('kor_word2vec', 'w', encoding='utf-8')

for i in range(embedding_arr.shape[0]):
    embedding_file.write(count[i][0])
    embedding_file.write('\t')

    for j in range(embedding_arr.shape[1]):
        embedding_file.write(str(embedding_arr[i][j]))
        if j != embedding_arr.shape[1] - 1:
            embedding_file.write('\t')
    embedding_file.write('\n')

embedding_file.close()
# Step 6: embeddings을 시각화한다.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")
