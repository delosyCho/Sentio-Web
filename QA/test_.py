import QA_Model
import tensorflow as tf
x_p = tf.placeholder(dtype=tf.float32, shape=[10, 70, 400])
x_q = tf.placeholder(dtype=tf.float32, shape=[10, 20, 400])

sanet = QA_Model.SA_NET()

start_prediction_s, end_prediction_s, p_loss_s = sanet.create_graph(
                    x_p=x_p, x_q=x_q, p_length=70, q_length=20,
                    context_encoding=False)
