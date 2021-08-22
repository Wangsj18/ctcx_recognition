import tensorflow as tf
from tensorflow.contrib import rnn
import parameters as pa
import numpy as np
import video_utils


def build_lstm_network(batch_time_input, n_unit, layer_num):
    """
    build rnn network
    :param batch_time_input: [batch, time_step, n_input]
    :return:prediction_list
    """
    keep_prob = 1.0
    print("Lstm dp:", keep_prob)
    lstm_cell = rnn.MultiRNNCell([rnn.DropoutWrapper(cell=rnn.BasicLSTMCell(num_units=n_unit), input_keep_prob=1.0,
                                                     output_keep_prob=keep_prob) for _ in range(layer_num)])
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, batch_time_input, dtype=tf.float32)
    return outputs


def build_convlstm_dp_network(batch_time_input, n_unit, layer_num=1):
    """
    build rnn network
    :param batch_time_input: [batch, time_step, n_input]
    :return:prediction_list
    """
    convlstm_cell = rnn.ConvLSTMCell(conv_ndims=2, input_shape=[17, 17, 1], output_channels=n_unit, kernel_shape=[3, 3])
    keep_prob = 1.0
    print("Convlstm dp:", keep_prob)
    lstm_cell = rnn.MultiRNNCell(
        [rnn.DropoutWrapper(cell=convlstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob) for _ in range(layer_num)])
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, batch_time_input, dtype=tf.float32, time_major=False)
    return outputs


def build_network_one_feat(batch_time_inputs, n_unit, n_classes, name):
    batch_time_input = batch_time_inputs[0]
    with tf.variable_scope(name):
        if name == 'jc_net':
            lstm_outputs_dp = build_lstm_network(batch_time_input, n_unit, layer_num=1)
        out_weights = tf.Variable(tf.random_normal([n_unit, n_classes]))
        out_bias = tf.Variable(tf.random_normal([n_classes]))
        lstm_prediction_list = []
        for i in range(lstm_outputs_dp.shape[1]):
            output = lstm_outputs_dp[:, i, :]
            lstm_prediction_list.append((tf.matmul(output, out_weights) + out_bias))
    return lstm_prediction_list


def build_network_coocmap(batch_time_inputs, n_classes, name):
    batch_time_input = batch_time_inputs[0]
    with tf.variable_scope(name):
        lstm_outputs_dp = build_convlstm_dp_network(batch_time_input, pa.convlstm_units, layer_num=1)
        out_weights = tf.Variable(tf.random_normal([pa.convlstm_units, n_classes]))
        out_bias = tf.Variable(tf.random_normal([n_classes]))
        lstm_prediction_list = []
        for i in range(lstm_outputs_dp.shape[1]):
            outputs = lstm_outputs_dp[:, i, :]
            ave_pool2d = tf.layers.AveragePooling2D(pool_size=[17, 17], strides=[17, 17], padding='SAME')
            output = tf.squeeze(ave_pool2d(outputs), [1, 2])
            lstm_prediction_list.append((tf.matmul(output, out_weights) + out_bias))
    return lstm_prediction_list


def build_network_coocmap_cls(batch_time_inputs, n_classes, name='coocmap_net'):
    batch_time_input = batch_time_inputs[0]
    batch_time_cls = batch_time_inputs[1]
    with tf.variable_scope(name):
        lstm_outputs_dp = build_convlstm_dp_network(batch_time_input, pa.convlstm_units, layer_num=1)
        out_weights = tf.Variable(tf.random_normal([pa.convlstm_units + 5, n_classes]))
        out_bias = tf.Variable(tf.random_normal([n_classes]))
        # Each output multiply by same fc layer:  list [time_step][batch, n_outputs]
        lstm_prediction_list = []
        for i in range(lstm_outputs_dp.shape[1]):
            outputs_1 = lstm_outputs_dp[:, i, :]
            ave_pool2d = tf.layers.AveragePooling2D(pool_size=[17, 17], strides=[17, 17], padding='SAME')
            output_1 = tf.squeeze(ave_pool2d(outputs_1), [1, 2])
            output_3 = batch_time_cls[:, i, :]
            output = tf.concat([output_1, output_3], 1)
            lstm_prediction_list.append((tf.matmul(output, out_weights) + out_bias))
    return lstm_prediction_list


def build_fusion_network_map(batch_time_inputs, n_classes, name='fusion_net'):
    batch_time_input_1 = batch_time_inputs[0]
    batch_time_input_2 = batch_time_inputs[1]
    batch_time_cls = batch_time_inputs[2]
    with tf.variable_scope(name):
        with tf.variable_scope('cooc_convlstm'):
            lstm_outputs_dp_1 = build_convlstm_dp_network(batch_time_input_1, pa.convlstm_units, layer_num=1)
        with tf.variable_scope('jc_lstm'):
            lstm_outputs_dp_2 = build_lstm_network(batch_time_input_2, pa.lstm_units, layer_num=1)
        out_weights = tf.Variable(tf.random_normal([pa.convlstm_units + pa.lstm_units + 5, n_classes]))
        out_bias = tf.Variable(tf.random_normal([n_classes]))
        # Each output multiply by same fc layer:  list [time_step][batch, n_outputs]
        lstm_prediction_list = []
        for i in range(lstm_outputs_dp_1.shape[1]):
            outputs_1 = lstm_outputs_dp_1[:, i, :, :, :]
            ave_pool2d = tf.layers.AveragePooling2D(pool_size=[17, 17], strides=[17, 17], padding='SAME')
            output_1 = tf.squeeze(ave_pool2d(outputs_1),[1, 2])
            output_2 = lstm_outputs_dp_2[:, i, :]
            output_3 = batch_time_cls[:, i, :]
            output = tf.concat([output_1, output_2, output_3], 1)
            lstm_prediction_list.append((tf.matmul(output, out_weights) + out_bias))
    return lstm_prediction_list


def build_fusion_network_map_jc(batch_time_inputs, n_classes, name='fusion_net'):
    batch_time_input_1 = batch_time_inputs[0]
    batch_time_input_2 = batch_time_inputs[1]
    with tf.variable_scope(name):
        with tf.variable_scope('cooc_convlstm'):
            lstm_outputs_dp_1 = build_convlstm_dp_network(batch_time_input_1, pa.convlstm_units, layer_num=1)
        with tf.variable_scope('jc_lstm'):
            lstm_outputs_dp_2 = build_lstm_network(batch_time_input_2, pa.lstm_units, layer_num=1)
        out_weights = tf.Variable(tf.random_normal([pa.convlstm_units + pa.lstm_units, n_classes]))
        out_bias = tf.Variable(tf.random_normal([n_classes]))
        lstm_prediction_list = []
        for i in range(lstm_outputs_dp_1.shape[1]):
            outputs_1 = lstm_outputs_dp_1[:, i, :, :, :]
            ave_pool2d = tf.layers.AveragePooling2D(pool_size=[17, 17], strides=[17, 17], padding='SAME')
            output_1 = tf.squeeze(ave_pool2d(outputs_1), [1, 2])
            output_2 = lstm_outputs_dp_2[:, i, :]
            output = tf.concat([output_1, output_2], 1)
            lstm_prediction_list.append((tf.matmul(output, out_weights) + out_bias))
    return lstm_prediction_list


def build_network_one_feat_cls(batch_time_inputs, n_unit, n_classes, name):
    batch_time_input = batch_time_inputs[0]
    batch_time_cls = batch_time_inputs[1]
    with tf.variable_scope(name):
        if name == 'jc_net':
            lstm_outputs_dp = build_lstm_network(batch_time_input, n_unit, layer_num=1)
        out_weights = tf.Variable(tf.random_normal([n_unit + 5, n_classes]))
        out_bias = tf.Variable(tf.random_normal([n_classes]))
        # Each output multiply by same fc layer:  list [time_step][batch, n_outputs]
        lstm_prediction_list = []
        for i in range(lstm_outputs_dp.shape[1]):
             output_1 = lstm_outputs_dp[:, i, :]
             output_3 = batch_time_cls[:, i, :]
             output = tf.concat([output_1, output_3], 1)
             lstm_prediction_list.append((tf.matmul(output, out_weights) + out_bias))
    return lstm_prediction_list


def build_rnn_loss(lstm_prediction_list, batch_time_class_label):
    """
    Build rnn loss tensor
    :param lstm_prediction_list: list [time_step][batch, n_outputs]
    :param batch_time_class_label: [batch, time_step, n_classes]
    :return: total loss
    """
    t_bc_label_list = tf.unstack(batch_time_class_label, axis=1)
    time_batch_loss_list = []
    for i in range(len(lstm_prediction_list)):
        time_batch_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=lstm_prediction_list[i], labels=t_bc_label_list[i])
        time_batch_loss_list.append(time_batch_loss)
    loss = tf.reduce_mean(time_batch_loss_list[pa.label_delay_frames:])
    return loss

