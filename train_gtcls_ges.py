import tensorflow as tf
import numpy as np
import os
import sys
import parameters as pa
import video_utils
import rnn_network
import random
from tqdm import tqdm
import logging
from datetime import datetime


LEARNING_RATE = 0.001
EPOCHS = 20

logger = logging.getLogger('main')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

file_handler = logging.FileHandler(filename=os.path.join(pa.logs_folder, 'train_gtcls_ges-{}.log'.format(datetime.now().strftime('%Y_%m_%d_%H_%M'))))
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(level=logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(ch)


def build_training_ops(loss_tensor, itr_per_epoch):
    """
    Build training ops
    :param loss_tensor:
    :return: [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]
    """
    global_step = tf.Variable(0, trainable=False)

    decaying_learning_rate = tf.train.exponential_decay(
        LEARNING_RATE, global_step, itr_per_epoch * 5, 0.9, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate)
    grads = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Add summary for every gradient
    for grad, var in grads:
        if grad is not None:
            if 'rnn' in var.op.name or 'rconv' in var.op.name:  # Only summary rnn gradients
                tf.summary.histogram(var.op.name + "/gradient", grad)

    summary_op = tf.summary.merge_all()
    return [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]


def main(argv=None):
    batch_size = 16
    time_step = pa.time_step
    num_classes = 9

    feats = ['cooc_map', 'jc']
    labels = ['ges']

    logger.info('Using features: %s' % feats)
    logger.info('Using labels: %s' % labels)
    logger.info('Delay frames: %d' % pa.label_delay_frames)
    logger.info('Save to: %s' % pa.checkpoint_path)

    logger.debug('Preparing index for training data')
    # overall data list
    train_idxes = video_utils.extract_data_list(pa.train_list, time_step, labels, interval=1)
    itr_per_epoch_train = len(train_idxes) // batch_size
    logger.info('Original training set size: %d' % len(train_idxes))
    if batch_size * itr_per_epoch_train < len(train_idxes):
        itr_per_epoch_train += 1
        last_itr_samples_train = batch_size * itr_per_epoch_train - len(train_idxes)
        train_idxes.extend(random.sample(train_idxes, last_itr_samples_train))
    train_idxes_filled = random.sample(train_idxes, len(train_idxes))
    logger.info('Actual training set size: %d' % len(train_idxes))

    val_idxes = video_utils.extract_data_list(pa.val_list, time_step, labels, interval=1)
    itr_per_epoch_val = len(val_idxes) // batch_size
    logger.info('Original validation set size: %d' % len(val_idxes))
    if batch_size * itr_per_epoch_val < len(val_idxes):
        itr_per_epoch_val += 1
        last_itr_samples_val = batch_size * itr_per_epoch_val - len(val_idxes)
        val_idxes.extend(random.sample(val_idxes, last_itr_samples_val))
    logger.info('Actual validation set size: %d' % len(val_idxes))

    # define input placeholder
    logger.debug('Defining inputs and outputs')
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        # batch_time_feature holder:
        tf_btf_1 = tf.placeholder(tf.float32, [batch_size, time_step, 17, 17, 1])
        tf_btf_2 = tf.placeholder(tf.float32, [batch_size, time_step, 30])  # 30
        tf_btf_3 = tf.placeholder(tf.int32, [batch_size, time_step])

        # batch_time label(classes) holder
        tf_btl = tf.placeholder(tf.int32, [batch_size, time_step])

    logger.debug('Defining gesture model')
    with tf.device('/GPU:0'):
        with tf.variable_scope("ges_net"):
            btl_onehot = tf.one_hot(tf_btl, num_classes, axis=-1)
            btf3_onehot = tf.one_hot(tf_btf_3, 5, axis=-1)
            if len(feats) == 2:
                pred = rnn_network.build_fusion_network_map([tf_btf_1, tf_btf_2, btf3_onehot], num_classes)
            elif len(feats) == 1:
                if 'cooc_map' in feats:
                    pred = rnn_network.build_network_coocmap_cls([tf_btf_1, btf3_onehot], num_classes,
                                                                 name='coocmap_net')
                elif 'jc' in feats:
                    pred = rnn_network.build_network_one_feat_cls([tf_btf_2, btf3_onehot], pa.lstm_units, num_classes, name='jc_net')
            loss = rnn_network.build_rnn_loss(pred, btl_onehot)
            lgdts_tensor = build_training_ops(loss, itr_per_epoch_train)
            logger.debug('Defining evaluation criterion')
            # model evaluation
            btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
            btc_pred_max = tf.argmax(btc_pred, 2)
            l_max = tf.argmax(btl_onehot, 2)
            correct_prediction = tf.equal(tf.argmax(btc_pred, 2), tf.argmax(btl_onehot, 2))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with sess.as_default():
        all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        rnn_var = [var for var in all_var]
        rnn_saver = tf.train.Saver(max_to_keep=10)
        bestsaver = tf.train.Saver(max_to_keep=1)
        rnn_ckpt = tf.train.get_checkpoint_state(pa.pretrained_weight)

        if rnn_ckpt:
            logger.info('Loading weights from %s' % pa.pretrained_weight)
            rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
            start_epoch = int(rnn_ckpt.model_checkpoint_path[-1])
        else:
            logger.info('Training the model from scratch')
            sess.run(tf.variables_initializer(rnn_var))
            start_epoch = -1

        logger.debug('Training Started.')
        best = {"acc": 0, "step": -1}

        for epoch in range(EPOCHS):
            epoch = epoch + start_epoch + 1
            all_epochs = EPOCHS + start_epoch
            logger.info('Epoch %d / %d start' % (epoch, all_epochs))
            total_loss_per_epoch = []
            for idx in tqdm(range(itr_per_epoch_train)):
                if pa.rnn_saved_jcla_folder != "":
                    btl = video_utils.extract_labels(train_idxes_filled, idx, batch_size, labels, use_folder='train')
                else:
                    btjc, btl = video_utils.extract_jc_labels(train_idxes_filled, idx, batch_size, labels, use_folder='train')
                btf_3 = video_utils.extract_gt_cls(train_idxes_filled, idx, batch_size, use_folder='train', abs_cls=True)
                if len(feats) == 2:
                    if 'cooc_map' in feats:
                        btf_1 = video_utils.extract_cooc_map(train_idxes_filled, idx, batch_size, use_folder='train')
                    if pa.rnn_saved_jcla_folder != "":
                        btf_2 = video_utils.extract_jcla(train_idxes_filled, idx, batch_size, use_folder='train')
                    else:
                        btf_2 = video_utils.extract_less_bone_length_joint_angle_sign(btjc)
                    feed_dict = {tf_btl: btl, tf_btf_1: btf_1, tf_btf_2: btf_2, tf_btf_3: btf_3}
                elif len(feats) == 1:
                    if 'jc' in feats:
                        if pa.rnn_saved_jcla_folder != "":
                            btf_2 = video_utils.extract_jcla(train_idxes_filled, idx, batch_size, use_folder='train')
                        else:
                            btf_2 = video_utils.extract_less_bone_length_joint_angle_sign(btjc)
                        feed_dict = {tf_btl: btl, tf_btf_2: btf_2, tf_btf_3: btf_3}
                    elif 'cooc_map' in feats:
                        btf_1 = video_utils.extract_cooc_map(train_idxes_filled, idx, batch_size, use_folder='train')
                        feed_dict = {tf_btl: btl, tf_btf_1: btf_1, tf_btf_3: btf_3}

                loss_num, g_step_num, lr_num, train_op = sess.run(lgdts_tensor[0:4], feed_dict=feed_dict)
                total_loss_per_epoch.append(loss_num)
                if g_step_num % 200 == 0:
                    logger.info(
                        'epoch=%.2f step=%d, lr=%f, loss=%g' % (g_step_num / itr_per_epoch_train, g_step_num, lr_num, loss_num))

            logger.info('Epoch %d / %d finished' % (epoch, all_epochs))
            logger.info(
                'Epoch=%.2f Mean Loss=%g' % (epoch, np.mean(total_loss_per_epoch)))
            rnn_saver.save(sess, pa.checkpoint_path + "/ckpt", global_step=epoch)
            logger.info('saving latest model...')

            val_acc_per_epoch = []

            for val_idx in tqdm(range(itr_per_epoch_val)):
                if pa.rnn_saved_jcla_folder != "":
                    btl_test = video_utils.extract_labels(val_idxes, val_idx, batch_size, labels, use_folder='val')
                else:
                    btjc_test, btl_teat = video_utils.extract_jc_labels(val_idxes, val_idx, batch_size, labels, use_folder='val')
                btf_3_test = video_utils.extract_gt_cls(val_idxes, val_idx, batch_size, use_folder='val', abs_cls=True)
                if len(feats) == 2:
                    if 'cooc_map' in feats:
                        btf_1_test = video_utils.extract_cooc_map(val_idxes, val_idx, batch_size, use_folder='val')
                    if pa.rnn_saved_jcla_folder != "":
                        btf_2_test = video_utils.extract_jcla(val_idxes, val_idx, batch_size, use_folder='val')
                    else:
                        btf_2_test = video_utils.extract_less_bone_length_joint_angle_sign(btjc_test)
                    feed_dict_test = {tf_btl: btl_test, tf_btf_1: btf_1_test, tf_btf_2: btf_2_test,
                                      tf_btf_3: btf_3_test}
                elif len(feats) == 1:
                    if 'jc' in feats:
                        if pa.rnn_saved_jcla_folder != "":
                            btf_2_test = video_utils.extract_jcla(val_idxes, val_idx, batch_size, use_folder='val')
                        else:
                            btf_2_test = video_utils.extract_less_bone_length_joint_angle_sign(btjc_test)
                        feed_dict_test = {tf_btl: btl_test, tf_btf_2: btf_2_test, tf_btf_3: btf_3_test}
                    elif 'cooc_map' in feats:
                        btf_1_test = video_utils.extract_cooc_map(val_idxes, val_idx, batch_size, use_folder='val')
                        feed_dict_test = {tf_btl: btl_test, tf_btf_1: btf_1_test, tf_btf_3: btf_3_test}
                btc_pred_num, l_max_num, acc = sess.run([btc_pred_max, l_max, accuracy], feed_dict=feed_dict_test)
                val_acc_per_epoch.append(acc)

            val_acc = np.mean(val_acc_per_epoch)
            logger.info('Val Accuracy=%.6f' % val_acc)
    
            # remember the best acc
            if val_acc > best["acc"]:
                best["acc"] = val_acc
                best["epoch"] = epoch
                logger.info("saving best model...")
                bestsaver.save(sess, pa.checkpoint_path+"/best/ckpt", global_step=epoch)
            logger.info("Best eval on val accuracy: %s at epoch %s, last epoch %s accuracy is %s" % (
                best["acc"], best["epoch"], epoch, val_acc))

    sess.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.app.run()
exit(0)
