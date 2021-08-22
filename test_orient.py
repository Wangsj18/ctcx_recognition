import glob
import tensorflow as tf
import sys
import parameters as pa
import rnn_network
import argparse
import numpy as np
import os
import video_utils as vu
import metrics.edit_distance as ed
import itertools
import time
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval


def infer_npy_slide_extravoting(base_name):
    feats = ['jc']  # ['cooc_map', 'jc']
    time_step = 5 * 15
    batch_size = 1
    num_classes = 5

    if pa.rnn_saved_jcla_folder != "":
        jcla_path = os.path.join(pa.rnn_saved_jcla_folder, base_name + '.npy')
        jcla_npy = np.load(jcla_path)
        time_len = jcla_npy.shape[0]
        jcla = np.zeros((jcla_npy.shape[0] + time_step - 1, jcla_npy.shape[1]))
        jcla[(time_step - 1):,:] = jcla_npy
    else:
        joints_path = os.path.join(pa.rnn_saved_joints_folder, base_name + '.npy')
        tjc_npy = np.load(joints_path)
        time_len = tjc_npy.shape[0]
        tjc = np.zeros((tjc_npy.shape[0] + time_step - 1, tjc_npy.shape[1], tjc_npy.shape[2]))
        tjc[(time_step - 1):, :, :] = tjc_npy

    cooc_path = os.path.join(pa.rnn_saved_cooc_folder, base_name + '.npy')
    cooc_npy = np.load(cooc_path)
    cooc = np.zeros((cooc_npy.shape[0] + time_step - 1, cooc_npy.shape[1], cooc_npy.shape[2]))
    cooc[(time_step - 1):, :, :] = cooc_npy

    runtime_profile = {
        'rec': [],
        'vote': [],
    }

    # batch_time_feature holder:
    tf_btf_1 = tf.placeholder(tf.float32, [batch_size, time_step, 17, 17, 1])
    tf_btf_2 = tf.placeholder(tf.float32, [batch_size, time_step, 30])  # 30
    # batch_time label(classes) holder
    tf_btl = tf.placeholder(tf.int32, [batch_size, time_step])
    with tf.variable_scope("cls_net"): # "rnn-net","cls_net", "ges_net"
        btl_onehot = tf.one_hot(tf_btl, num_classes, axis=-1)
        if len(feats) == 2:
            pred = rnn_network.build_fusion_network_map_jc([tf_btf_1, tf_btf_2], num_classes)
        elif len(feats) == 1:
            if 'jc' in feats:
                pred = rnn_network.build_network_one_feat([tf_btf_2], pa.lstm_units, num_classes, name='jc_net')
            elif 'cooc_map' in feats:
                pred = rnn_network.build_network_coocmap([tf_btf_1], num_classes, name='coocmap_net')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        rnn_ckpt = tf.train.get_checkpoint_state(pa.checkpoint_path_cls+'/best')
        rnn_saver = tf.train.Saver()
        if rnn_ckpt:
            rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
            print("Loading weights from:", pa.checkpoint_path_cls+'/best')
        else:
            raise RuntimeError("No check point save file.")

        btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
        bt_pred = tf.argmax(btc_pred, 2)

        if 'cooc_map' in feats:
            cooc = np.expand_dims(cooc, -1)

        print("Save to:", pa.rnn_predicted_abs_cls_folder)
        pred_list = []
        M = -7
        cand_list = []
        num_step_desc = tqdm(range(time_len))
        for step in num_step_desc:
            ckpt_time = getTime()
            if len(feats) == 2:
                btf_1 = np.zeros((batch_size, time_step, 17, 17, 1))
                btf_1[0, :, :, :, :] = cooc[step:step + time_step, :, :, :]
                if pa.rnn_saved_jcla_folder != "":
                    btf_2 = np.zeros((batch_size, time_step, jcla.shape[-1]))
                    btf_2[0, :, :] = jcla[step:step + time_step, :]
                else:
                    btjc = np.zeros((batch_size, time_step, 18, 3))
                    btjc[0, :, :, :] = tjc[step:step + time_step, :, :]
                    btf_2 = vu.extract_less_bone_length_joint_angle_sign(btjc)
                feed_dict = {tf_btf_1: btf_1, tf_btf_2: btf_2}
            elif len(feats) == 1:
                if "cooc_map" in feats:
                    btf_1 = np.zeros((batch_size, time_step, 17, 17, 1))
                    btf_1[0, :, :, :, :] = cooc[step:step + time_step, :, :, :]
                    feed_dict = {tf_btf_1: btf_1}
                else:
                    if pa.rnn_saved_jcla_folder != "":
                        btf_2 = np.zeros((batch_size, time_step, jcla.shape[-1]))
                        btf_2[0, :, :] = jcla[step:step + time_step, :]
                    else:
                        btjc = np.zeros((batch_size, time_step, 18, 3))
                        btjc[0, :, :, :] = tjc[step:step + time_step, :, :]
                        btf_2 = vu.extract_less_bone_length_joint_angle_sign(btjc)
                    feed_dict = {tf_btf_2: btf_2}

            bt_pred_num = sess.run(bt_pred, feed_dict=feed_dict)
            pred_result = bt_pred_num[0, M:]
            ckpt_time, rec_time = getTime(ckpt_time)
            runtime_profile['rec'].append(rec_time)
            voting_result = max_voting(pred_result)
            cand_list.append(voting_result)
            if len(cand_list) < 7:
                pred_list.append(voting_result)
            else:
                assert len(cand_list) == 7
                sec_voting_result = max_voting(cand_list)
                pred_list.append(sec_voting_result)
                del cand_list[0]
            ckpt_time, vote_time = getTime(ckpt_time)
            runtime_profile['vote'].append(vote_time)

            num_step_desc.set_description('rec time: {rec:.4f}'.format(rec=np.mean(runtime_profile['rec'])))


    def save_label(label_list, csv_file):
        str_list = ["%d" % e for e in label_list]
        str_line = ",".join(str_list)

        with open(csv_file, 'w') as label_file:
            label_file.write(str_line)
        print("saved: %s" % csv_file)

    save_path = os.path.join(pa.rnn_predicted_abs_cls_folder, base_name + ".csv")
    save_label(pred_list, save_path)


def max_voting(voting_list):
    """
    :param voting_list: a list of int
    :return: the most common value
    """
    if not isinstance(voting_list, list):
        voting_list = voting_list.tolist()
    voting_results = {}
    for i in voting_list:
        voting_results = update_dict_count(i, voting_results)
    voting_tuple = sorted(zip(voting_results.values(), voting_results.keys()))
    max_value = voting_tuple[-1][0]
    max_candidates = []
    for j in voting_results.keys():
        if voting_results[j] == max_value:
            max_candidates.append(j)
    if len(max_candidates) == 1:
        return max_candidates[0]
    else:
        voting_list_reverse = voting_list[::-1]
        idx = [voting_list_reverse.index(x) for x in max_candidates]
        output = voting_list_reverse[min(idx)]
        return output


def update_dict_count(key, dict):
    if key not in dict.keys():
        dict[key] = 1
    else:
        dict[key] += 1
    return dict


def predict_from_test_folder():
    with open(pa.test_list) as f:
        csv_list_test = f.readlines()
    for label_path in tqdm(csv_list_test):
        base_folder = label_path[-8:-5]
        print("Predict: %s" % base_folder)
        tf.reset_default_graph()
        infer_npy_slide_extravoting(base_folder)


def run_accuracy_on_predict_out():
    with open(pa.test_list) as f:
        label_files = f.readlines()
    correct_sum, all_sum = 0, 0
    for label in label_files:
        label = label[:-1]
        pred_name = os.path.basename(label)
        pred_path = os.path.join(pa.rnn_predicted_abs_cls_folder, pred_name)
        cls_path = os.path.join(pa.label_abs_folder, pred_name)
        gt_label = vu.load_label(cls_path)
        pred_label = vu.load_label(pred_path)
        correct_bool = np.equal(pred_label, gt_label)
        acc = np.sum(correct_bool != 0) / len(gt_label)
        print("%s - ACC:%.4f" % (pred_name, acc))
        correct_sum += np.sum(correct_bool != 0)
        all_sum += len(gt_label)
    accuracy = correct_sum / all_sum
    print("OVERALL - ACC:%.4f" % accuracy)


def run_edit_distance_on_predict_out():
    with open(pa.test_list) as f:
        label_files = f.readlines()
    sum_n, sum_i, sum_d, sum_s = 0, 0, 0, 0
    for label in label_files:
        label = label[:-1]
        pred_name = os.path.basename(label)
        pred_path = os.path.join(pa.rnn_predicted_abs_cls_folder, pred_name)
        cls_path = os.path.join(pa.label_abs_folder, pred_name)
        gt_label = vu.load_label(cls_path)
        pred_label = vu.load_label(pred_path)
        gt_group = itertools.groupby(gt_label)
        gt_group = [k for k, g in gt_group]
        pred_group = itertools.groupby(pred_label)
        pred_group = [k for k, g in pred_group]

        S, D, I = ed.SDI(pred_group, gt_group)
        N = len(gt_group)
        acc = (N - I - D - S) / N
        print("%s - N:%d S:%d, D:%d, I:%d, ACC:%.4f" % (pred_name, N, S, D, I, acc))
        # Sum
        sum_n = sum_n + N
        sum_i = sum_i + I
        sum_d = sum_d + D
        sum_s = sum_s + S
    sum_acc = (sum_n - sum_i - sum_d - sum_s) / sum_n
    print("OVERALL - N:%d S:%d, D:%d, I:%d, ACC:%.4f" % (sum_n, sum_s, sum_d, sum_i, sum_acc))


def compute_f1score():
    with open(pa.test_list) as f:
        label_files = f.readlines()
    gt_list = []
    pred_list = []
    for label in label_files:
        label = label[:-1]
        pred_name = os.path.basename(label)
        pred_path = os.path.join(pa.rnn_predicted_abs_cls_folder, pred_name)
        abs_path = os.path.join(pa.label_abs_folder, pred_name)
        gt_label = vu.load_label(abs_path)
        pred_label = vu.load_label(pred_path)
        gt_list.extend(gt_label)
        pred_list.extend(pred_label)
    precision = precision_score(gt_list, pred_list, average="macro")
    recall = recall_score(gt_list, pred_list, average="macro")
    accuracy = accuracy_score(gt_list, pred_list)
    f1score = f1_score(gt_list, pred_list, average="macro")
    print("OVERALL precision -", precision)
    print("OVERALL recall -", recall)
    print("OVERALL accuracy -", accuracy)
    print("OVERALL f1score -", f1score)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='detect orientations')
    parser.add_argument("-p", help="Predict videos from test folder", default=False, action="store_true")
    parser.add_argument("-a", help="Compute bit-to-bit accuracy of predicted labels and ground truth labels",
                        default=False, action="store_true")
    parser.add_argument("-e", help="Compute Edit Distance of predicted labels and ground truth labels", default=False,
                        action="store_true")
    parser.add_argument("-f", help="Compute F1 score",
                        default=False, action="store_true")

    args = parser.parse_args()
    if args.p:
        predict_from_test_folder()
    elif args.a:
        run_accuracy_on_predict_out()
    elif args.e:
        run_edit_distance_on_predict_out()
    elif args.f:
        compute_f1score()
    else:
        print("Please specify an argument.")
