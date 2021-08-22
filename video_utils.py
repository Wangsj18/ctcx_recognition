import tensorflow as tf
import sys
import numpy as np
import os
import logging
import parameters as pa


def load_label(csv_file):
    """
    Label file is a csv file using number to mark gesture for each frame
    example content: 0,0,0,2,2,2,2,2,0,0,0,0,0
    :param csv_file:
    :return: list of int
    """
    with open(csv_file, 'r') as label_file:
        labels = label_file.read()

    labels = labels.split(",")
    new_l = []
    for l in labels:
        new_l.append(int(l))
    # Labels is a list of int, representing gesture for each frame
    return new_l


def load_label_cls_ges(cls_file, ges_file, delay=0):
    """
    Label file is a csv file using number to mark gesture for each frame
    example content: 0,0,0,2,2,2,2,2,0,0,0,0,0
    :param csv_file:
    :return: list of int
    """
    with open(cls_file, 'r') as cls:
        cls_labels = cls.read()
    with open(ges_file, 'r') as ges:
        ges_labels = ges.read()

    cls_labels = cls_labels.split(",")
    print(len(cls_labels))
    if delay!=0:
        cls_pad = ['0'] * delay
        cls_labels = cls_labels[:-1*delay]
        cls_pad.extend(cls_labels)
    else:
        cls_pad = cls_labels
    ges_labels = ges_labels.split(",")
    print(len(ges_labels))
    labels = []
    for a, (cls, ges) in enumerate(zip(cls_pad, ges_labels)):
        if cls == '0' or ges == '0':
            labels.append(0)
        else:
            labels.append((int(cls)- 1) * 8 + int(ges))
    # Labels is a list of int, representing gesture for each frame
    return labels


def cooc_map_clip(npy_cooc, start):
    clip_length = pa.time_step
    cooc = np.load(npy_cooc)
    cooc_cut = np.expand_dims(cooc[start:start + clip_length], -1)
    return cooc_cut


def feat_clip(npy, start):
    clip_length = pa.time_step
    feat = np.load(npy)
    feat_cut = feat[start:start + clip_length]
    return feat_cut


def jc_label_clip(npy_joints, label_file, start):
    """
    :param npy_joints:  # feature sequence
    :param label_file:  # csv labels
    :param clip_length:  # clip frame length
    :return:
    """
    clip_length = pa.time_step
    joints = np.load(npy_joints)
    joints_cut = joints[start:start + clip_length]
    labels = load_label(label_file)
    labels = np.array(labels, dtype=np.int64)
    assert len(labels) == joints.shape[0]
    labels_cut = labels[start:start + clip_length]
    labels_cut = labels_delay(labels_cut, pa.label_delay_frames)
    return joints_cut, labels_cut


def label_clip(label_file, start, delay=pa.label_delay_frames):
    """
    :param label_file:  # csv labels
    :param clip_length:  # clip frame length
    :return:
    """
    clip_length = pa.time_step
    labels = load_label(label_file)
    labels = np.array(labels, dtype=np.int64)
    labels_cut = labels[start:start + clip_length]
    if delay != 0:
        labels_cut = labels_delay(labels_cut, delay)
    return labels_cut


def labels_delay(labels, delay_frames):
    z = np.zeros((delay_frames), dtype=np.int64)
    l = len(labels)  # Original label len
    labels = np.concatenate((z, labels))  # len: delay + origin
    labels = labels[:l]
    return labels


def extract_data_list(data_list, clip_length, label_list, interval = 15):
    """
    :param data_list: the list file of train/val/test set, path defined in parameters.py
    :return: zip of [video_no., frame_no.]
    """
    video_frame_list = []
    with open(data_list, 'r') as f:
        videos = f.readlines()
    for b, video_path in enumerate(videos):
        video_base = os.path.basename(video_path).split(".")[0]
        labels = load_label(os.path.join(pa.label_abs_folder, video_base+'.csv'))
        frame_idx = np.arange(0, len(labels)-clip_length+1, interval)
        video_idx = b * np.ones(len(labels)-clip_length+1, dtype=np.int32)
        if label_list == ['ref_cls']:
            ref_labels = load_label(os.path.join(pa.label_ref_folder, video_base+'.csv'))
            frame_idx = [frame_idx[i] for i in range(len(frame_idx)) if ref_labels[i] > 0]
            video_idx = [video_idx[i] for i in range(len(video_idx)) if ref_labels[i] > 0]
        zipped_idx = zip(video_idx, frame_idx)
        video_frame_list.extend(list(zipped_idx))
    return video_frame_list


def extract_cooc_map(data_idx, batch_idx, batch_size, use_folder='train'):
    if use_folder == 'train':
        csv_list = pa.train_list
    elif use_folder == 'val':
        csv_list = pa.val_list
    elif use_folder == 'test':
        csv_list = pa.test_list
    with open(csv_list, "r") as f:
        csv_files = f.readlines()
    btcooc = []
    idxes = data_idx[batch_idx*batch_size:(batch_idx+1)*batch_size]
    for b in range(batch_size):
        idx = idxes[b]
        label_path = csv_files[idx[0]][:-1]
        base_name = label_path[-7:-4]
        cooc_path = os.path.join(pa.rnn_saved_cooc_folder, base_name+'.npy')
        cooc_cut = cooc_map_clip(cooc_path, idx[1])
        btcooc.append(cooc_cut)
    return np.asarray(btcooc)


def extract_jcla(data_idx, batch_idx, batch_size, use_folder='train'):
    if use_folder == 'train':
        csv_list = pa.train_list
    elif use_folder == 'val':
        csv_list = pa.val_list
    elif use_folder == 'test':
        csv_list = pa.test_list
    with open(csv_list, "r") as f:
        csv_files = f.readlines()
    btjcla = []
    idxes = data_idx[batch_idx*batch_size:(batch_idx+1)*batch_size]
    for b in range(batch_size):
        idx = idxes[b]
        label_path = csv_files[idx[0]][:-1]
        base_name = label_path[-7:-4]
        jcla_path = os.path.join(pa.rnn_saved_jcla_folder, base_name+'.npy')
        jcla_cut = feat_clip(jcla_path, idx[1])
        btjcla.append(jcla_cut)
    return np.asarray(btjcla)


def extract_jc_labels(data_idx, batch_idx, batch_size, labels, use_folder='train'):
    if use_folder == 'train':
        csv_list = pa.train_list
    elif use_folder == 'val':
        csv_list = pa.val_list
    elif use_folder == 'test':
        csv_list = pa.test_list
    with open(csv_list, "r") as f:
        csv_files = f.readlines()
    btjc = []  # batch time joint coordinate
    btl = []
    idxes = data_idx[batch_idx*batch_size:(batch_idx+1)*batch_size]
    for b in range(batch_size):
        idx = idxes[b]
        label_path = csv_files[idx[0]][:-1]
        base_name = label_path[-7:-4]
        joints_path = os.path.join(pa.rnn_saved_joints_folder, base_name + '.npy')
        if labels == ['ges']:
            label_path = os.path.join(pa.label_ges_folder, base_name+'.csv')
        elif labels == ['abs_cls']:
            label_path = os.path.join(pa.label_abs_folder, base_name + '.csv')
        elif labels == ['cmd_cls']:
            label_path = os.path.join(pa.label_cmd_folder, base_name + '.csv')
        elif labels == ['com_cls']:
            label_path = os.path.join(pa.label_com_folder, base_name + '.csv')
        joints_cut, labels_cut = jc_label_clip(joints_path, label_path, idx[1])
        btl.append(labels_cut)
        btjc.append(joints_cut)
        return np.asarray(btjc), np.asarray(btl)


def extract_labels(data_idx, batch_idx, batch_size, labels, use_folder='train'):
    if use_folder == 'train':
        csv_list = pa.train_list
    elif use_folder == 'val':
        csv_list = pa.val_list
    elif use_folder == 'test':
        csv_list = pa.test_list
    with open(csv_list, "r") as f:
        csv_files = f.readlines()
    btl = []
    idxes = data_idx[batch_idx*batch_size:(batch_idx+1)*batch_size]
    for b in range(batch_size):
        idx = idxes[b]
        label_path = csv_files[idx[0]][:-1]
        base_name = label_path[-7:-4]
        if 'ges' in labels:
            ges_path = os.path.join(pa.label_ges_folder, base_name+'.csv')
        elif 'abs_cls' in labels:
            ges_path = os.path.join(pa.label_abs_folder, base_name+'.csv')
        elif 'cmd_cls' in labels:
            ges_path = os.path.join(pa.label_cmd_folder, base_name+'.csv')
        elif 'com_ges' in labels:
            ges_path = os.path.join(pa.label_com_folder, base_name + '.csv')
        labels_cut = label_clip(ges_path, idx[1])
        btl.append(labels_cut)
    return np.asarray(btl)


def extract_gt_cls(data_idx, batch_idx, batch_size, use_folder='train', abs_cls=True):
    if use_folder == 'train':
        csv_list = pa.train_list
    elif use_folder == 'val':
        csv_list = pa.val_list
    elif use_folder == 'test':
        csv_list = pa.test_list
    with open(csv_list, "r") as f:
        csv_files = f.readlines()
    btl = []
    idxes = data_idx[batch_idx*batch_size:(batch_idx+1)*batch_size]
    for b in range(batch_size):
        idx = idxes[b]
        label_path = csv_files[idx[0]][:-1]
        base_name = label_path[-7:-4]
        if abs_cls:
            label_path = os.path.join(pa.label_abs_folder, base_name+'.csv')
        labels_cut = label_clip(label_path, idx[1], delay=0)
        btl.append(labels_cut)
    return np.asarray(btl)


def extract_less_bone_length_joint_angle_sign(btjc):
    """
    Produce batch_time_feature array
    :param btjc:
    :return:
    """
    batch_size = btjc.shape[0]
    batch_time_feature = []
    for i in range(batch_size):
        tjc = btjc[i]
        time_feature = _extract_less_length_angle_from_sequence_sign(tjc)
        batch_time_feature.append(time_feature)
    return np.asarray(batch_time_feature)


def _extract_less_length_angle_from_sequence_sign(tjc):
    tjc = np.asarray(tjc)
    v_len = tjc.shape[0]
    assert v_len > 0

    time_feature = []
    for time in range(v_len):
        features_list = []
        joint_coor = tjc[time]  # jc for 1 frame, contains all joint positions

        def occluded(b1, b2):
            if np.less(joint_coor[b1, :], 0).any() or np.less(joint_coor[b2, :],
                                                              0).any():  # At least 1 part is not visible
                return True

        # Head
        head_b1, head_b2 = pa.bones_head[0]
        if occluded(head_b1, head_b2):
            # Head occluded
            head_norm = 1.
        else:
            head_norm = np.linalg.norm(joint_coor[head_b1, 0:2] - joint_coor[head_b2, 0:2]) + 1e-7

        # Body
        list_bone_length = []
        list_joint_angle = []
        for b_num, (b1, b2) in enumerate(pa.bones_body):
            coor1 = joint_coor[b1, 0:2]
            coor2 = joint_coor[b2, 0:2]
            # At least 1 part is not visible
            if occluded(b1, b2):
                # bone length for (b1, b2) = 0
                # joint angle for (b1, b2) = (sin)0, (cos)0
                list_bone_length.append(0)
                list_joint_angle.append(0)
                list_joint_angle.append(0)
            else:  # Both parts are visible
                bone_vec = coor1 - coor2
                bone_norm = np.linalg.norm(bone_vec) + 1e-7
                bone_cross = np.cross(bone_vec, (0, 1))
                bone_dot = np.dot(bone_vec, (0, 1))
                bone_sin = np.true_divide(bone_cross, bone_norm)
                bone_cos = np.true_divide(bone_dot, bone_norm)
                # wrt_h : With respect to head length
                len_wrt_h = np.true_divide(bone_norm * np.sign(bone_vec[0]), head_norm)
                list_bone_length.append(len_wrt_h)
                list_joint_angle.append(bone_sin)
                list_joint_angle.append(bone_cos)
        features_list.extend(list_bone_length)
        features_list.extend(list_joint_angle)
        time_feature.append(features_list)
    return np.asarray(time_feature)

