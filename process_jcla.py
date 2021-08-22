import sys
import parameters as pa
import numpy as np
import os
import math
import time
from tqdm import tqdm

files = os.listdir(pa.rnn_saved_joints_folder)

for file in tqdm(files):
    file_path = os.path.join(pa.rnn_saved_joints_folder, file)
    tjc = np.load(file_path)
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
    jcla = np.asarray(time_feature)
    np.save(os.path.join(pa.rnn_saved_jcla_folder,file), jcla)

