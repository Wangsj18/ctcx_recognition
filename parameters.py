"""
Global parameters
"""
import os


label_abs_folder = "dataset/labels_body_orient"
label_cmd_folder = "dataset/labels_cmd_dir"
label_ges_folder = "dataset/labels_gesture"
label_com_folder = "dataset/labels_ges_with_dir"
video_avi_folder = "dataset/videos"
train_list = "/data/wangsijia/datasets/CTPG/record_revised/train_list.txt"
val_list = "/data/wangsijia/datasets/CTPG/record_revised/val_list.txt"
test_list = "/data/wangsijia/datasets/CTPG/record_revised/test_list.txt"

rnn_saved_joints_folder = "dataset/rnn_input_features/kpt_coordinates"
rnn_saved_cooc_folder = "dataset/rnn_input_features/kpt_cooc_feats"
rnn_saved_jcla_folder = "dataset/rnn_input_features/kpt_jcla_feats"

checkpoint_path_cls = "checkpoints/cls/jcla_abs_cls"
checkpoint_path = "checkpoints/ges/coocmap_jcla_abs_gtcls_delay7"
pretrained_weight = ""
rnn_predicted_abs_cls_folder = "rnn_out/cls/jcla_abs_cls/voting_w75"
rnn_predicted_out_folder = "rnn_out/ges/coocmap_jcla_abs_cls/voting_delay7_w75"  # Predicted labels folder
logs_folder = 'logs'

lstm_units = 32
convlstm_units = 16
label_delay_frames = 7 # Only delay in gesture
time_step = 15 * 10


bones = [[5,7], [7,9], [6,8], [8,10], [5,6], [0,9], [0,10], [9,10], [5,9], [6,10], [0,17]]  # Bone connections.
bones_body = bones[:10]
bones_head = bones[10:]


police_dict = {
                0: "--",
                1: "Stop",
                2: "Go Straight",
                3: "Left Turn",
                4: "Left Turn Wait",
                5: "Right Turn",
                6: "Lane Change",
                7: "Slow Down",
                8: "Pull Over"}

police_dict_chinese = {
                0: "--",
                1: "停止",
                2: "直行",
                3: "左转",
                4: "左待转",
                5: "右转",
                6: "变道",
                7: "减速",
                8: "靠边停车"}

orient_dict = {
                0: "--",
                1: "Self",
                2: "Left",
                3: "Opposite",
                4: "Right"}

complete_dict = {0: "--",
                 1: "Self - Stop", 2: "Left - Go Straight", 3: "Right - Left Turn",
                 4: "Right - Left Turn Wait", 5: "Left - Right Turn", 6: "Self - Lane Change",
                 7: "Self - Slow Down", 8: "Self - Pull Over",
                 9: "Left - Stop", 10: "Opposite - Go Straight", 11: "Self - Left Turn",
                 12: "Self - Left Turn Wait", 13: "Opposite - Right Turn", 14: "Left - Lane Change",
                 15: "Left - Slow Down", 16: "Left - Pull Over",
                 17: "Opposite - Stop", 18: "Right - Go Straight", 19: "Left - Left Turn",
                 20: "Left - Left Turn Wait", 21: "Right - Right Turn", 22: "Opposite - Lane Change",
                 23: "Opposite - Slow Down", 24: "Opposite - Pull Over",
                 25: "Right - Stop", 26: "Self - Go Straight", 27: "Opposite - Left Turn",
                 28: "Opposite - Left Turn Wait", 29: "Self - Right Turn", 30: "Right - Lane Change",
                 31: "Right - Slow Down", 32: "Right - Pull Over"
                 }


def create_necessary_folders():
    def create(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    dirs = [rnn_predicted_out_folder, rnn_predicted_abs_cls_folder, checkpoint_path, checkpoint_path_cls]
    [create(d) for d in dirs]


create_necessary_folders()