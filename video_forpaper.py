import cv2
import glob
import parameters as pa
import os
import numpy as np

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
    labels = [int(l) for l in labels]
    # Labels is a list of int, representing gesture for each frame
    return labels


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
    if delay!=0:
        cls_pad = ['0'] * delay
        cls_labels = cls_labels[:-1*delay]
        cls_pad.extend(cls_labels)
    else:
        cls_pad = cls_labels
    ges_labels = ges_labels.split(",")
    labels = []
    for a, (cls, ges) in enumerate(zip(cls_pad, ges_labels)):
        if cls == '0' or ges == '0':
            labels.append(0)
        else:
            labels.append((int(cls)- 1) * 8 + int(ges))
    # Labels is a list of int, representing gesture for each frame
    return labels


if __name__ == "__main__":
    predicted_ges_files = glob.glob(os.path.join(pa.rnn_predicted_out_folder, "*.csv"))
    for predicted in predicted_ges_files:
        name = os.path.splitext(os.path.basename(predicted))[0]
        ges_labels = predicted
        cls_labels = os.path.join(pa.rnn_predicted_abs_cls_folder, os.path.basename(predicted))
        labels = load_label_cls_ges(cls_labels, ges_labels, delay=pa.label_delay_frames)
        gt_ges_labels = os.path.join(pa.label_ges_folder, os.path.basename(predicted))
        gt_cls_labels = os.path.join(pa.label_abs_folder, os.path.basename(predicted))
        gt_labels = load_label_cls_ges(gt_cls_labels, gt_ges_labels, delay=0)

        video_file = os.path.join("dataset/estimated_pose_alphapose", "AlphaPose_{}.avi".format(name))
        out_path = os.path.join(pa.rnn_predicted_out_folder, "demo", name+".avi")
        if not os.path.exists(os.path.join(pa.rnn_predicted_out_folder, "demo")):
            os.makedirs(os.path.join(pa.rnn_predicted_out_folder, "demo"))

        if name == "101":
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 15, (1080, 1080))  # (1080, 1080)
        else:
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 15, (1920, 1080))  # (1920, 1080)

        print("Path to save: " + out_path)

        cap = cv2.VideoCapture(video_file)
        frame_cnt = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_cnt == -1:
                    temp_gt_label = np.zeros((2,))
                frame_cnt = frame_cnt + 1
                label_num = labels[frame_cnt]  # A gesture number
                gt_label_num = gt_labels[frame_cnt]
                if gt_label_num != temp_gt_label[1]:
                    temp_gt_label[0] = temp_gt_label[1]
                    temp_gt_label[1] = gt_label_num
                label_en = pa.complete_dict[label_num]  # english word
                if label_num == temp_gt_label[1] or label_num == temp_gt_label[0]:
                    if label_num != 0:
                        if label_num in [4, 12, 20, 28]:
                            if name == "101":
                                cv2.putText(frame, label_en, (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=3)
                            else:
                                cv2.putText(frame, label_en, (560, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                                            thickness=4)
                        elif label_num in [1, 9, 17, 25]:
                            if name == "101":
                                cv2.putText(frame, label_en, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),
                                            thickness=3)
                            else:
                                cv2.putText(frame, label_en, (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                                            thickness=4)
                        else:
                            if name == "101":
                                cv2.putText(frame, label_en, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=3)
                            else:
                                cv2.putText(frame, label_en, (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                                            thickness=4)
                    else:
                        if name == "101":
                            cv2.putText(frame, label_en, (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=3)
                        else:
                            cv2.putText(frame, label_en, (920, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                                        thickness=4)
                else:
                    #print(frame_cnt)
                    if label_num != 0:
                        if label_num in [4, 12, 20, 28]:
                            if name == "101":
                                cv2.putText(frame, label_en, (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
                            else:
                                cv2.putText(frame, label_en, (560, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=4)
                        elif label_num in [1, 9, 17, 25]:
                            if name == "101":
                                cv2.putText(frame, label_en, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                            thickness=3)
                            else:
                                cv2.putText(frame, label_en, (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                            thickness=4)
                        else:
                            if name == "101":
                                cv2.putText(frame, label_en, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
                            else:
                                cv2.putText(frame, label_en, (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                            thickness=4)
                    else:
                        if name == "101":
                            cv2.putText(frame, label_en, (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=3)
                        else:
                            cv2.putText(frame, label_en, (920, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=4)
                out.write(frame)
            else:
                break
        out.release()
        cap.release()