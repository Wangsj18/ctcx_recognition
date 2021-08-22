import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import glob
import parameters as pa
import matplotlib
matplotlib.use('Agg')

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

def load_label_cls_ges(cls_file, ges_file, delay):
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


class_num = 9 # [33, 5, 9] 33 for command, 5 for orientations, 9 for gestures

sns.set()
predicted_files = glob.glob(os.path.join(pa.rnn_predicted_out_folder, "*.csv"))
pred = []
gt = []
gt_count = []
if class_num == 33:
    out_path = os.path.join(pa.rnn_predicted_out_folder, "matrix_com.png")
elif class_num == 5:
    out_path = os.path.join(pa.rnn_predicted_out_folder, "matrix_cls.png")
elif class_num == 9:
    out_path = os.path.join(pa.rnn_predicted_out_folder, "matrix_ges.png")

for predicted in predicted_files:
    cls_file = os.path.join(pa.rnn_predicted_abs_cls_folder, os.path.basename(predicted))
    if class_num == 33:
        labels = load_label_cls_ges(cls_file, predicted, pa.label_delay_frames)
    elif class_num == 5:
        labels = load_label(cls_file)
    elif class_num == 9:
        labels = load_label(predicted)
    pred.extend(labels)
    cls_label = os.path.join(pa.label_abs_folder, os.path.basename(predicted))
    ges_label = os.path.join(pa.label_ges_folder, os.path.basename(predicted))
    if class_num == 33:
        gt_labels = load_label_cls_ges(cls_label, ges_label, 0)
    elif class_num == 5:
        gt_labels = load_label(cls_label)
    elif class_num == 9:
        gt_labels = load_label(ges_label)
    gt.extend(gt_labels)

for i in range(class_num):
    gt_count.append(gt.count(i))

if class_num == 33:
    C = confusion_matrix(gt, pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
    classes = ['N', 'SS', 'LG', 'RL', 'RLW', 'LR', 'SC', 'SD', 'SP', 'LS', 'OG', 'SL', 'SLW', 'OR', 'LC', 'LD', 'LP',
              'OS', 'RG', 'LL', 'LLW', 'RR', 'OC', 'OD', 'OP', 'RS','SG','OL','OLW', 'SR', 'RC', 'RD', 'RP',]
    textsize = 5
    fig, ax1 = plt.subplots(figsize=(12, 8))
elif class_num == 5:
    C = confusion_matrix(gt, pred, labels=[0, 1, 2, 3, 4])
    classes = ['N', 'Self', 'Left', 'Opposite', 'Right', ]
    textsize = 9
    fig, ax1 = plt.subplots(figsize=(6, 4))
elif class_num == 9:
    C = confusion_matrix(gt, pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8,])
    classes = ['N', 'ST', 'GS', 'LT', 'LW', 'RT', 'LC', 'SD', 'PO']
    textsize = 7
    fig, ax1 = plt.subplots(figsize=(6, 4))

#print(C)
C2 = C.copy()

for i in range(class_num):
    for j in range(class_num):
        C2[i][j] = (C[i][j] / np.sum(C[i]) * 10000)
#print(C2)

plt.imshow(C2, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title('Confusion Matrix')
# plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

for size in ax1.get_xticklabels():
    #size.set_fontname('Times New Roman')
    if class_num == 33:
        size.set_fontsize('7')
    else:
        size.set_fontsize('9')
for size in ax1.get_yticklabels():
    #size.set_fontname('Times New Roman')
    if class_num == 33:
        size.set_fontsize('7')
    else:
        size.set_fontsize('9')


thresh = C.max() / 2.
iters = np.reshape([[[i,j] for j in range(class_num)] for i in range(class_num)],(C.size,2))
for i, j in iters:
    if C2[i, j] / 100 >= 50:
        plt.text(j, i, format('%.1f' % (C2[i, j] / 100)), ha='center', va='center', fontsize=textsize, color = 'w')
    else:
        plt.text(j, i, format('%.1f' % (C2[i, j] / 100)), ha='center', va='center', fontsize=textsize)

plt.ylabel('True label', fontdict={'size': 10})
plt.xlabel('Recognized label', fontdict={'size': 10})
plt.tight_layout()
plt.grid(0)
plt.savefig(out_path, format='png', dpi=300)
