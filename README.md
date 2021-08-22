# Skeleton-based Traffic Command Recognition at Road Intersections for Intelligent Vehicles
This is a project that recognizes 4 cross-type directions and 8 kinds of Chinese traffic command gestures, leading to understanding of the meaning of Chinese traffic commands at road intersections.

This project includes code, models and dataset, supporting a manuscript under review. 

### Dataset

Download link: 

Baidu Netdisk: https://pan.baidu.com/s/1x-u_ms7iK-oWapeC0K0vJA  (code: j6u1)
Google Drive: https://drive.google.com/drive/folders/1WcCRwzkwCwWFf1e8Q-ITnotBcEAqLDJx?usp=sharing

Put downloaded videos, labels and pre-generated input vectors in the directory of 'dataset' under the project path.

### Environment

We implement the project under Tensorflow-gpu-1.14.0 and cuda 10.0.

### Test

The checkpoints files have been put in the 'checkpoints/'. 

Change configuration and hyperparameters in parameters.py and the test file as you like.

Run the body orientation classification model:

```
python test_orient.py -p
```

Run the gesture recognition model with the predicted orientation results:

```
python test_gtcls_ges.py -p
```

### Evaluation

Compute the accuracy of the classified body orientations:

```
python test_orient.py -a
```

Compute the edit accuracy / f1-score of the recognized gestures:

```
python test_gtcls_ges.py -e
python test_gtcls_ges.py -f
```

Compute the edit accuracy / f1-score of the combined command:

```
python test_gtcls_ges.py -ce
python test_gtcls_ges.py -cf
```

Compute the confusion matrix of the recognized results:

```
python plot_confusion_matrix.py
```

### Training

Change configuration and hyperparameters in parameters.py and the main file as you like.

Train for a body orientation classification model:

```
python train_orient.py
```

Train for a gesture recognition model:

```
python train_gtcls_ges.py
```

### Video demo

After the body orientations and gestures are recognized, a video demo could be produced by running:

```
python video_forpaper.py
```

