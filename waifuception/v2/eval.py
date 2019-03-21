import math
import os
import os.path as osp
from pathlib import Path
import shutil
import sys
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import model_from_json
import traceback

from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

def preprocess_image(img_path):
    with Image.open(img_path) as img:
        arr = np.array(img.convert('RGB').resize((224,224))).astype(np.float32)

    arr /= 255.0
    arr = (arr - 0.5) * 2.0
    return arr

def filter_labels(eval_metadata, labelset):
    labels = np.zeros(len(labelset))

    if '1girl' in eval_metadata['tags']:
        if 'female' in labelset:
            labels[labelset.index('female')] = 1
    elif '1boy' in eval_metadata['tags']:
        labels[labelset.index('male')] = 1

    labels[labelset.index(eval_metadata['rating'])] = 1
    for idx, label in enumerate(labelset):
        if label in eval_metadata['tags']:
            labels[idx] = 1

    return labels

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        model_arch_json = f.read()

    print("Loading architecture...")
    model = model_from_json(model_arch_json)

    print("Loading weights...")
    model.load_weights(sys.argv[2])

    print("Loading eval data...")
    with open(Path(sys.argv[3]) / 'meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)

    with open(sys.argv[4], 'r', encoding='utf-8') as f:
        labels = json.load(f)

    xs = []
    ys = []
    for post in meta:
        fp = Path(sys.argv[3]) / '{}.jpg'.format(post['id'])

        ys.append(filter_labels(post, labels))
        xs.append(preprocess_image(fp))

    print("Evaluating...")
    ys = np.array(ys).astype(np.int)
    preds = model.predict(np.array(xs), batch_size=len(xs))
    pred_score = sigmoid(preds)

    print(pred_score.shape)

    roc_tpr = []
    roc_fpr = []

    cond_positive = (ys > 0)
    cond_negative = (ys < 1)

    n_cp = np.sum(np.where(cond_positive, 1, 0), axis=0).astype(np.float)
    n_cn = np.sum(np.where(cond_negative, 1, 0), axis=0).astype(np.float)

    for threshold in np.linspace(0.0, 1.0, 50):
        pred_positive = (pred_score >= threshold)
        true_positive = np.where(pred_positive & cond_positive, 1, 0)
        false_positive = np.where(pred_positive & cond_negative, 1, 0)

        n_tp = np.sum(true_positive, axis=0).astype(np.float)
        n_fp = np.sum(false_positive, axis=0).astype(np.float)

        tpr = []
        fpr = []
        for tp, fp, cp, cn in zip(n_tp, n_fp, n_cp, n_cn):
            if cp < 1 or cn < 1:
                continue

            tpr.append(tp / cp)
            fpr.append(fp / cn)

        roc_fpr.append(np.mean(fpr))
        roc_tpr.append(np.mean(tpr))

    for pred, true_label, l in zip(pred_score[0], ys[0], labels):
        predicted_label = False
        if (pred >= 0.5):
            predicted_label = True

        true_label = (true_label > 0)
        print("{}: {} / {}".format(l, predicted_label, true_label))

    plt.plot(roc_fpr, roc_tpr, 'k', label='ROC')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
