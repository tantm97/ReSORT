from __future__ import print_function
import os
import numpy as np

def load_features(src):
    # print("[+] Load features....")
    data = []
    label = []
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f.endswith('.npy')]
    for i, f in enumerate(files):
        data.append(np.load(os.path.join(src, f)))
        label.append(f.split(".")[0])
    # print("[+] Load features finished")
    return np.array(data), np.array(label)


def save_feature(save_path, feature):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, feature[0,:])


def calculate_similarity_euclid(thumb, src, tolerance=1.16):
    features, labels = load_features(src)
    if len(features) == 0:
        return -1, -1
    preds = np.linalg.norm(features - thumb, axis=1)
    sort_prob = np.argsort(preds)
    # print('===================>', preds)
    # print(preds)
    # print('------------->', preds[sort_prob[0]])
    if preds[sort_prob[0]] > tolerance:
        return -1, -2
    else:
        return int(labels[sort_prob[0]]), preds[sort_prob[0]]


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))


def calculate_similarity_cosine(thumb, src, tolerance=0.44):
    features, labels = load_features(src)
    if len(features) == 0:
        return -1, -1
    preds = cosin_metric(features, thumb[0])
    sort_prob = np.argsort(preds)
    # print('===================>', preds)
    if preds[sort_prob[-1]] < tolerance:
        return -1, -2
    return int(labels[sort_prob[-1]]), preds[sort_prob[-1]]