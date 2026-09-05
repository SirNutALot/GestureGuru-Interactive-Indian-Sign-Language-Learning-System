"""Shared ISL helpers: paths, alphabet, landmark preprocessing, smoothing."""

from __future__ import annotations

import copy
import itertools
import os
import string
from collections import deque

import numpy as np

# Project root = parent of src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.h5")

# Digits 1-9 + A-Z (must match training label order)
ALPHABET = ["1", "2", "3", "4", "5", "6", "7", "8", "9"] + list(string.ascii_uppercase)

HISTORY_LEN = 7
MIN_VOTES = 4
STABLE_HOLD_TIME = 0.30
MAX_HANDS = 2


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, lm in enumerate(landmarks.landmark):
        lx = min(int(lm.x * image_width), image_width - 1)
        ly = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp[index][0] = temp[index][0] - base_x
        temp[index][1] = temp[index][1] - base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(list(map(abs, temp))) if temp else 1.0

    def normalize_(n):
        return n / max_value

    return list(map(normalize_, temp))


def majority_vote(history):
    valid = [h for h in history if h is not None]
    if not valid:
        return None, 0
    vals, counts = np.unique(valid, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx], int(counts[idx])


def empty_hand_histories(maxlen=HISTORY_LEN):
    histories = {"Left": deque(maxlen=maxlen), "Right": deque(maxlen=maxlen)}
    for side in histories:
        for _ in range(maxlen):
            histories[side].append(None)
    return histories


def reset_hand_history(histories, side):
    histories[side].clear()
    for _ in range(histories[side].maxlen):
        histories[side].append(None)


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.h5 not found at:\n  {model_path}\n"
            "Place the trained model in the models/ folder, or run setup/training first."
        )
    from tensorflow import keras

    return keras.models.load_model(model_path)


def predict_label(model, pre_processed):
    import pandas as pd

    df = pd.DataFrame(pre_processed).transpose()
    preds = model.predict(df, verbose=0)
    if preds.ndim == 2 and preds.shape[0] >= 1:
        pred_class = int(np.argmax(preds, axis=1)[0])
    else:
        pred_class = int(np.argmax(preds))
    if 0 <= pred_class < len(ALPHABET):
        return ALPHABET[pred_class]
    return None


def open_camera(indices=(0, 1)):
    import cv2

    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError("Could not open webcam (tried indices 0 and 1).")
