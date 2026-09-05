#!/usr/bin/env python3
"""ISL Keyboard Mapper — stable signs are typed into the focused text field."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KController
from pynput.keyboard import Key

from common import (
    MAX_HANDS,
    MIN_VOTES,
    STABLE_HOLD_TIME,
    calc_landmark_list,
    empty_hand_histories,
    load_model,
    majority_vote,
    open_camera,
    pre_process_landmark,
    predict_label,
    reset_hand_history,
)

WINDOW_NAME = "ISL Keyboard Mapper"
TYPING_DEBOUNCE_SEC = 0.65
POST_TYPE_DELAY_SEC = 0.10


def type_character(kb, ch):
    try:
        if ch.isalpha() and ch.isupper():
            kb.press(Key.shift)
            kb.press(ch.lower())
            kb.release(ch.lower())
            kb.release(Key.shift)
        else:
            kb.press(ch)
            kb.release(ch)
        time.sleep(POST_TYPE_DELAY_SEC)
    except Exception as e:
        print("Typing error:", e)


def main():
    print("Loading model...")
    model = load_model()
    print("Keyboard mapper ready.")
    print("Focus a text field, then sign. Press '[' to toggle typing, ESC to quit.")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    kb = KController()

    hand_histories = empty_hand_histories()
    last_typed_label = {"Left": None, "Right": None}
    last_typed_time = {"Left": 0.0, "Right": 0.0}
    prev_observed_label = {"Left": None, "Right": None}
    stable_start = {"Left": 0.0, "Right": 0.0}
    typing_enabled = True

    cap = open_camera()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame.flags.writeable = True

            seen = set()
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    side = handedness.classification[0].label
                    seen.add(side)
                    features = pre_process_landmark(calc_landmark_list(frame, hand_landmarks))
                    try:
                        label = predict_label(model, features)
                    except Exception as e:
                        label = None
                        print("Prediction error:", e)
                    hand_histories[side].append(label)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            for side in ("Left", "Right"):
                if side not in seen:
                    hand_histories[side].append(None)

            now = time.time()
            for side in ("Left", "Right"):
                maj_label, maj_count = majority_vote(hand_histories[side])
                if maj_label and maj_count >= MIN_VOTES:
                    if prev_observed_label[side] != maj_label:
                        prev_observed_label[side] = maj_label
                        stable_start[side] = now
                    held = now - stable_start[side]
                    if (
                        held >= STABLE_HOLD_TIME
                        and (now - last_typed_time[side]) >= TYPING_DEBOUNCE_SEC
                        and maj_label != last_typed_label[side]
                    ):
                        if typing_enabled:
                            print(f"[{side}] Typing: {maj_label}")
                            type_character(kb, maj_label)
                        last_typed_label[side] = maj_label
                        last_typed_time[side] = now
                        reset_hand_history(hand_histories, side)
                        stable_start[side] = 0.0
                        prev_observed_label[side] = None
                else:
                    prev_observed_label[side] = None
                    stable_start[side] = 0.0

            left_maj, left_votes = majority_vote(hand_histories["Left"])
            right_maj, right_votes = majority_vote(hand_histories["Right"])
            cv2.putText(
                frame,
                f"L: {left_maj or '-'} ({left_votes})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                f"R: {right_maj or '-'} ({right_votes})",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Typing: {'ON' if typing_enabled else 'OFF'}  ([ to toggle)",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                frame,
                "ESC: Quit  |  Focus a text field to receive keys",
                (10, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 160, 160),
                1,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("["):
                typing_enabled = not typing_enabled
                print("Typing toggled:", typing_enabled)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
