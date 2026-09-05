#!/usr/bin/env python3
"""ISL Practice / Test — show a target letter/number; user signs it to learn."""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp

from common import (
    ALPHABET,
    HISTORY_LEN,
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

WINDOW_NAME = "ISL Practice / Test"
TARGET_SHOW_TIME = 2.0


def draw_ui(frame, target, score, attempts, last_result, show_target=True):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (6, 6), (w - 6, 120), (16, 16, 16), -1)
    if show_target and target:
        cv2.putText(
            frame, f"TARGET: {target}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 230, 0), 3
        )
    else:
        cv2.putText(
            frame, "TARGET: -", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (180, 180, 180), 2
        )
    cv2.putText(
        frame,
        f"SCORE: {score}   ATTEMPTS: {attempts}",
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 220, 220),
        2,
    )
    cv2.putText(
        frame,
        f"LAST: {last_result}",
        (12, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        frame,
        "ESC:Quit   SPACE:Skip   R:Reset   T:Toggle target",
        (12, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (160, 160, 160),
        1,
    )


def main():
    print("Loading model...")
    model = load_model()
    print("Practice mode ready. Sign the TARGET shown on screen.")

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hand_histories = empty_hand_histories()
    score = 0
    attempts = 0
    current_target = random.choice(ALPHABET)
    show_feedback_until = 0.0
    last_result = ""
    show_target_flag = True

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
                time.sleep(0.02)
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
                    except Exception:
                        label = None
                    hand_histories[side].append(label)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for side in ("Left", "Right"):
                if side not in seen:
                    hand_histories[side].append(None)

            now = time.time()
            if not show_feedback_until:
                min_tail = max(2, int(round(STABLE_HOLD_TIME * 20)))
                for side in ("Left", "Right"):
                    maj_label, maj_count = majority_vote(hand_histories[side])
                    if not maj_label or maj_count < MIN_VOTES:
                        continue
                    tail = 0
                    for v in reversed(hand_histories[side]):
                        if v == maj_label:
                            tail += 1
                        else:
                            break
                    if tail < min_tail:
                        continue

                    if maj_label == current_target:
                        last_result = "Correct"
                        score += 1
                    else:
                        last_result = f"Wrong ({maj_label})"
                    attempts += 1
                    show_feedback_until = now + TARGET_SHOW_TIME
                    reset_hand_history(hand_histories, side)
                    break

            if show_feedback_until and now >= show_feedback_until:
                current_target = random.choice(ALPHABET)
                show_feedback_until = 0.0
                last_result = ""

            draw_ui(frame, current_target, score, attempts, last_result, show_target_flag)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                last_result = "Skipped"
                attempts += 1
                show_feedback_until = time.time() + TARGET_SHOW_TIME
            elif key == ord("r"):
                score = 0
                attempts = 0
                last_result = "Reset"
            elif key == ord("t"):
                show_target_flag = not show_target_flag

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
