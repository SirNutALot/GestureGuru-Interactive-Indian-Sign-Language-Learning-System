#!/usr/bin/env python3
"""ISL Basic Detection — live webcam debug view of recognized signs (1-9, A-Z)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp

from common import (
    MAX_HANDS,
    calc_landmark_list,
    load_model,
    open_camera,
    pre_process_landmark,
    predict_label,
)

WINDOW_NAME = "ISL Detection (Debug)"


def main():
    print("Loading model...")
    model = load_model()
    print("Model loaded. Press ESC to quit.")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = open_camera()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame.flags.writeable = True

            labels = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    side = handedness.classification[0].label
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    features = pre_process_landmark(landmark_list)
                    try:
                        label = predict_label(model, features)
                    except Exception as e:
                        label = None
                        print("Prediction error:", e)

                    if label:
                        labels.append(f"{side}: {label}")

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            y = 40
            for text in labels or ["No hand detected"]:
                cv2.putText(
                    frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 80, 255), 2
                )
                y += 40

            cv2.putText(
                frame,
                "ESC: Quit  |  Debug detection mode",
                (12, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 160, 160),
                1,
            )
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
