# gesture_control.py
# Real-time gesture -> game control (thumbs->WASD, palm->mouse, open palm->space, fist->LClick, two->RClick)
# Usage: python gesture_control.py
#
# Controls while running:
#   - 'c' in the display window: calibrate current palm position as screen center (recommended)
#   - 'q' in the display window: quit (releases held keys)
#
# IMPORTANT: Run this with your game/window in focus for inputs to land in that window.
# On macOS give Terminal accessibility permissions. On some games anti-cheat may prevent synthetic inputs.

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pynput.keyboard import Controller as KController, Key
from pynput.mouse import Controller as MController, Button
import pyautogui

# ---------- Configuration (tweak if needed) ----------
THUMB_DIR_THRESHOLD = 0.12   # normalized units, sensitivity for thumb direction
SPREAD_OPEN_THRESHOLD = 0.26 # when finger spread > this -> open_palm (jump)
FINGER_UP_Y_GAP = 0.02       # small gap tolerance for finger up detection
SMOOTHING_ALPHA = 0.22       # mouse smoothing (0..1) - larger = more responsive
GESTURE_HISTORY = 7          # number frames to smooth
MIN_MAJORITY = 4             # min votes for majority
CURSOR_SENSITIVITY = 1.8     # scale palm movement -> screen movement
# ---------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize controllers
kb = KController()
mouse = MController()
screen_w, screen_h = pyautogui.size()

# helper functions
def lm_to_np(landmark_list, w, h):
    """Return Nx2 array of pixel coords from mediapipe landmarks."""
    return np.array([[int(lm.x * w), int(lm.y * h), lm.x, lm.y] for lm in landmark_list.landmark])

def palm_center(land_np):
    """Estimate palm center as mean of several palm landmarks (wrist, index_mcp, pinky_mcp, middle_mcp)."""
    idx = [0, 1, 5, 9, 13, 17]  # wrist + some palm points
    pts = np.array([ [land_np[i,2], land_np[i,3]] for i in idx ])  # normalized coords (x,y)
    cx = pts[:,0].mean()
    cy = pts[:,1].mean()
    return cx, cy

def finger_extended(land_np, finger):
    """
    Very simple heuristic: check if tip is above PIP (for vertical fingers).
    finger: 'index','middle','ring','pinky'
    Returns bool.
    """
    mapping = {
        'thumb': (4, 2),    # tip, knuckle-ish (we won't use thumb here)
        'index': (8, 6),
        'middle': (12, 10),
        'ring': (16, 14),
        'pinky': (20, 18)
    }
    tip, pip = mapping[finger]
    # Use normalized y coords: smaller means higher on image
    return (land_np[tip,3] + FINGER_UP_Y_GAP) < land_np[pip,3]

def count_extended_fingers(land_np):
    fingers = ['index','middle','ring','pinky']
    return sum(1 for f in fingers if finger_extended(land_np, f))

def finger_spread(land_np):
    """Normalized measure of spread: distance between index tip and pinky tip."""
    ix, iy = land_np[8,2], land_np[8,3]
    px, py = land_np[20,2], land_np[20,3]
    return np.hypot(ix - px, iy - py)

def thumb_direction(land_np):
    """
    Compute vector from wrist to thumb tip in normalized coords and return direction string.
    Possible outputs: 'up','down','left','right', or None
    """
    wrist_x, wrist_y = land_np[0,2], land_np[0,3]
    thumb_x, thumb_y = land_np[4,2], land_np[4,3]
    dx = thumb_x - wrist_x
    dy = thumb_y - wrist_y
    # note: y increases downward in image coords
    adx = abs(dx)
    ady = abs(dy)
    if max(adx, ady) < THUMB_DIR_THRESHOLD:
        return None
    # choose dominant axis
    if adx >= ady:
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'

def majority_vote(history, min_count=MIN_MAJORITY):
    """Return majority element in history (list) if it occurs >=min_count, else None"""
    valid = [h for h in history if h]
    if not valid: return None
    vals, counts = np.unique(valid, return_counts=True)
    idx = np.argmax(counts)
    if counts[idx] >= min_count:
        return vals[idx]
    return None

# mapping
thumb_to_key = {
    'up': 'w',
    'left': 'a',
    'right': 'd',
    'down': 's'
}

# state
history = deque(maxlen=GESTURE_HISTORY)
current_movement_key = None
pressed_keys = set()
prev_mouse_x, prev_mouse_y = screen_w//2, screen_h//2
calib_center = None  # if None, will map image center to screen center

# Setup MediaPipe
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    print("Starting. Press 'c' to calibrate center (hold hand at center and press), 'q' to quit.")
    last_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera not available.")
                break
            frame = cv2.flip(frame, 1)  # mirror so it's natural
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = None
            palm_active_for_cursor = False
            # default draw text
            draw_text = "None"

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                lm = results.multi_hand_landmarks[0]
                land_np = lm_to_np(lm, w, h)

                # compute primitives
                spread = finger_spread(land_np)
                fcount = count_extended_fingers(land_np)  # index..pinky only
                tdir = thumb_direction(land_np)

                # classify with rules (order matters)
                # 1) open palm (jump): many fingers extended AND large spread
                if fcount >= 4 and spread > SPREAD_OPEN_THRESHOLD:
                    gesture = 'open_palm'       # jump
                # 2) fist: no fingers extended
                elif fcount == 0 and ( (land_np[4,2]-land_np[0,2])**2 + (land_np[4,3]-land_np[0,3])**2 ) < 0.02:
                    # small thumb-wrist distance implies closed fist
                    gesture = 'fist'
                # 3) two-fingers (index+middle)
                elif finger_extended(land_np, 'index') and finger_extended(land_np, 'middle') and not finger_extended(land_np, 'ring'):
                    gesture = 'two'
                # 4) thumb directions (WASD)
                elif tdir:
                    gesture = 'thumb_' + tdir
                # 5) palm movement for cursor: when many fingers extended but NOT widely spread (palm used as cursor)
                elif fcount >= 3 and spread <= SPREAD_OPEN_THRESHOLD:
                    gesture = 'palm_cursor'
                    palm_active_for_cursor = True
                else:
                    gesture = None

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                # For cursor control compute palm center (normalized)
                pcx, pcy = palm_center(land_np)  # normalized coords 0..1
                # transform palm center to screen position:
                # if calibrated, treat calib_center as image normalized center mapped to screen center,
                # otherwise map image center (0.5,0.5) => screen center.
                if calib_center is None:
                    img_center_x, img_center_y = 0.5, 0.5
                else:
                    img_center_x, img_center_y = calib_center
                # displacement from image center (normalized)
                dx = (pcx - img_center_x)
                dy = (pcy - img_center_y)
                # map to screen delta
                target_x = prev_mouse_x + dx * screen_w * CURSOR_SENSITIVITY
                target_y = prev_mouse_y + dy * screen_h * CURSOR_SENSITIVITY

            else:
                # no hand
                pcx = pcy = None
                gesture = None
                palm_active_for_cursor = False

            # smoothing / majority
            history.append(gesture)
            smoothed = majority_vote(list(history))

            # --- Handle movement keys (thumb gestures) ---
            # if smoothed is thumb_* hold that key, else release
            if smoothed and smoothed.startswith('thumb_'):
                keychar = thumb_to_key.get(smoothed.replace('thumb_',''))
                if keychar:
                    if current_movement_key != keychar:
                        # release previous
                        if current_movement_key and current_movement_key in pressed_keys:
                            try:
                                kb.release(current_movement_key)
                            except Exception:
                                pass
                            pressed_keys.discard(current_movement_key)
                        # press new
                        try:
                            kb.press(keychar)
                        except Exception:
                            pass
                        pressed_keys.add(keychar)
                        current_movement_key = keychar
            else:
                # release held movement key
                if current_movement_key:
                    try:
                        kb.release(current_movement_key)
                    except Exception:
                        pass
                    pressed_keys.discard(current_movement_key)
                    current_movement_key = None

            # --- Handle click & jump on transition (edge detection) ---
            prev = history[-2] if len(history) >= 2 else None
            cur = history[-1] if len(history) >= 1 else None

            # Left click (fist) on transition
            if cur == 'fist' and prev != 'fist':
                try:
                    mouse.click(Button.left, 1)
                except Exception:
                    pass

            # Right click (two) on transition
            if cur == 'two' and prev != 'two':
                try:
                    mouse.click(Button.right, 1)
                except Exception:
                    pass

            # Jump (space) on open_palm transition
            if cur == 'open_palm' and prev != 'open_palm':
                try:
                    kb.press(Key.space)
                    kb.release(Key.space)
                except Exception:
                    pass

            # --- Cursor movement (if palm_active_for_cursor is True) ---
            if palm_active_for_cursor and pcx is not None:
                # compute target screen position as smoothed pos
                tx = int(np.clip(target_x, 0, screen_w-1))
                ty = int(np.clip(target_y, 0, screen_h-1))
                # exponential smoothing
                smx = int(prev_mouse_x * (1.0 - SMOOTHING_ALPHA) + tx * SMOOTHING_ALPHA)
                smy = int(prev_mouse_y * (1.0 - SMOOTHING_ALPHA) + ty * SMOOTHING_ALPHA)
                try:
                    mouse.position = (smx, smy)
                except Exception:
                    pass
                prev_mouse_x, prev_mouse_y = smx, smy

            # Overlay feedback
            disp = smoothed if smoothed is not None else "None"
            cv2.putText(frame, f"Gesture: {disp}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(frame, f"Held: {','.join(sorted(list(pressed_keys))) if pressed_keys else 'None'}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, "Press 'c' to calibrate center, 'q' to quit", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow('Gesture Game Control', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                # calibrate: set current palm center as image-center for mapping
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    img_cx, img_cy = pcx, pcy
                    calib_center = (img_cx, img_cy)
                    print("Calibrated center to:", calib_center)
                else:
                    print("No hand to calibrate; put hand roughly center and press 'c' again.")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        # release pressed keys
        for k in list(pressed_keys):
            try:
                kb.release(k)
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        print("Clean exit.")
