# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:23:32 2026

@author: NDTES_YOG
"""

# -*- coding: utf-8 -*-
"""
virtual_workspace.py
---------------------
A unified system combining both Virtual Mouse and Virtual Keyboard.
Toggle modes using your physical keyboard:
  - Press 'm' to switch to MOUSE mode
  - Press 'k' to switch to KEYBOARD mode
  - Press 'q' to quit

Requirements: pip install opencv-python mediapipe pyautogui numpy pynput
"""

import cv2
import time
import numpy as np
import pyautogui
from pynput.keyboard import Controller
from hand_tracking_module import HandDetector

# ---------- Configuration ----------
CAM_WIDTH, CAM_HEIGHT = 1000, 600
FRAME_REDUCTION = 120          # Margin for mouse movement
SMOOTHENING = 4                # Mouse cursor smoothing factor
CLICK_DISTANCE_THRESHOLD = 35  # Pinch threshold (thumb to index tip)
CLICK_COOLDOWN = 0.4           # Cooldown between clicks/keypresses

# Initialize controllers
keyboard_controller = Controller()
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# Keyboard layout configuration
KEY_ROWS = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
]
KEY_W, KEY_H = 70, 70
KEY_GAP = 10
START_X, START_Y = 30, 150


class Key:
    def __init__(self, x, y, w, h, text):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text

    def draw(self, frame, highlighted=False):
        # Green if hovered/pressed, pink otherwise
        color = (0, 255, 0) if highlighted else (255, 0, 255)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, cv2.FILLED)
        cv2.putText(
            frame,
            self.text,
            (self.x + 15, self.y + 45),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            2,
        )

    def contains(self, px, py):
        return self.x < px < self.x + self.w and self.y < py < self.y + self.h


def build_keys():
    keys = []
    for row_idx, row in enumerate(KEY_ROWS):
        for col_idx, char in enumerate(row):
            x = START_X + col_idx * (KEY_W + KEY_GAP)
            y = START_Y + row_idx * (KEY_H + KEY_GAP)
            keys.append(Key(x, y, KEY_W, KEY_H, char))
    
    # Add Space and Backspace at the bottom row
    space_y = START_Y + len(KEY_ROWS) * (KEY_H + KEY_GAP)
    keys.append(Key(START_X, space_y, 400, KEY_H, "SPACE"))
    keys.append(Key(START_X + 410, space_y, 190, KEY_H, "BACKSPACE"))
    return keys


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    detector = HandDetector(max_hands=1, detection_conf=0.7, track_conf=0.7)
    keys = build_keys()
    
    # State tracking variables
    mode = "MOUSE"  # Default mode: "MOUSE" or "KEYBOARD"
    typed_text = ""
    last_click_time = 0
    prev_x, prev_y = 0, 0
    prev_time = 0

    print("System Started. Default mode: MOUSE.")
    print("Press 'm' for Mouse mode, 'k' for Keyboard mode, 'q' to quit.")
    
    frame_count = 0

    # ONE unified loop to handle everything cleanly
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # Mirror frame for natural interaction
        
        # Increment frame counter and only track AI gestures on alternating frames
        frame_count += 1
        landmarks = []
        
        if frame_count % 2 == 0:
            frame = detector.find_hands(frame)
            landmarks = detector.find_position(frame)
        
        # ------------------ MOUSE MODE ------------------
        if mode == "MOUSE":
            # Draw the restricted movement frame boundaries
            cv2.rectangle(
                frame,
                (FRAME_REDUCTION, FRAME_REDUCTION),
                (CAM_WIDTH - FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION),
                (255, 0, 255),
                2,
            )

            if len(landmarks) != 0:
                index_x, index_y = landmarks[8][1], landmarks[8][2]
                fingers = detector.fingers_up()

                # Move Cursor: Index up, Middle down
                if fingers[1] == 1 and fingers[2] == 0:
                    target_x = np.interp(index_x, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, screen_width))
                    target_y = np.interp(index_y, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, screen_height))

                    # Apply smoothing
                    curr_x = prev_x + (target_x - prev_x) / SMOOTHENING
                    curr_y = prev_y + (target_y - prev_y) / SMOOTHENING

                    pyautogui.moveTo(curr_x, curr_y)
                    cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

                # Click: Both Index and Middle fingers are up, then pinch thumb to index
                elif fingers[1] == 1 and fingers[2] == 1:
                    length, frame, line_info = detector.find_distance(4, 8, frame)
                    if length < CLICK_DISTANCE_THRESHOLD:
                        now = time.time()
                        if now - last_click_time > CLICK_COOLDOWN:
                            cv2.circle(frame, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                            pyautogui.click()
                            last_click_time = now

        # ------------------ KEYBOARD MODE ------------------
        elif mode == "KEYBOARD":
            hovered_key = None
            if len(landmarks) != 0:
                index_x, index_y = landmarks[8][1], landmarks[8][2]

                for key in keys:
                    if key.contains(index_x, index_y):
                        hovered_key = key
                        break

                if hovered_key:
                    # Pinch thumb (4) and index (8) to execute keypress
                    length, frame, line_info = detector.find_distance(4, 8, frame, draw=False)
                    if length < CLICK_DISTANCE_THRESHOLD:
                        now = time.time()
                        if now - last_click_time > CLICK_COOLDOWN:
                            last_click_time = now
                            if hovered_key.text == "SPACE":
                                typed_text += " "
                                keyboard_controller.press(" ")
                                keyboard_controller.release(" ")
                            elif hovered_key.text == "BACKSPACE":
                                typed_text = typed_text[:-1]
                                keyboard_controller.press("\b")
                                keyboard_controller.release("\b")
                            else:
                                typed_text += hovered_key.text
                                keyboard_controller.press(hovered_key.text.lower())
                                keyboard_controller.release(hovered_key.text.lower())

            # Render key visual elements
            for key in keys:
                key.draw(frame, highlighted=(key is hovered_key))

            # Render output live text box
            cv2.rectangle(frame, (START_X, 40), (CAM_WIDTH - 30, 110), (50, 50, 50), cv2.FILLED)
            cv2.putText(frame, typed_text[-30:], (START_X + 10, 90), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 2)

        # UI Overlay Metadata
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame, f"MODE: {mode}", (CAM_WIDTH - 250, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)

        cv2.imshow("Virtual Workspace", frame)
        
        # Listen for key actions
        key_stroke = cv2.waitKey(1) & 0xFF
        if key_stroke == ord("q"):
            break
        elif key_stroke == ord("m"):
            mode = "MOUSE"
        elif key_stroke == ord("k"):
            mode = "KEYBOARD"

    # Clean execution end
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()