"""
calibrate.py — Driveway Zone Calibration Tool

Usage:
    python calibrate.py                  # uses auto-detected camera
    python calibrate.py --camera 1       # specific camera index
    python calibrate.py --video clip.mp4 # use a video file

Steps:
    1. Click and drag to draw the danger zone rectangle
    2. Click once to set the yellow line (caution threshold)
    3. Click once to set the red line (high danger threshold)
    4. Press S to save and print the values to copy into main.py
    5. Press R to reset and start over
    6. Press Q to quit without saving
"""

import cv2
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description="Driveway Zone Calibration Tool")
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--camera", type=int, default=None)
args = parser.parse_args()

if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {args.video}")
elif args.camera is not None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
else:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera found. Use --camera N or --video path.mp4")

# --- Display scaling ---
# Scale the window to fit within this max size while preserving aspect ratio
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

def get_scale(frame):
    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    return scale

def to_display(x, y, scale):
    return int(x * scale), int(y * scale)

def to_original(x, y, scale):
    return int(x / scale), int(y / scale)

# --- State ---
scale = 1.0
danger_x1 = danger_y1 = danger_x2 = danger_y2 = None
yellow_y = None
red_y = None
drawing = False
drag_start = None
stage = "DANGER_ZONE"   # stages: DANGER_ZONE -> YELLOW_LINE -> RED_LINE -> DONE

def reset():
    global danger_x1, danger_y1, danger_x2, danger_y2
    global yellow_y, red_y, drawing, drag_start, stage
    danger_x1 = danger_y1 = danger_x2 = danger_y2 = None
    yellow_y = None
    red_y = None
    drawing = False
    drag_start = None
    stage = "DANGER_ZONE"

def mouse_callback(event, x, y, flags, param):
    global drawing, drag_start, scale
    global danger_x1, danger_y1, danger_x2, danger_y2
    global yellow_y, red_y, stage

    # Convert display coords back to original resolution
    ox, oy = to_original(x, y, scale)

    if stage == "DANGER_ZONE":
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drag_start = (ox, oy)
            danger_x1, danger_y1 = ox, oy
            danger_x2, danger_y2 = ox, oy

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            danger_x2, danger_y2 = ox, oy

        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            danger_x2, danger_y2 = ox, oy
            danger_x1, danger_x2 = min(danger_x1, danger_x2), max(danger_x1, danger_x2)
            danger_y1, danger_y2 = min(danger_y1, danger_y2), max(danger_y1, danger_y2)
            stage = "YELLOW_LINE"

    elif stage == "YELLOW_LINE":
        if event == cv2.EVENT_LBUTTONDOWN:
            yellow_y = oy
            stage = "RED_LINE"

    elif stage == "RED_LINE":
        if event == cv2.EVENT_LBUTTONDOWN:
            red_y = oy
            stage = "DONE"

cv2.namedWindow("Calibrate — Driveway Safety System")
cv2.setMouseCallback("Calibrate — Driveway Safety System", mouse_callback)

# Grab a reference frame (for video files, use frame 0; for camera, use live)
ret, base_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read a frame from the source.")

if args.video:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Compute display scale from the first frame
scale = get_scale(base_frame)
display_w = int(base_frame.shape[1] * scale)
display_h = int(base_frame.shape[0] * scale)

print("\n=== Calibration Tool ===")
print(f"Frame size: {base_frame.shape[1]}x{base_frame.shape[0]}  Display scale: {scale:.2f}")
print("Step 1: Click and drag to draw the DANGER ZONE rectangle over the road")
print("Step 2: Click to set the YELLOW line (caution threshold)")
print("Step 3: Click to set the RED line (high danger threshold)")
print("S = save  |  R = reset  |  Q = quit\n")

while True:
    if args.video:
        frame = base_frame.copy()
    else:
        ret, frame = cap.read()
        if not ret:
            break

    overlay = frame.copy()

    # Instructions overlay
    instructions = {
        "DANGER_ZONE": "Step 1: Click + drag to draw the DANGER ZONE over the road",
        "YELLOW_LINE": "Step 2: Click to place the YELLOW (caution) line",
        "RED_LINE":    "Step 3: Click to place the RED (danger) line",
        "DONE":        "Done! Press S to save, R to reset",
    }
    cv2.putText(overlay, instructions[stage], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Draw danger zone
    if danger_x1 is not None and danger_x2 is not None:
        cv2.rectangle(overlay, (danger_x1, danger_y1), (danger_x2, danger_y2), (255, 0, 0), 2)

    # Draw yellow line
    if yellow_y is not None and danger_x1 is not None:
        cv2.line(overlay, (danger_x1, yellow_y), (danger_x2, yellow_y), (0, 255, 255), 2)
        cv2.putText(overlay, f"YELLOW y={yellow_y}", (danger_x2 + 5, yellow_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw red line
    if red_y is not None and danger_x1 is not None:
        cv2.line(overlay, (danger_x1, red_y), (danger_x2, red_y), (0, 0, 255), 2)
        cv2.putText(overlay, f"RED y={red_y}", (danger_x2 + 5, red_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Coordinate labels on danger zone corners
    if danger_x1 is not None and stage != "DANGER_ZONE":
        cv2.putText(overlay, f"({danger_x1},{danger_y1})", (danger_x1, danger_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(overlay, f"({danger_x2},{danger_y2})", (danger_x2 - 100, danger_y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    display_frame = cv2.resize(overlay, (display_w, display_h))
    cv2.imshow("Calibrate — Driveway Safety System", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit without saving.")
        break

    elif key == ord('r'):
        reset()
        print("Reset. Start over from Step 1.")

    elif key == ord('s'):
        if stage != "DONE":
            print("Not all steps complete yet — finish all 3 steps before saving.")
            continue

        # Validate yellow is above red (lower y = higher on screen)
        if yellow_y > red_y:
            yellow_y, red_y = red_y, yellow_y
            print("(Yellow and red lines were swapped — auto-corrected)")

        print("\n=== CALIBRATION VALUES ===")
        print("Copy these into main.py:\n")
        print(f"danger_x1 = {danger_x1}")
        print(f"danger_y1 = {danger_y1}")
        print(f"danger_x2 = {danger_x2}")
        print(f"danger_y2 = {danger_y2}")
        print(f"yellow_y  = {yellow_y}")
        print(f"red_y     = {red_y}")
        print("\n==========================\n")

        # Also write to a file so values aren't lost if terminal scrolls
        out_path = os.path.join(os.path.dirname(__file__), "reports", "calibration.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("# Driveway calibration values\n")
            f.write(f"danger_x1 = {danger_x1}\n")
            f.write(f"danger_y1 = {danger_y1}\n")
            f.write(f"danger_x2 = {danger_x2}\n")
            f.write(f"danger_y2 = {danger_y2}\n")
            f.write(f"yellow_y  = {yellow_y}\n")
            f.write(f"red_y     = {red_y}\n")
        print(f"Values also saved to: reports/calibration.txt")
        break

cap.release()
cv2.destroyAllWindows()
