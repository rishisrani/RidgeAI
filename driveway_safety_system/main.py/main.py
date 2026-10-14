import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from web_server import update_state, start_server

parser = argparse.ArgumentParser(description="Driveway Safety System")
parser.add_argument("--video", type=str, default=None, help="Path to a video file instead of live camera")
parser.add_argument("--camera", type=int, default=None, help="Camera index override (default: auto-detect)")
parser.add_argument("--headless", action="store_true", help="Skip cv2.imshow (no display required)")
parser.add_argument("--test-record", action="store_true", help="Lower risk threshold to 1 so any detection triggers a recording")
parser.add_argument("--skip", type=int, default=2, help="Run YOLO every N frames (default 2). Use 1 to disable skipping.")
args = parser.parse_args()

model = YOLO("yolov8n.pt")

if args.video:
    camera = cv2.VideoCapture(args.video)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open video file: {args.video}")
    print(f"Using video file: {args.video}")
elif args.camera is not None:
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    print(f"Using camera index {args.camera}")
else:
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        print("Camera index 1 not found, trying index 0...")
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No camera found at index 0 or 1. Use --camera N or --video path.mp4")

# Confirm codec support by writing a real test frame and checking file size
_test_path = "reports/_codec_test.mp4"
_test_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
_test_writer = cv2.VideoWriter(_test_path, _test_fourcc, 30, (640, 480))
_test_writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
_test_writer.release()
if os.path.exists(_test_path) and os.path.getsize(_test_path) > 0:
    print("Codec check PASSED: mp4v supported — event videos will save correctly")
    os.remove(_test_path)
else:
    print("WARNING: mp4v codec not supported — event videos may not save correctly")
_test_writer = None

if args.test_record:
    print("TEST-RECORD MODE: risk threshold set to 1 — any detection will trigger a recording")

# Auto-detect headless: if --headless not set, try opening a test window
if not args.headless:
    try:
        cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_test")
    except cv2.error:
        print("No display detected — switching to headless mode automatically")
        args.headless = True

danger_x1 = 200
danger_y1 = 100

danger_x2 = 1000
danger_y2 = 600

yellow_y = 400
red_y = 500

previous_positions = {}
track_history = {}
last_seen_danger_frames = {}
object_memory = {}
velocity_history = {}
detection_confirmations = {}
predicted_tti = {}
future_positions = {}

counted_vehicles = set()
counted_near_misses = set()

heatmap_points = []

traffic_samples = []
risk_history = []

top_events = []

peak_traffic = 0
average_traffic = 0

max_risk_today = 0

highest_risk_timestamp = "None"

peak_traffic = 0
average_traffic = 0

highest_risk_object = None
highest_risk_score = 0

os.makedirs("events", exist_ok=True)
os.makedirs("reports", exist_ok=True)
last_event_time = 0

os.makedirs("videos", exist_ok=True)

last_danger_time = 0
threat_level = "SAFE"
smoothed_risk = 0
safe_frame_count = 0
near_miss_count = 0
danger_event_count = 0
vehicle_count_total = 0
car_count = 0
truck_count = 0
bus_count = 0
motorcycle_count = 0
person_count = 0
bicycle_count = 0
low_events = 0
medium_events = 0
high_events = 0
critical_events = 0
fps = 0
prev_time = time.time()
frame_buffer = deque(maxlen=150)
recording_event = False
recording_frames_left = 0

POST_EVENT_FRAMES = 150
FPS = 30

video_writer = None

frame_height = None
frame_width = None
cooldown_active = False
frame_count = 0
last_results = None
last_infer_scale = 1.0

start_server()

while True:

    success, frame = camera.read()

    if not success:
        break

    if frame_height is None:
        frame_height, frame_width = frame.shape[:2]

    current_fps_time = time.time()

    dt = max(current_fps_time - prev_time, 0.001)
    fps = 1 / dt

    prev_time = current_fps_time

    frame_buffer.append(frame.copy())

    vehicle_detected = False
    object_count = 0
    rejected_detections = 0
    approaching_count = 0
    risk_score = 0
    highest_risk_object = None
    highest_risk_score = 0
    threat_database = {}
    occupied_pixels = 0
    adaptive_threshold = 1 if args.test_record else 8
    occupancy_percent = 0
    traffic_state = "LIGHT"
    frame_count += 1

    if frame_count % args.skip == 0 or last_results is None:
        infer_h, infer_w = frame.shape[:2]
        infer_scale = min(640 / infer_w, 640 / infer_h, 1.0)
        if infer_scale < 1.0:
            infer_frame = cv2.resize(frame, (int(infer_w * infer_scale), int(infer_h * infer_scale)))
        else:
            infer_frame = frame
        last_results = model.track(
            infer_frame,
            persist=True,
            verbose=False
        )
        last_infer_scale = infer_scale

    results = last_results
    decision = "SAFE TO REVERSE"
    

    for k in list(object_memory.keys()):
        object_memory[k] -= 1

        if object_memory[k] <= 0:
            del object_memory[k]

            if k in velocity_history:
                del velocity_history[k]

            if k in track_history:
                del track_history[k]

            if k in previous_positions:
                del previous_positions[k]

            if k in detection_confirmations:
                del detection_confirmations[k]
            
            if k in predicted_tti:
                del predicted_tti[k]
            
            if k in future_positions:
                del future_positions[k]

            if k in counted_vehicles:
                counted_vehicles.remove(k)

            if k in counted_near_misses:
                counted_near_misses.remove(k)


    for k in list(last_seen_danger_frames.keys()):
        last_seen_danger_frames[k] -= 1
        if last_seen_danger_frames[k] <= 0:
            del last_seen_danger_frames[k]

    cv2.rectangle(
                frame,
                (danger_x1, danger_y1),
                (danger_x2, danger_y2),
                (255,0,0),
                 2
    )

    cv2.line(
            frame,
            (danger_x1, yellow_y),
            (danger_x2, yellow_y),
            (0,255,255),
            2
        )

    cv2.line(
            frame,
            (danger_x1, red_y),
            (danger_x2, red_y),
            (0,0,255),
            2
        )

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if confidence < 0.3:
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])
            object_id = track_id
            object_risk = 0
            object_memory[track_id] = 30
            reasons = []


            if track_id not in detection_confirmations:
                detection_confirmations[track_id] = 0

            detection_confirmations[track_id] += 1

            if detection_confirmations[track_id] < 1:
                rejected_detections += 1
                continue

            if track_id not in counted_vehicles:

                counted_vehicles.add(track_id)

                if class_name == "car":
                    car_count += 1

                elif class_name == "truck":
                    truck_count += 1

                elif class_name == "bus":
                    bus_count += 1

                elif class_name == "motorcycle":
                    motorcycle_count += 1

                elif class_name == "person":
                    person_count += 1

                elif class_name == "bicycle":
                    bicycle_count += 1

                if class_name in [
                    "car",
                    "truck",
                    "bus",
                    "motorcycle"
                ]:
                    vehicle_count_total += 1

            if class_name in [ "car", "truck", "bus", "motorcycle", "person", "bicycle"]:

                x1, y1, x2, y2 = (int(v / last_infer_scale) for v in box.xyxy[0])
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                box_height = y2 - y1

                inside_danger_zone = (
                    danger_x1 < center_x < danger_x2
                    and
                    danger_y1 < center_y < danger_y2
                )

                if center_y > red_y:
                    zone = "RED"

                elif center_y > yellow_y:
                    zone = "YELLOW"

                else:
                    zone = "GREEN"


                if inside_danger_zone:
                    vehicle_detected = True
                    object_count += 1

                    heatmap_points.append((center_x, center_y))

                    if len(heatmap_points) > 5000:
                        heatmap_points.pop(0)

                    last_seen_danger_frames[track_id] = 30

                    if class_name in ["car", "truck", "bus"]:
                        risk_score += 2
                        object_risk += 2
            
                    elif class_name == "motorcycle":
                        risk_score += 2
                        object_risk += 2
            
                    elif class_name in ["person", "bicycle"]:
                        risk_score += 1
                        object_risk += 1

                    if box_height > 250:
                        risk_score += 3
                        object_risk += 3
                        reasons.append("Very Close Vehicle")

                    elif box_height > 180:
                        risk_score += 1
                        object_risk += 1
                        reasons.append("Close Vehicle")


                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append(
                    (center_x, center_y)
                )

                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                
                points = track_history[track_id]

                for i in range(1, len(points)):
                    cv2.line(
                        frame,
                        points[i-1],
                        points[i],
                        (0,255,255),
                        2
                    )

                track_age = len(track_history[track_id])

                current_position = (center_x, center_y)

                if object_id in previous_positions:
                    
                    prev_x, prev_y = previous_positions[object_id]
                    dx = center_x - prev_x
                    dy = center_y - prev_y

                    ema_alpha = 0.3
                    if object_id not in velocity_history:
                        velocity_history[object_id] = (float(dx), float(dy))

                    prev_avg_dx, prev_avg_dy = velocity_history[object_id]
                    avg_dx = ema_alpha * dx + (1 - ema_alpha) * prev_avg_dx
                    avg_dy = ema_alpha * dy + (1 - ema_alpha) * prev_avg_dy
                    velocity_history[object_id] = (avg_dx, avg_dy)

                    movement = (avg_dx**2 + avg_dy**2) ** 0.5

                    if movement < 10:
                        speed_class = "SLOW"

                    elif movement < 30:
                        speed_class = "MEDIUM"

                    else:
                        speed_class = "FAST"

                    approaching = avg_dy > 0

                    if abs(avg_dx) > abs(avg_dy):

                        if avg_dx > 0:
                            direction = "RIGHT"

                        else:
                            direction = "LEFT"

                    else:

                        if avg_dy > 0:
                            direction = "APPROACHING"

                        else:
                            direction = "LEAVING"

                    distance_to_bottom = max(0, danger_y2 - y2)

                    if avg_dy > 1 and fps > 0:

                        relative_speed = avg_dy * (box_height / 100)

                        speed_pixels_per_second = relative_speed * fps

                        tti_seconds = (
                            distance_to_bottom
                            /
                            speed_pixels_per_second
                        )

                        predicted_tti[track_id] = tti_seconds

                    else:
                        predicted_tti[track_id] = None

                    future_x = center_x + (avg_dx * 10)
                    future_y = center_y + (avg_dy * 10)

                    future_positions[track_id] = (
                        int(future_x),
                        int(future_y)
                    )    

                    is_stationary = movement < 3

                    if inside_danger_zone and is_stationary:
                        risk_score += 2
                        object_risk += 2

                    if inside_danger_zone:
                        
                        if movement > 20:
                            risk_score += 1
                            object_risk += 1

                        if movement > 50:
                            risk_score += 2
                            object_risk += 2

                        if track_id in future_positions:

                                fx, fy = future_positions[track_id]

                                # Check every point along the predicted path, not just the endpoint
                                path_enters_red = False
                                path_enters_danger = False
                                for step in range(1, 11):
                                    t = step / 10
                                    px = int(center_x + t * (fx - center_x))
                                    py = int(center_y + t * (fy - center_y))
                                    if danger_x1 < px < danger_x2 and red_y < py < danger_y2:
                                        path_enters_red = True
                                    if danger_x1 < px < danger_x2 and danger_y1 < py < danger_y2:
                                        path_enters_danger = True

                                if path_enters_red:
                                    risk_score += 2
                                    object_risk += 2
                                    reasons.append("Predicted Red Zone Entry")
                                elif path_enters_danger and not inside_danger_zone:
                                    risk_score += 1
                                    object_risk += 1
                                    reasons.append("Predicted Danger Zone Entry")

                        if zone == "RED":
                            risk_score += 4
                            object_risk += 4
                            reasons.append("Red Zone")

                            if approaching:
                                risk_score += 3
                                object_risk += 3
                                reasons.append("Approaching in Red Zone")

                        elif zone == "YELLOW":
                            risk_score += 2
                            object_risk += 2
                            reasons.append("Yellow Zone")
                        else:
                            risk_score += 1
                            object_risk += 1

                        box_area = (x2 - x1) * (y2 - y1)
                        occupied_pixels += box_area



                    if predicted_tti.get(track_id) is not None:

                        if predicted_tti[track_id] < 3:
                            risk_score += 4
                            object_risk += 4
                            reasons.append("TTI < 3s")

                        elif predicted_tti[track_id] < 6:
                            risk_score += 2
                            object_risk += 2
                            reasons.append("TTI < 6s")

                    print(
                         f"ID={track_id} "
                        f"Risk={object_risk} "
                        f"Zone={zone}"
                    )

                    cv2.putText(
                        frame,
                        f"dy:{dy}",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,255,0),
                        2
                    )

                    cv2.putText(
                        frame,
                        zone,
                        (x1, y1 - 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2
                    )

                    cv2.putText(
                        frame,
                        direction,
                        (x1, y1 - 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,0),
                        2
                    )

                    cv2.putText(
                        frame,
                        speed_class,
                        (x1, y1 - 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2
                    )

                    if movement > 10:

                        cv2.putText(
                            frame,
                            f"Moving {movement:.1f}px",
                            (x1, y2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,255),
                            2
                        )
                    
                    if predicted_tti.get(track_id) is not None:

                        cv2.putText(
                            frame,
                            f"TTI:{predicted_tti[track_id]:.1f}",
                            (x1, y2 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2
                        )
                        
                    if approaching:

                        approaching_count += 1
                        risk_score += 2
                        object_risk += 2
                        reasons.append("Approaching Motion")
                        cv2.putText(
                        frame,
                        "APPROACHING",
                        (x1, y2 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2
                        )

                    else:

                        cv2.putText(
                            frame,
                            "LEAVING",
                            (x1, y2 + 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2
                        )

                    threat_database[track_id] = {
                            "class": class_name,
                            "risk": object_risk,
                            "zone": zone,
                            "tti": predicted_tti.get(track_id),
                            "reasons": reasons
                        }

                    if object_risk > highest_risk_score:
                        highest_risk_score = object_risk
                        highest_risk_object = track_id

                    if (
                        object_risk >= 10
                        and approaching
                        and zone == "RED"
                        and track_id not in counted_near_misses
                    ):
                            counted_near_misses.add(track_id)
                            near_miss_count += 1                        

                    cv2.arrowedLine(
                        frame,
                        (prev_x, prev_y),
                        (center_x, center_y),
                        (0,255,255),
                        2
                    )

                    if approaching and movement > 50 and len(track_history[track_id]) > 5:
                        threat_level = "DANGER"
                        last_danger_time = time.time()

                    elif approaching and movement > 20 and threat_level != "DANGER":
                        threat_level = "CAUTION"

                previous_positions[object_id] = current_position

                
                color = (0,255,0)

                if track_id == highest_risk_object:
                    color = (0,0,255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )


                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0,0,255),
                    -1
                )


                if track_id in future_positions:

                    fx, fy = future_positions[track_id]

                    cv2.circle(
                        frame,
                        (fx, fy),
                        8,
                        (255,0,255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "PREDICTED",
                        (x1, y1 - 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,0,255),
                        2
                    )

                    cv2.line(
                        frame,
                        (center_x, center_y),
                        (fx, fy),
                        (255,0,255),
                        2
                    )

                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Age:{track_age}",
                    (x1, y1 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )


    current_time = time.time()
    cooldown_active = (current_time - last_danger_time < 5)
    if cooldown_active:
        threat_level = "DANGER"
    elif threat_level != "DANGER":
        threat_level = "SAFE"
    active_threat = (
        threat_level == "DANGER"
        and len(object_memory) > 0
    )

    smoothed_risk = (smoothed_risk * 0.8) + (risk_score * 0.2)

    risk_history.append(
        (
            time.time(),
            smoothed_risk
        )
    )

    if len(risk_history) > 5000:
        risk_history.pop(0)

    if risk_score > max_risk_today:
        max_risk_today = risk_score

        highest_risk_timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


    if risk_score >= adaptive_threshold or cooldown_active or active_threat:
        decision = "WAIT FOR TRAFFIC"

        if recording_event:
            recording_frames_left = POST_EVENT_FRAMES

        if not recording_event:

            recording_event = True
            recording_frames_left = POST_EVENT_FRAMES

            danger_event_count += 1

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            video_name = f"videos/event_{timestamp}.mp4"

            height, width = frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            video_writer = cv2.VideoWriter(
                video_name,
                fourcc,
                FPS,
                (width, height)
            )

            print(f"Recording video: {video_name}")
            print(f"Buffered frames: {len(frame_buffer)}")
            print("Video writer opened:", video_writer.isOpened())

            for buffered_frame in frame_buffer:
                video_writer.write(buffered_frame)

        if current_time - last_event_time > 10:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"events/event_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            if risk_score >= 20:
                severity = "CRITICAL"

            elif risk_score >= 12:
                severity = "HIGH"

            elif risk_score >= 8:
                severity = "MEDIUM"

            else:
                severity = "LOW"
            

            if severity == "LOW":
                low_events += 1

            elif severity == "MEDIUM":
                medium_events += 1

            elif severity == "HIGH":
                high_events += 1

            elif severity == "CRITICAL":
                critical_events += 1

            top_events.append({
                "risk": risk_score,
                "time": timestamp,
                "severity": severity
            })

            if len(top_events) > 100:
                top_events.pop(0)


            log_path = "reports/driveway_log.txt"
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log_lines = f.readlines()
                if len(log_lines) >= 1000:
                    with open(log_path, "w") as f:
                        f.writelines(log_lines[-999:])
            with open(log_path, "a") as log:
                if (
                    highest_risk_object is not None
                    and highest_risk_object in threat_database
                ):
                    threat = threat_database[highest_risk_object]
                    log.write(
                        f"{timestamp} | "
                        f"Decision={decision} | "
                        f"Severity={severity} | "
                        f"Class={threat['class']} | "
                        f"Risk={threat['risk']} | "
                        f"Zone={threat['zone']} | "
                        f"TTI={threat['tti']} | "
                        f"Reasons={','.join(threat['reasons'])}\n"
                    )

            last_event_time = current_time


    elif risk_score >= 4:
        decision = "CAUTION"
    else:
        decision = "SAFE TO REVERSE"

    if risk_score < 4:
        safe_frame_count += 1
    else:
        safe_frame_count = 0

    if decision == "WAIT FOR TRAFFIC":
        overlay_text = "UNSAFE TO REVERSE"
        overlay_color = (0, 0, 255)
    elif decision == "CAUTION":
        overlay_text = "CAUTION"
        overlay_color = (0, 165, 255)
    else:
        overlay_text = "SAFE TO REVERSE"
        overlay_color = (0, 255, 0)

    cv2.putText(
        frame,
        overlay_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        overlay_color,
        3
    )

    if occupancy_percent > 40:
        traffic_state = "HEAVY"
    elif occupancy_percent > 15:
        traffic_state = "MODERATE"
    else:
        traffic_state = "LIGHT"

    cv2.putText(
        frame,
        f"Traffic: {traffic_state}",
        (50, 850),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    danger_zone_area = (
        (danger_x2 - danger_x1)
        *
        (danger_y2 - danger_y1)
    )

    occupancy_percent = (
        occupied_pixels
        /
        danger_zone_area
    ) * 100

    occupancy_percent = min(
       occupancy_percent,
        100
    )

    if occupancy_percent > 40:
        adaptive_threshold = 6

    elif occupancy_percent < 10:
        adaptive_threshold = 10

    traffic_samples.append(occupancy_percent)

    if len(traffic_samples) > 5000:
        traffic_samples.pop(0)

    if len(traffic_samples) > 0:

        peak_traffic = max(traffic_samples)

        average_traffic = (
            sum(traffic_samples)
            /
            len(traffic_samples)
        )

    cv2.putText(
        frame,
        f"Occupancy: {occupancy_percent:.1f}%",
        (50,900),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Rejected: {rejected_detections}",
        (50, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (50,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Threat: {threat_level}",
        (50,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )
    
    cv2.putText(
        frame,
        f"Approaching: {approaching_count}",
        (50,200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Tracked: {len(previous_positions)}",
        (50,250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Decision: {decision}",
        (50,300),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Top Threat: {highest_risk_object} ({highest_risk_score})",
        (50, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )


    if (
        highest_risk_object is not None
        and highest_risk_object in threat_database
    ):

        threat = threat_database[highest_risk_object]

        cv2.putText(
            frame,
            f"Primary: {threat['class']}",
            (50, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            f"Risk: {threat['risk']}",
            (50, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        if threat["tti"] is not None:

            cv2.putText(
                frame,
                f"TTI: {threat['tti']:.1f}s",
                (50, 600),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,165,255),
                2
            )

        if len(threat["reasons"]) > 0:

            cv2.putText(
                frame,
                threat["reasons"][0],
                (50, 550),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

        cv2.putText(
            frame,
            f"Threshold: {adaptive_threshold}",
            (50,950),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (50, 1100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Peak Traffic: {peak_traffic:.1f}%",
            (50,1000),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Avg Traffic: {average_traffic:.1f}%",
            (50,1050),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Vehicles: {vehicle_count_total}",
            (50,700),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Danger Events: {danger_event_count}",
            (50,750),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            f"Near Misses: {near_miss_count}",
            (50,800),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,165,255),
            2
        )


        cv2.putText(
            frame,
            f"Risk Score: {smoothed_risk:.1f}",
            (50, 650),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )


    if recording_event and video_writer is not None:

        video_writer.write(frame)

        recording_frames_left -= 1

        if recording_frames_left <= 0:

            video_writer.release()

            recording_event = False
            video_writer = None    

    if not args.headless:
        display_h, display_w = frame.shape[:2]
        scale = min(1280 / display_w, 720 / display_h, 1.0)
        if scale < 1.0:
            display_frame = cv2.resize(frame, (int(display_w * scale), int(display_h * scale)))
        else:
            display_frame = frame
        cv2.imshow("Driveway Vehicle Detection", display_frame)

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    update_state(
        frame_jpeg=jpeg.tobytes(),
        decision=decision,
        threat_level=threat_level,
        risk_score=risk_score,
        smoothed_risk=smoothed_risk,
        object_count=object_count,
        approaching_count=approaching_count,
        vehicle_count_total=vehicle_count_total,
        near_miss_count=near_miss_count,
        danger_event_count=danger_event_count,
        highest_risk_score=highest_risk_score,
        highest_risk_object=highest_risk_object,
        occupancy_percent=occupancy_percent,
        traffic_state=traffic_state,
    )

    if not args.headless and cv2.waitKey(1) == ord('q'):
        break

if video_writer is not None:
    video_writer.release()


average_risk = 0

if len(risk_history) > 0:

    average_risk = (
    sum(r[1] for r in risk_history)
    /
    len(risk_history)
    )   

with open("reports/daily_stats.txt", "w") as stats:

    report_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    stats.write(
        f"Report Generated: {report_time}\n"
    )

    stats.write(
        "============================\n"
    )

    stats.write(
        f"Vehicles: {vehicle_count_total}\n"
    )

    stats.write(
        f"Cars: {car_count}\n"
    )

    stats.write(
        f"Trucks: {truck_count}\n"
    )

    stats.write(
        f"Buses: {bus_count}\n"
    )

    stats.write(
        f"Motorcycles: {motorcycle_count}\n"
    )

    stats.write(
        f"People: {person_count}\n"
    )

    stats.write(
        f"Bicycles: {bicycle_count}\n"
    )

    stats.write(
        f"Danger Events: {danger_event_count}\n"
    )

    stats.write(
        f"Near Misses: {near_miss_count}\n"
    )

    stats.write(
        f"Low Events: {low_events}\n"
    )

    stats.write(
        f"Medium Events: {medium_events}\n"
    )

    stats.write(
        f"High Events: {high_events}\n"
    )

    stats.write(
        f"Critical Events: {critical_events}\n"
    )

    stats.write(
        f"Average Risk: {average_risk:.2f}\n"
    )

    stats.write(
        f"Max Risk: {max_risk_today}\n"
    )

    stats.write(
        f"Highest Risk Time: {highest_risk_timestamp}\n"
    )

    stats.write(
        f"Peak Traffic: {peak_traffic:.1f}%\n"
    )

    stats.write("\nTop Events\n")
    stats.write("===========\n")

    top_events.sort(
        key=lambda x: x["risk"],
        reverse=True
    )

    for event in top_events[:5]:

        stats.write(
            f"{event['time']} | "
            f"Risk={event['risk']} | "
            f"{event['severity']}\n"
        )

    stats.write(
        f"Heatmap Points: {len(heatmap_points)}\n"
    )


if len(heatmap_points) > 0:

    if frame_height is None or frame_width is None:
        print("No frame dimensions available. Skipping heatmap.")
    else:
        heatmap = np.zeros(
            (frame_height, frame_width),
            dtype=np.uint8
        )

        for x, y in heatmap_points:

            cv2.circle(
                heatmap,
                (x, y),
                20,
                10,
                -1
            )

        heatmap = cv2.GaussianBlur(
            heatmap,
            (51,51),
            0
        )

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        cv2.imwrite(
            "reports/heatmap.png",
            heatmap_color
        )

camera.release()
cv2.destroyAllWindows()

if len(risk_history) > 0:

    plt.figure(figsize=(10,5))

    risk_values = [r[1] for r in risk_history]

    plt.plot(risk_values)

    plt.title("Risk Score Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Risk")

    plt.savefig("reports/risk_graph.png")

    plt.close()

if len(traffic_samples) > 0:

    plt.figure(figsize=(10,5))

    plt.plot(traffic_samples)

    plt.title("Traffic Occupancy")
    plt.xlabel("Frame")
    plt.ylabel("Occupancy %")

    plt.savefig("reports/traffic_graph.png")

    plt.close()

event_counts = [
    low_events,
    medium_events,
    high_events,
    critical_events
]

labels = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

plt.figure(figsize=(8,5))
plt.bar(labels, event_counts)
plt.title("Event Severity Distribution")
plt.ylabel("Count")
plt.savefig("reports/event_distribution.png")
plt.close()    