import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(1)

danger_x1 = 200
danger_y1 = 100

danger_x2 = 1000
danger_y2 = 600

previous_positions = {}
track_history = {}

while True:

    success, frame = camera.read()
    if not success:
        break
    vehicle_detected = False
    object_count = 0
    approaching_count = 0
    risk_score = 0
    threat_level = "SAFE"
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )
    decision = "SAFE TO REVERSE"

    cv2.rectangle(
                frame,
                (danger_x1, danger_y1),
                (danger_x2, danger_y2),
                (255,0,0),
                 2
    )

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if confidence < 0.5:
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            if class_name in [ "car", "truck", "bus", "motorcycle", "person", "bicycle"]:
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                inside_danger_zone = (
                    danger_x1 < center_x < danger_x2
                    and
                    danger_y1 < center_y < danger_y2
                )

                if inside_danger_zone:
                    vehicle_detected = True
                    object_count += 1

                    if class_name in ["car", "truck", "bus"]:
                        risk_score += 2
            
                    elif class_name == "motorcycle":
                        risk_score += 2
            
                    elif class_name in ["person", "bicycle"]:
                        risk_score += 1

                object_id = track_id

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

                    movement = (dx**2 + dy**2) ** 0.5

                    if inside_danger_zone:
                        
                        if movement > 20:
                            risk_score += 1

                        if movement > 50:
                            risk_score += 2

                    approaching = dy > 0

                    cv2.putText(
                        frame,
                        f"dy:{dy}",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,255,0),
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
                        
                    if approaching:

                        approaching_count += 1
                        risk_score += 2

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

                    cv2.arrowedLine(
                        frame,
                        (prev_x, prev_y),
                        (center_x, center_y),
                        (0,255,255),
                        2
                    )



                    if approaching and movement > 50 and len(track_history[track_id]) > 5:
                        threat_level = "DANGER"

                    elif approaching and movement > 20 and threat_level != "DANGER":
                        threat_level = "CAUTION"

                previous_positions[object_id] = current_position

                
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0,0,255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"{confidence:.2f}",
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

    if risk_score >= 8:
        decision = "WAIT FOR TRAFFIC"
    elif risk_score >= 4:
        decision = "CAUTION"
    else:
        decision = "SAFE TO REVERSE"

    if vehicle_detected:

        cv2.putText(
            frame,
            "UNSAFE TO REVERSE",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

    else:

        cv2.putText(
            frame,
            "SAFE",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
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

    cv2.imshow("Driveway Vehicle Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()