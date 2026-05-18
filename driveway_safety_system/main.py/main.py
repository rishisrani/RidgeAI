import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(1)

while True:

    success, frame = camera.read()
    vehicle_detected = False
    results = model(frame)

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in [ "car", "truck", "bus", "motorcycle", "person", "bicycle"]:

                vehicle_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
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

    cv2.imshow("Driveway Vehicle Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()