import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    cv2.putText(
        frame,
        "Monitoring Driveway",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Driveway Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()