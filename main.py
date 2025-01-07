from ultralytics import YOLO
import cv2 as cv
import math

video = cv.VideoCapture("videos/output.avi")
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

model = YOLO("weights/best.pt")

classNames = ['1', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
              'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

out = cv.VideoWriter("videos/finished.avi", cv.VideoWriter_fourcc(*'MJPG'), 20, size)

while True: 
    ret, frame = video.read()
    if not ret:
        break

    results = model.predict(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ## Indice de confiabilidade (quanto maior maior será a certeza)
                conf = math.ceil(box.conf[0] * 100) / 100 if hasattr(box, 'conf') else 0
                cls = int(box.cls[0]) if hasattr(box, 'cls') else -1

                if 0 <= x1 < frame_width and 0 <= y1 < frame_height and conf > 0.5:
                    currentClass = classNames[cls] if cls < len(classNames) else "Unknown"
                    print(currentClass)

                    color = (255, 0, 0) if cls < 7 else (0, 255, 0)
                    cv.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    out.write(frame)
    cv.imshow("Result", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv.destroyAllWindows()
print("The video was successfully saved")
