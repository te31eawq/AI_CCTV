import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import cvzone
from sort import Sort

cap = cv2.VideoCapture("C:/Users/iot19/Documents/YOLOv8-DeepSort-Object-Detection-and-Tracking/assets/Videos/traffic.mp4")

model = YOLO("yolov8m.pt")  # YOLOv8 모델 로드

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# 마스크 로딩
mask = cv2.imread("c:/Users/iot19/Documents/YOLOv8-DeepSort-Object-Detection-and-Tracking/assets/masktest3.png")

# 추적기 초기화 (SORT)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# 차량 통과 범위 설정 (상단, 하단)
limitsUp = [250, 400, 575, 400]
limitsDown = [700, 450, 1125, 450]

# 차량 카운트 초기화
total_countsUp = {"car":[], "bus":[], "truck":[]}
total_countsDown = {"car":[], "bus":[], "truck":[]}

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break  # 비디오 끝

    # 마스크 처리
    imgRegion = cv2.bitwise_and(img, mask)

    # YOLO 모델을 통해 객체 감지
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box 좌표
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence 및 클래스 추출
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            # 차량만 감지 (car, bus, truck, motorbike)
            if currentclass in ['car', 'bus', 'truck'] and conf > 0.4:
                # 감지된 객체에 대해 사각형과 텍스트 표시
                cvzone.cornerRect(img, (x1, y1, w, h), l=15)
                cvzone.putTextRect(img, f'{currentclass}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=2, offset=3)

                # 객체 정보 배열에 추가
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

    # 차량 통과 범위 (라인 그리기)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), color=(0, 0, 0), thickness=5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), color=(0, 0, 0), thickness=5)

    # 추적기 업데이트
    resultracker = tracker.update(detections)
    for results in resultracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # 추적된 객체에 원 표시
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # 상단 라인 통과 카운트
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[3] + 15:
            total_countsUp[currentclass].append(id)

        # 하단 라인 통과 카운트
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[3] + 15:
            total_countsDown[currentclass].append(id)

    # 상단 및 하단 차량 카운트 출력
    cvzone.putTextRect(img, f'car_counts: {len(set(total_countsUp["car"]))}', (50, 50), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))
    cvzone.putTextRect(img, f'bus_counts: {len(set(total_countsUp["bus"]))}', (50, 80), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))
    cvzone.putTextRect(img, f'truck_counts: {len(set(total_countsUp["truck"]))}', (50, 110), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))

    cvzone.putTextRect(img, f'car_counts: {len(set(total_countsDown["car"]))}', (1000, 50), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))
    cvzone.putTextRect(img, f'bus_counts: {len(set(total_countsDown["bus"]))}', (1000, 80), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))
    cvzone.putTextRect(img, f'truck_counts: {len(set(total_countsDown["truck"]))}', (1000, 110), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))

    # 화면에 이미지 출력
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()
