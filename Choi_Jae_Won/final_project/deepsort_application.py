import datetime
from ultralytics import YOLO
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)

# 비디오 캡처 객체 초기화
video_cap = cv2.VideoCapture("detectfile.mp4")  # 'traffic.mp4' 경로에 맞게 수정하세요

# YOLOv8 모델 로드
model = YOLO(r'C:\Users\iot19\Documents\final_project\yolov8\weights\best.pt')

# GPU 사용 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # 모델을 GPU로 이동

# DeepSort 트래커 초기화
tracker = DeepSort(max_age=0)  # 추적할 객체가 일정 시간 감지되지 않으면 추적을 종료하도록 max_age 설정

# 기본적으로 YOLO의 출력 로그를 끄려면, 다음과 같이 로깅 설정을 변경합니다.
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # 프레임 크기 조정 (최적화)
    frame_resized = cv2.resize(frame, (640, 360))  # 크기 축소하여 처리 속도 개선

    # YOLO 모델을 이용해 객체 탐지
    detections = model(frame_resized, device=device)[0]  # GPU로 처리하도록 변경

    # 바운딩 박스와 신뢰도 초기화
    results = []

    ######################################
    # 객체 탐지 (DETECTION)
    ######################################

    for data in detections.boxes.data.tolist():
        confidence = data[4]

        # 신뢰도가 최소 신뢰도보다 낮은 탐지는 필터링
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # 신뢰도가 충분히 높으면 바운딩 박스 좌표와 클래스 ID 추출
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        
        # 바운딩 박스, 신뢰도 및 클래스 ID를 결과 리스트에 추가
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    ######################################
    # 객체 추적 (TRACKING)
    ######################################

    if len(results) > 0:  # 객체가 탐지되었으면
        # 새로 탐지된 객체를 이용해 트래커 업데이트
        tracks = tracker.update_tracks(results, frame=frame)
    else:
        # 객체가 탐지되지 않으면 추적 목록에서 추적을 종료
        tracks = []

    # 추적된 객체에 대해 반복문을 돌며 표시
    for track in tracks:
        # 추적이 확정되지 않은 객체는 무시
        if not track.is_confirmed():
            continue

        # 추적 ID와 바운딩 박스를 가져오기
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # 추적된 객체에 바운딩 박스를 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        
        # 추적 ID를 표시
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # 처리된 프레임을 화면에 표시
    cv2.imshow("Detected Vehicles with Tracking", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체 해제 및 창 닫기
video_cap.release()
cv2.destroyAllWindows()
