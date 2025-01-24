import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO(r'C:\Users\iot19\Documents\final_project\yolov8\weights\best.pt')

# 비디오 파일 열기
cap = cv2.VideoCapture('traffic.mp4')  # 동영상 파일 경로를 지정하세요

# 비디오 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지
    results = model(frame)

    # 첫 번째 결과에서 박스와 클래스 정보를 추출
    boxes = results[0].boxes
    names = results[0].names

    # 'car' 클래스만 필터링
    for box in boxes:
        class_id = int(box.cls)  # 클래스 ID
        class_name = names[class_id]  # 클래스 이름
        if class_name == 'car':  # 차량만 처리
            x1, y1, x2, y2 = box.xyxy[0]  # 좌표 추출

            # 차량에 사각형 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 'car' 텍스트를 각 차량 위에 표시
            cv2.putText(frame, 'car', 
                        (int(x1), int(y1) - 10),  # 텍스트 위치 (상자의 위쪽에 텍스트 배치)
                        cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
                        0.9,  # 폰트 크기
                        (0, 255, 0),  # 텍스트 색 (초록색)
                        2)  # 선 두께

    # 결과 영상 표시
    cv2.imshow('Detected Vehicles', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 객체 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
