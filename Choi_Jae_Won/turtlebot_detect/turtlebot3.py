from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드 (사전 학습된 모델)
model = YOLO('./best.pt')  # 사전 학습된 모델 파일 경로 (best.pt)

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 웹캠 장치 번호 (0은 기본 웹캠)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 객체 감지 실행
    results = model(frame)

    # 감지된 객체들에 대해 바운딩 박스를 그리기
    for result in results[0].boxes:  # results[0].boxes는 감지된 객체들의 정보
        x1, y1, x2, y2 = result.xyxy[0]  # 각 객체의 좌표 (x1, y1, x2, y2)
        confidence = result.conf[0]  # 객체에 대한 신뢰도

        # 바운딩 박스 그리기 (직접 그리기)
        color = (0, 255, 0)  # 바운딩 박스 색상 (초록색)
        thickness = 2  # 선의 두께
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # 신뢰도 표시
        label = f"{confidence:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 결과 이미지 화면에 표시
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
