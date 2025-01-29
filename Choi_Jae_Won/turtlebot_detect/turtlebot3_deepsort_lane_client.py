import cv2
import numpy as np
import math
from ultralytics import YOLO
import cvzone
from sort import Sort
from lane_detection import lane_detection  # lane_detection 모듈 불러오기

# 이미지 파일 경로
image_path = "image/turtlebot_lane.jpg"

# YOLO 모델 로딩
model = YOLO("./best.pt")

# 클래스 이름 설정 (차량만)
classNames = ["vehicle"]

# 이미지 로드
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Couldn't read the image file {image_path}")
    exit()
frame_height, frame_width = img.shape[:2]

# SORT 추적기 초기화
tracker = Sort(max_age=100, min_hits=2, iou_threshold=0.3)

# 이미지 처리 루프
def process_image(img):
    # YOLO 모델을 통해 객체 감지
    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # 차량만 탐지
            if cls < len(classNames) and classNames[cls] == 'vehicle' and conf > 0.4:
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

    # SORT 추적기 업데이트
    tracked_objects = tracker.update(detections)

    # 차선 감지 호출
    lane_image, yellow_points, white_points, blue_points = lane_detection(img)

    # 노란색 선의 기울기 계산
    print(f"yellow_points : {yellow_points}")
    if len(yellow_points[0]) == 2:  # yellow_points의 첫 번째 리스트에서 두 점을 확인
        # 노란색 차선의 두 점을 가져오기
        x1, y1 = yellow_points[0][0]
        x2, y2 = yellow_points[0][1]

        print(f"x1:{x1} ,y1:{y1}, x2:{x2}, y2:{y2}")

        # 기울기 계산 (slope = (y2 - y1) / (x2 - x1))
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1  # y = mx + b 형태로 절편 계산
        else:
            slope = float('inf')  # 수직선인 경우 무한 기울기
            intercept = None

        # 차선의 기울기 출력 (디버깅)
        print(f"Yellow lane slope: {slope}, intercept: {intercept}")
    else:
        slope = None
        intercept = None


    # 추적 결과와 차선 정보를 합쳐서 표시
    for obj in tracked_objects:
        x1, y1, x2, y2, id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        vehicle_id = int(id)
        
        # 차량 하단의 중심 계산
        vehicle_center = ((x1 + x2) // 2, y2)
        cv2.circle(lane_image, vehicle_center, 2, (0,255,0), 2)

        # 디버깅을 위한 출력 추가
        print(f"Vehicle ID: {vehicle_id}")
        print(f"Vehicle Center: {vehicle_center}")
        
        # 차량이 차선 왼쪽/오른쪽에 있는지 판단 (선분을 기준으로 계산)
        if slope is not None and slope != float('inf'):
            # 차량 중심이 차선과 비교해서 1차선, 2차선 판단
            # y = mx + b 형태에서 기울기 m과 절편 b를 사용하여 차량의 x좌표에 대응하는 y값 계산
            expected_y = slope * vehicle_center[0] + intercept  # 직선 상에서의 y값
            print(f"Expected Y (yellow lane): {expected_y}, Vehicle Y: {vehicle_center[1]}")

            # 차량의 하단 중심점이 차선 위에 있는지, 아래에 있는지 판단
            if vehicle_center[1] < expected_y:
                lane = 1  # 1차선
                color = (0, 255, 0)  # Green
            else:
                lane = 2  # 2차선
                color = (0, 0, 255)  # Red
        else:
            # 수직선인 경우 (차선이 수직일 경우)
            if vehicle_center[0] < x1:  # 왼쪽이면 1차선
                lane = 1
                color = (0, 255, 0)
            else:  # 오른쪽이면 2차선
                lane = 2
                color = (0, 0, 255)


        # 차량의 사각형 그리기 및 차선 번호 표시
        cv2.rectangle(lane_image, (x1, y1), (x2, y2), color, 2)
        cvzone.putTextRect(lane_image, f'ID: {vehicle_id} Lane: {lane}', 
                           (x1, y1 - 10), scale=1, thickness=2, offset=1, colorT=(255, 255, 255))

    # 결과 이미지 표시
    cv2.imshow("Lane and Vehicle Tracking", lane_image)
    cv2.waitKey(0)  # 키를 누를 때까지 대기

# 메인 함수
if __name__ == "__main__":
    process_image(img)
    cv2.destroyAllWindows()
