import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import Sort
import time
from scipy.spatial.distance import cdist

# 비디오 캡쳐 및 첫 번째 프레임 가져오기
cap = cv2.VideoCapture("c:/Users/iot19/Documents/YOLOv8-DeepSort-Object-Detection-and-Tracking/assets/Videos/detectfile.mp4")
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# YOLO 모델 로딩
model = YOLO(r'c:/Users/iot19/Documents/YOLOv8-DeepSort-Object-Detection-and-Tracking/src/best.pt')

# 클래스 이름을 하나로 설정 (차량만)
classNames = ["vehicle"]

# 마스크 로딩
mask = cv2.imread("c:/Users/iot19/Documents/YOLOv8-DeepSort-Object-Detection-and-Tracking/assets/masktest3.png")
if mask is None:
    print("Error: mask image not found")
    exit()

# 비디오 프레임의 크기 가져오기
success, img = cap.read()
if not success:
    print("Error: Couldn't read the first frame of the video")
    exit()
frame_height, frame_width = img.shape[:2]

# 마스크 이미지 크기 조정
mask_resized = cv2.resize(mask, (frame_width, frame_height))

# 추적기 초기화
tracker = Sort(max_age=200, min_hits=2, iou_threshold=0.3)

# 차량 번호가 1에서 30까지 순차적으로 사용되도록 하는 변수
id_counter = 1
id_map = {}

# 차량 추적 정보 저장
tracked_vehicles = {}

# 두 점 사이의 유클리안 거리 계산 함수
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# 이동 거리 필터링 함수 (수정된 버전)
def apply_distance_filter(distance, min_distance=0.9):
    """ 최소 이동 거리를 설정하여 너무 작은 값을 필터링 """
    if distance < min_distance:
        return 0  # 너무 작은 이동은 0으로 설정
    return distance

def adjust_speed_based_on_distance(x, y, frame_width, frame_height, min_distance=1.0, max_distance=3.0):
    """ x, y 좌표를 기반으로 객체의 상대적 거리를 추정하고 속도를 보정 """
    
    # y 위치가 작을수록, 즉 상단에 있을수록 더 큰 보정을 제공
    y_factor = y / frame_height  # 상단에 가까울수록 y_factor가 커짐
    x_factor = x / frame_width   # 왼쪽에 가까울수록 x_factor가 커짐

    # 화면 상에서의 중심으로부터 얼마나 벗어나있는지에 대한 값을 고려
    center_x_factor = abs(x - frame_width // 2) / (frame_width // 2)
    center_y_factor = abs(y - frame_height // 2) / (frame_height // 2)
    
    # 각 축의 비율을 합산하여 전체적인 보정 값을 계산
    distance_factor = 0.5 * (x_factor + y_factor) + 0.5 * (center_x_factor + center_y_factor)
    
    # 계산된 값은 0과 1 사이로 제한될 수 있도록 조정
    distance_factor = np.clip(distance_factor, min_distance, max_distance)
    
    return distance_factor


# 차량이 멈추었을 때 빨리 반응하도록 'zero_speed_count'를 수정
def adjust_zero_speed_threshold(threshold=2):
    """ 차량이 멈춘 상태에서 빨리 반응하도록 'zero_speed_count'를 수정 """
    return threshold  # 이 값을 조정하여 반응 속도를 더 민감하게 만들 수 있습니다.

# 픽셀당 실제 거리를 정의 (예: 1 픽셀 = 0.1 미터)
scale = 0.5  # scale 값을 더 큰 값으로 설정하여 속도 계산을 더 크게

# 마지막 평균 속도 추적 변수
last_avg_speed = 0  # 마지막 평균 속도

# 프레임 수를 기준으로 속도 계산 (이전 n 프레임의 평균 속도 계산)
frame_buffer = {}  # 차량별로 속도를 저장할 버퍼
buffer_size = 5  # 속도 계산에 사용할 프레임 수

# 비디오 처리 루프
while True:
    success, img = cap.read()
    if not success:
        break  # 비디오 끝

    # 현재 비디오의 시간 (밀리초 단위)
    current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    # 첫 2초 동안은 속도 계산을 건너뜀
    if current_time_ms < 2000:
        cv2.putText(img, 'Skipping speed calculation for the first 2 seconds', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # 마스크 적용
        imgRegion = cv2.bitwise_and(img, mask_resized)

        # YOLO 모델을 통해 객체 감지
        results = model(imgRegion, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentclass = classNames[cls]

                if currentclass == 'vehicle' and conf > 0.4:
                    currentarray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentarray))

                    # 파란색 상자 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # SORT 추적기 업데이트
        resultracker = tracker.update(detections)
        for results in resultracker:
            x1, y1, x2, y2, id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if id not in id_map:
                id_map[id] = id_counter
                id_counter += 1
                if id_counter > 30:
                    id_counter = 1

            vehicle_id = id_map[id]

            # 차량의 속도가 0으로 갱신될 수 있도록 수정된 속도 계산 부분
            if id in tracked_vehicles:
                tracked_vehicle = tracked_vehicles[id]
                previous_box = tracked_vehicle['box']
                previous_centroid = tracked_vehicle['centroid']
                previous_w, previous_h = previous_box[2] - previous_box[0], previous_box[3] - previous_box[1]
                current_centroid = (cx, cy)

                # 이동 거리 필터링
                distance = euclidean_distance(previous_centroid, current_centroid)
                distance = apply_distance_filter(distance)  # 이동 거리가 너무 작으면 0으로 처리

                # 속도 계산 (픽셀 단위 이동 거리)
                if distance > 0:
                    speed = distance
                else:
                    speed = 0  # 이동이 없다면 속도는 0

                # 객체의 y 좌표를 기반으로 거리 보정
                distance_factor = adjust_speed_based_on_distance(cx, cy, frame_width, frame_height)  # 수정된 부분

                # 속도 보정: 가까운 객체는 빠르게, 먼 객체는 느리게
                adjusted_speed = speed * distance_factor

                # 보정된 속도 값이 음수로 나오지 않도록 처리
                adjusted_speed = max(0, adjusted_speed)

                # 차량이 멈추었을 경우, 즉시 속도를 0으로 갱신
                if distance == 0:  # 차량이 멈추었을 경우
                    adjusted_speed = 0

                # 프레임별 속도 버퍼에 저장
                if id not in frame_buffer:
                    frame_buffer[id] = []
                frame_buffer[id].append(adjusted_speed)

                # 속도 버퍼의 평균값을 속도로 사용
                if len(frame_buffer[id]) > buffer_size:
                    frame_buffer[id].pop(0)

                avg_speed = np.mean(frame_buffer[id])  # 평균 속도

                # 픽셀 단위에서 km/h로 변환
                speed_in_kmh = avg_speed * scale * 3.6  # scale (m/pixel) * 3.6 (초에서 시간으로 변환)

                # 속도가 0인 연속된 프레임 수 추적
                if adjusted_speed == 0:
                    tracked_vehicles[id]['zero_speed_count'] += 1
                else:
                    tracked_vehicles[id]['zero_speed_count'] = 0  # 속도가 다시 0이 아닌 값이면 카운트 리셋

                # 2-3프레임 이상 0km/h가 지속되면 빨간색 상자 표시
                if tracked_vehicles[id]['zero_speed_count'] >= adjust_zero_speed_threshold():
                    tracked_vehicles[id]['red_box'] = True  # 빨간색 상자 표시 상태로 설정
                    speed_in_kmh = 0  # 차량이 멈춘 것으로 간주하여 속도 0으로 설정

                # 차량이 빨간색 상자 상태라면 계속 빨간색으로 표시
                if tracked_vehicles[id].get('red_box', False):
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 상자 표시
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                tracked_vehicles[id]['box'] = [x1, y1, x2, y2]
                tracked_vehicles[id]['centroid'] = current_centroid
                tracked_vehicles[id]['speed'] = speed_in_kmh  # km/h 단위로 속도 업데이트

                # 차량의 이동 방향을 화살표로 표시
                if 'previous_centroid' in tracked_vehicles[id]:
                    prev_centroid = tracked_vehicles[id]['previous_centroid']
                    cv2.arrowedLine(img, prev_centroid, current_centroid, (0, 255, 0), 3)  # 이동 방향을 화살표로 표시

                # 차량의 중심점을 이전 값으로 갱신
                tracked_vehicles[id]['previous_centroid'] = current_centroid

            else:
                tracked_vehicles[id] = {'box': [x1, y1, x2, y2], 'id': vehicle_id, 'centroid': (cx, cy), 'speed': 0, 'zero_speed_count': 0, 'red_box': False}

            # 차량 ID와 보정된 속도 (km/h) 표시
            cvzone.putTextRect(img, f'ID: {vehicle_id} Speed: {tracked_vehicles[id]["speed"]:.2f} km/h', 
                            (x1, y1 - 10), scale=1, thickness=1, offset=1, colorT=(255, 255, 255))

        # 전체 차량 속도의 평균 계산
        all_speeds = [vehicle['speed'] for vehicle in tracked_vehicles.values()]
        if all_speeds:
            avg_vehicle_speed = np.mean(all_speeds)
            last_avg_speed = avg_vehicle_speed  # 마지막 평균 속도 업데이트
        else:
            avg_vehicle_speed = last_avg_speed  # 차량이 없으면 마지막 평균 속도를 사용

        # 평균 속도 표시
        cvzone.putTextRect(img, f'Average Speed: {avg_vehicle_speed:.2f} km/h', 
                           (20, 30), scale=1, thickness=2, offset=1, colorT=(0, 255, 0), colorR=(0, 0, 0))

    # 사라진 차량 추적 관리
    for id in list(tracked_vehicles.keys()):
        if id not in [r[4] for r in resultracker]:
            del tracked_vehicles[id]

    # 출력 이미지를 화면에 표시
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키를 누르면 종료
        break

cv2.destroyAllWindows()
cap.release()
