# main.py

import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import Sort
import threading
from lane_detection2 import lane_detection  # lane_detection 모듈 불러오기
from socket_manager2 import SocketManager  # socket_manager 모듈 불러오기

# 서버 IP 주소와 포트 정의
SERVER_IP = "10.10.14.28"  # 예시: 로컬호스트
SERVER_PORT = 5000  # 예시: 12345번 포트

# 비디오 캡쳐 및 첫 번째 프레임 가져오기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# YOLO 모델 로딩
model = YOLO("./best.pt")

# 클래스 이름 설정 (차량만)
classNames = ["vehicle"]

# 비디오 프레임 크기 가져오기
success, img = cap.read()
if not success:
    print("Error: Couldn't read the first frame of the video")
    exit()
frame_height, frame_width = img.shape[:2]

# SORT 추적기 초기화
tracker = Sort(max_age=100, min_hits=2, iou_threshold=0.3)

# 차량 추적 정보 저장
tracked_vehicles = {}
id_counter = 1  # 차량 ID 생성용
id_map = {}

# 빨간색 상자 표시 여부 저장
red_box_flags = {}

# 픽셀당 실제 거리 (1 픽셀 = 몇 미터인지 설정)
scale = 0.05  # 1 픽셀 = 0.05m (실제 상황에 따라 조정)

# 평균 속도 추적 변수
last_avg_speed = 0
frame_buffer = {}  # 차량별 속도 저장 버퍼
buffer_size = 7  # 속도 계산에 사용할 프레임 수

# 사고 및 정상 상태 메시지 전송 여부
accident_sent = False
okay_sent = False

# 사고 및 차선 변경 감지 변수
lane_accident = {1: False, 2: False}  # 1차선, 2차선 사고 여부 추적

# 사고 상태 추적을 위한 변수 (이전 상태 추적)
last_accident_status = "Accident@0@0\n"

def update_accident_status(socket_manager):
    global accident_sent, lane_accident, okay_sent, last_accident_status
    # 사고 상태에 변화가 있을 때만 메시지 전송
    current_accident_status = f"Accident@{int(lane_accident[1])}@{int(lane_accident[2])}\n"
    
    if current_accident_status != last_accident_status:
        send_thread = threading.Thread(target=socket_manager.send_msg, args=(current_accident_status,))
        send_thread.start()
        last_accident_status = current_accident_status  # 최신 사고 상태 저장
        accident_sent = True  # 사고 메시지 전송 상태 업데이트
        okay_sent = False

    # 두 차선에 빨간색 차량이 모두 없으면 Normal 상태로 전송
    elif not any(lane_accident.values()) and not okay_sent:
        message_to_send = "Normal\n"
        send_thread = threading.Thread(target=socket_manager.send_msg, args=(message_to_send,))
        send_thread.start()
        okay_sent = True
        accident_sent = False


def detect_lane_change(id, prev_lane, current_lane):
    global lane_accident
    # 이전 차선에서 사고 상태를 리셋
    if prev_lane != current_lane:
        if prev_lane == 1 and lane_accident[1]:
            lane_accident[1] = False  # 1차선의 사고 상태를 False로 설정
        elif prev_lane == 2 and lane_accident[2]:
            lane_accident[2] = False  # 2차선의 사고 상태를 False로 설정
        
        # 현재 차선에서 사고 상태를 True로 설정
        if current_lane == 1:
            lane_accident[1] = True
        elif current_lane == 2:
            lane_accident[2] = True

# 두 점 사이의 맨해튼 거리 계산 함수
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 차량 위치에 대한 부드럽게 처리하는 필터 (smooth_position 함수 사용)
def smooth_rectangle_position(prev_pos, current_pos, alpha=0.5):
    smoothed_x1 = alpha * current_pos[0] + (1 - alpha) * prev_pos[0]
    smoothed_y1 = alpha * current_pos[1] + (1 - alpha) * prev_pos[1]
    smoothed_x2 = alpha * current_pos[2] + (1 - alpha) * prev_pos[2]
    smoothed_y2 = alpha * current_pos[3] + (1 - alpha) * prev_pos[3]
    return (int(smoothed_x1), int(smoothed_y1), int(smoothed_x2), int(smoothed_y2))

# 차량 위치에 대한 부드럽게 처리하는 필터
def smooth_position(prev_pos, current_pos, alpha=0.5):
    smoothed_x = alpha * current_pos[0] + (1 - alpha) * prev_pos[0]
    smoothed_y = alpha * current_pos[1] + (1 - alpha) * prev_pos[1]
    return (int(smoothed_x), int(smoothed_y))

# 비디오 처리 루프
def video_thread(socket_manager):
    global last_avg_speed, id_counter, accident_sent, okay_sent, lane_accident, red_box_flags
    while True:
        success, img = cap.read()
        if not success:
            break  # 비디오 끝

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

        # 차선 감지
        lane_image, yellow_points, white_points, green_points = lane_detection(img)
        
        # yellow_points가 비어있는 경우 기본값 설정
        if not yellow_points:
            yellow_points = [[(0, frame_height), (frame_width, 0)]]

        print(green_points)

        if len(yellow_points[0]) == 2:  # yellow_points의 첫 번째 리스트에서 두 점을 확인
            x1, y1 = yellow_points[0][0]
            x2, y2 = yellow_points[0][1]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1  # y = mx + b 형태로 절편 계산
            else:
                slope = float('inf')  # 수직선인 경우 무한 기울기
                intercept = None
        else:
            slope = None
            intercept = None

        cv2.line(lane_image,yellow_points[0][0], yellow_points[0][1], (0,255,0), 2)

        # 추적되지 않는 차량(사라진 차량) 처리
        tracked_ids = set()
        for obj in tracked_objects:
            x1, y1, x2, y2, id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vehicle_center = ((x1 + x2) // 2, y2)
            cv2.circle(lane_image, vehicle_center, 2, (0,255,0), 2)

            # 차선 왼쪽/오른쪽 판단 (기울기 사용)
            if slope is not None and slope != float('inf'):
                expected_y = slope * vehicle_center[0] + intercept
                if vehicle_center[1] < expected_y:
                    lane = 1
                else:
                    lane = 2
            else:
                if vehicle_center[0] < x1:
                    lane = 1
                else:
                    lane = 2

            # 차량 추적 정보 저장 및 갱신
            vehicle_id = id_map.get(id, id_counter)
            if id not in id_map:
                id_map[id] = vehicle_id
                id_counter += 1

            if id in tracked_vehicles:
                prev_centroid = tracked_vehicles[id]['centroid']
                smoothed_centroid = smooth_position(prev_centroid, (cx, cy))
                cx, cy = smoothed_centroid


                # 차량이 속한 차선 계산 (현재 차선)
                current_lane = 1 if lane == 1 else 2  # 1차선과 2차선 구분

                # 차량이 이전에 속한 차선 (현재 추적된 차량 정보에서 가져오기)
                prev_lane = tracked_vehicles[id].get('lane', current_lane)

                # 차선 변경 감지
                detect_lane_change(id, prev_lane, current_lane)

                # 차량 정보 갱신
                tracked_vehicles[id]['lane'] = current_lane  # 현재 차선 정보 갱신

                # 속도 계산
                distance = manhattan_distance(prev_centroid, (cx, cy))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                time_interval = 1 / fps
                speed = (distance / time_interval) * scale * 3.6

                if id not in frame_buffer:
                    frame_buffer[id] = []
                frame_buffer[id].append(speed)
                if len(frame_buffer[id]) > buffer_size:
                    frame_buffer[id].pop(0)

                avg_speed = np.mean(frame_buffer[id])
                tracked_vehicles[id]['speed'] = avg_speed

                # 빨간색 상자 표시 여부 결정
                if avg_speed <= 1.8:
                    red_box_flags[id] = True
                    box_color = (0, 0, 255)
                else:
                    if id not in red_box_flags:
                        red_box_flags[id] = False
                    box_color = (0, 0, 255) if red_box_flags[id] else (255, 0, 0)
                
                 # 여기서 smooth_rectangle_position 사용하여 상자 위치 부드럽게 처리
                x1, y1, x2, y2 = smooth_rectangle_position(
                    tracked_vehicles[id].get('prev_rect', (x1, y1, x2, y2)),
                    (x1, y1, x2, y2)
                )

                cv2.rectangle(lane_image, (x1, y1), (x2, y2), box_color, 2)
                cvzone.putTextRect(lane_image, f'ID: {vehicle_id} Speed: {avg_speed:.2f} km/h Lane : {lane}',
                                   (x1, y1 - 10), scale=1, thickness=2, offset=1, colorT=(255, 255, 255))

                # 차선 정보 추가
                tracked_vehicles[id]['lane'] = current_lane
                tracked_vehicles[id]['centroid'] = (cx, cy)

                # 사고 상태 갱신
                if red_box_flags.get(id, False):
                    lane_accident[current_lane] = True

                tracked_ids.add(id)
            else:
                tracked_vehicles[id] = {'centroid': (cx, cy), 'speed': 0}

        # 사라진 차량 처리: 추적되지 않은 차량은 red_box_flags에서 삭제
        for id in list(red_box_flags.keys()):
            if id not in tracked_ids:
                del red_box_flags[id]

                # 사고 상태 갱신
                for lane in lane_accident.keys():
                    lane_accident[lane] = False

        # 사고 상태 업데이트
        update_accident_status(socket_manager)

        # 평균 속도 표시
        all_speeds = [v['speed'] for v in tracked_vehicles.values() if 'speed' in v]
        avg_speed = np.mean(all_speeds) if all_speeds else last_avg_speed
        last_avg_speed = avg_speed
        cvzone.putTextRect(lane_image, f'Average Speed: {avg_speed:.2f} km/h', 
                           (20, 30), scale=1, thickness=2, offset=1, colorT=(0, 255, 0))

        # 이미지 리사이즈
        img_resized = cv2.resize(lane_image, (640, 480))

        # 화면 출력
        cv2.imshow("Image", lane_image)
        key = cv2.waitKey(1) & 0xFF == ord('q')
        if key:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 소켓과 비디오 처리 각각 다른 스레드에서 실행
    socket_manager = SocketManager(SERVER_IP, SERVER_PORT)
    threading.Thread(target=socket_manager.connect, daemon=True).start()
    video_thread(socket_manager)
