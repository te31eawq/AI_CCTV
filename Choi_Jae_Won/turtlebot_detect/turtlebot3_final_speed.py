import cv2
import numpy as np
from ultralytics import YOLO
import math
from sort import Sort
import threading
import time
from lane_detection2 import lane_detection
from socket_manager import SocketManager

# 서버 IP 주소와 포트 정의
SERVER_IP = "10.10.14.28"
SERVER_PORT = 5000

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

# 각 초록색 선분의 실제 거리 (50cm = 0.5m)
line_distance = 0.5  # 미터 단위

# 각 차량이 마지막으로 지난 초록색 선분과 시간 기록
last_cross_info = {}  # {id: (last_line_index, last_cross_time)}

# 사고 여부를 판단하는 딕셔너리
accident_info = {}  # {id: {'last_time': <time>, 'last_line': <line_index>, 'status': 'accident' or 'safe'}}

def get_line_equation(p1, p2):
    """두 점 p1, p2를 통해 직선 방정식 y = mx + b 구하기"""
    x1, y1 = p1
    x2, y2 = p2
    # 기울기 m 계산
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # x1 == x2이면 수직선
    # y절편 b 계산
    b = y1 - m * x1 if m != float('inf') else None
    return m, b

def is_crossing_line(vehicle_center, m, b):
    """차량의 밑면 중심이 직선 y = mx + b를 지나는지 여부 확인"""
    x, y = vehicle_center
    if m == float('inf'):  # 수직선
        return x == b  # 직선의 x좌표가 b일 경우에만 교차
    else:
        return y >= m * x + b - 10 and y <= m * x + b + 10  # 직선의 범위 내에서만 교차로 간주

def video_thread(socket_manager):
    global tracked_vehicles, accident_info
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
        
        # 초록색 선분 정의 (각 선분은 두 점으로 구성)
        if not green_points:
            green_points = [
                [[0, 311], [203, 477]],  # 선분 1
                [[0, 162], [493, 477]],  # 선분 2
                [[445, 0], [636, 119]],  # 선분 3
                [[232, 0], [636, 253]],  # 선분 4
                [[6, 0], [636, 405]]     # 선분 5
            ]
        
        green_points = sorted(green_points,key=lambda points: (points[0][1] + points[1][1]) // 2)

        # 초록색 선분을 그리기
        for i, points in enumerate(green_points):
            cv2.line(lane_image, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)

            # 선분 번호를 표시 (중간 지점에 번호를 표시)
            mid_point = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
            cv2.putText(lane_image, f"Line {i + 1}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # 각 선분에 대한 직선 방정식 구하기
        line_equations = [get_line_equation(points[0], points[1]) for points in green_points]

        # 추적된 차량 처리
        for obj in tracked_objects:
            x1, y1, x2, y2, id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vehicle_center = (cx, y2)  # 차량의 밑면 중심

            # id를 정수로 변환하여 처리
            id = int(id)

            # 초록색 선분을 통과하는지 확인
            for i, (m, b) in enumerate(line_equations):
                if is_crossing_line(vehicle_center, m, b):
                    current_time = time.time()  # 현실 시간 (초)

                    # 차량이 이전에 지난 선분 정보 확인
                    if id in last_cross_info:
                        last_line_index, last_time = last_cross_info[id]
                        if last_line_index != i:  # 다른 선분을 지날 때만 속도 계산
                            time_diff = current_time - last_time
                            if time_diff > 0.1:  # 너무 짧은 시간 차이는 무시
                                # 거리 계산 및 속도 계산 (실제 거리 기준)
                                distance = line_distance * abs(i - last_line_index)  # 선분 간 거리 계산
                                speed = distance / time_diff  # 초속 (m/s) 계산
                                tracked_vehicles[id] = {'speed': speed}  # 초속으로 저장
                            # 현재 선분 정보로 갱신
                            last_cross_info[id] = (i, current_time)
                    else:
                        # 처음 선분을 지날 때는 정보만 기록
                        last_cross_info[id] = (i, current_time)

            # 사고 여부 판단
            if id in last_cross_info:
                last_line_index, last_time = last_cross_info[id]
                current_time = time.time()
                if current_time - last_time > 5:  # 5초 이상 지나면 사고 발생으로 판단
                    if id not in accident_info:
                        accident_info[id] = {'last_time': current_time, 'last_line': last_line_index, 'status': 'accident'}
                        # 사고 메시지를 전송 (Accident@Line N)
                        accident_message = f"Accident@Line{last_line_index + 1}\n"
                        socket_manager.send_msg(accident_message)  # 각 차량마다 사고 메시지 전송


            # 차량 정보 갱신
            # 파란색 박스 그리기
            cv2.rectangle(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 차량 ID와 속도 표시 (속도가 없는 경우 "Speed: N/A"로 표시)
            speed_text = f"ID: {id} "
            if id in tracked_vehicles:
                speed = tracked_vehicles[id]['speed']
                speed_text += f"Speed: {speed:.2f} m/s"  # 초속으로 표시
            else:
                speed_text += "Speed: N/A"

            cv2.putText(lane_image, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 이미지 출력
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
