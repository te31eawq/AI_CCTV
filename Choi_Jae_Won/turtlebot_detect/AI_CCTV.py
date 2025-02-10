# -*- coding: utf-8 -*-
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
SERVER_IP = "10.10.14.21"
SERVER_PORT = 5000

# 비디오 캡쳐 및 첫 번째 프레임 가져오기
cap = cv2.VideoCapture('http://10.10.14.88:8080/?action=stream')
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# YOLO 모델 로딩
model = YOLO('./best.pt')

# 클래스 이름 설정 (차량만)
classNames = ["vehicle"]

# 비디오 프레임 크기 가져오기
success, img = cap.read()
if not success:
    print("Error: Couldn't read the first frame of the video")
    exit()

white_points = []
green_points = []

frame_height, frame_width = img.shape[:2]

# SORT 추적기 초기화
tracker = Sort(max_age=100, min_hits=2, iou_threshold=0.3)

# 차량 추적 정보 저장
tracked_vehicles = {}
old_tracked_vehicle = {}

# 저속 차량 감지용
caution_time = {}

# 차량 정보를 저장할 리스트
vehicles_passed = []

# 각 초록색 선분의 실제 거리 (50cm = 0.5m)
line_distance = 0.5  # 미터 단위

# 각 차량이 마지막으로 지난 초록색 선분과 시간 기록
last_cross_info = {}  # {id: (last_line_index, last_cross_time)}
last_cross_info2 = {}  # {id: (last_line_index, last_cross_time)}

# 사고 여부를 판단하는 딕셔너리
accident_info = {}  # {id: {'last_time': <time>, 'last_line': <line_index>, 'status': 'accident' or 'safe'}}

start_time = time.time()
lane_lines = None  # 초기에는 lane_lines를 None으로 설정하여 초기화

# 차선
lane_counts = {}
old_lane_counts = {}
# 각 차량이 한 번만 카운트되도록 하기 위한 집합
counted_vehicles = set()

def calculate_iou(box1, box2):
    # Unpack the coordinates of both boxes
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    
    # Calculate the area of intersection
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    
    # Check if there is an intersection
    if xi2 > xi1 and yi2 > yi1:
        intersection_area = (xi2 - xi1) * (yi2 - yi1)
    else:
        intersection_area = 0

    # Calculate the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def filter_detections_by_iou(detections, iou_thresh=0.5):
    if len(detections) == 0:
        return detections

    filtered = []
    for i, det in enumerate(detections):
        duplicate = False
        for j, fdet in enumerate(filtered):
            iou = calculate_iou(det[:4], fdet[:4])
            if iou > iou_thresh:
                # 상황에 따라 confidence가 높은 것을 남기거나, 둘 다 남길 수 있음
                duplicate = True
                # 예시: confidence가 더 높은 것을 남김 (즉, 중복 제거)
                if det[4] > fdet[4]:
                    filtered[j] = det  # 더 높은 detection으로 교체
                break
        if not duplicate:
            filtered.append(det)
    return np.array(filtered)

def draw_bounding_box_smoothly(frame, x1, y1, x2, y2, color, alpha=0.6):
    # float 형식으로 좌표 변경
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    
    overlay = frame.copy()  # 원본 이미지를 복사해서 겹칠 예정
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 사각형을 복사본에 그린다.
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 원본 이미지와 복사본을 합성하여 부드러운 효과

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

def filter_similar_lines(green_points, threshold=60):
    """
    y값이 비슷한 선분들을 필터링하여 하나만 남기기
    threshold는 y값의 차이가 이 값 이하인 선분을 비슷하다고 판단하여 제외한다.
    """
    filtered_lines = []
    
    for i, points1 in enumerate(green_points):
        # 현재 선분의 y값
        _, y1 = points1[0]
        _, y2 = points1[1]
        avg_y1 = (y1 + y2) // 2  # 선분의 평균 y값
        
        # 다른 선분들과 비교
        is_similar = False
        for j, points2 in enumerate(filtered_lines):
            _, y3 = points2[0]
            _, y4 = points2[1]
            avg_y2 = (y3 + y4) // 2  # 선분의 평균 y값
            
            # 평균 y값 차이가 threshold 이내면 비슷한 선분으로 판단
            if abs(avg_y1 - avg_y2) < threshold:
                is_similar = True
                break
        
        # 비슷한 선분이 없으면 추가
        if not is_similar:
            filtered_lines.append(points1)
    
    return filtered_lines

## 속도 추가용

def add_middle_lines(green_points):
    middle_lines = []
    for i in range(len(green_points) - 1):
        # 첫 번째 선분의 시작점과 두 번째 선분의 시작점의 중간 점
        x1, y1 = green_points[i][0]  # 첫 번째 선분의 시작 점
        x2, y2 = green_points[i + 1][0]  # 두 번째 선분의 시작 점
        start_mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 첫 번째 선분의 끝점과 두 번째 선분의 끝점의 중간 점
        x3, y3 = green_points[i][1]  # 첫 번째 선분의 끝 점
        x4, y4 = green_points[i + 1][1]  # 두 번째 선분의 끝 점
        end_mid_point = ((x3 + x4) // 2, (y3 + y4) // 2)

        # 중간 선분을 추가
        middle_lines.append([start_mid_point, end_mid_point])

    return middle_lines

def update_lane_lines_and_middle_lines(green_points):
    """기존 선분과 추가 선분을 결합하여 속도 측정"""
    middle_lines = add_middle_lines(green_points)
    all_lines = green_points + middle_lines  # 기존 선분들 + 추가 선분들
    return all_lines

def calculate_speed_for_added_lines(vehicle_center, last_cross_info, line_equations, current_time, id, green_points):

    cx, cy = vehicle_center
    """추가된 선분을 통과할 때 속도 계산"""
    for i, (m, b) in enumerate(line_equations):
        # 기존 초록색 선분만 사용하도록 수정
        if i < len(green_points):  # 기존 초록색 선분만 처리
            if is_crossing_line(vehicle_center, m, b):
                if id in last_cross_info:
                    last_line_index, last_time = last_cross_info[id]
                    if last_line_index != i:
                        time_diff = current_time - last_time
                        if time_diff > 0.1:
                            # 기존 선분만 반영
                            distance = (line_distance / 2)
                            speed = (distance / time_diff) * 100 # * distance_factor
                            speed = int(speed)
                            tracked_vehicles.setdefault(id, {}).update({'speed': speed})
                        last_cross_info[id] = (i, current_time)
                        last_cross_info2[id] = (i, current_time)
                else:
                    last_cross_info[id] = (i, current_time)
                    last_cross_info2[id] = (i, current_time)

    # 중간 선분들을 통과할 때도 속도 계산
    for i, (m, b) in enumerate(line_equations[len(green_points):]):  # 중간 선분들만 처리
        idx = i + len(green_points)  # 중간 선분 인덱스
        if is_crossing_line(vehicle_center, m, b):
            if id in last_cross_info:
                last_line_index, last_time = last_cross_info[id]
                if last_line_index != idx:
                    time_diff = current_time - last_time
                    if time_diff > 0.1:
                        # 중간 선분만 반영
                        distance = (line_distance / 2)
                        speed = (distance / time_diff) * 100 # * distance_factor
                        speed = int(speed)
                        tracked_vehicles.setdefault(id, {}).update({'speed': speed})
                    last_cross_info[id] = (idx, current_time)
            else:
                last_cross_info[id] = (idx, current_time)

## 차선 변경용

def assign_lane(vehicle_center, white_points):
    cx, cy = vehicle_center

    for i in range(len(white_points) - 1):
        # 현재 선분의 두 점
        p1 = white_points[i][0]
        p2 = white_points[i][1]

        # 다음 선분의 두 점
        p3 = white_points[i + 1][0]
        p4 = white_points[i + 1][1]

        # 첫 번째 직선 방정식 (p1, p2)
        m1, b1 = get_line_equation(p1, p2)

        # 두 번째 직선 방정식 (p3, p4)
        m2, b2 = get_line_equation(p3, p4)

        # 첫 번째 직선의 y값 구하기
        y1_at_cx = m1 * cx + b1 if m1 != float('inf') else None  # 수직선일 경우 None
        y2_at_cx = m2 * cx + b2 if m2 != float('inf') else None  # 수직선일 경우 None

        # 두 직선 사이에 차량의 y값이 위치하는지 확인
        if m1 != float('inf') and m2 != float('inf'):
            if min(y1_at_cx, y2_at_cx) <= cy <= max(y1_at_cx, y2_at_cx):
                lane_number = i + 1  # 차선 번호는 1부터 시작
                return lane_number if lane_number in [1, 2] else -1
        elif m1 == float('inf'):
            if min(p1[1], p2[1]) <= cy <= max(p1[1], p2[1]):
                lane_number = i + 1
                return lane_number if lane_number in [1, 2] else -1
        elif m2 == float('inf'):
            if min(p3[1], p4[1]) <= cy <= max(p3[1], p4[1]):
                lane_number = i + 1
                return lane_number if lane_number in [1, 2] else -1

    return -1  # 차선이 1, 2가 아닌 경우

# 사고 발생 여부 추적을 위한 변수
accident_sent = {}
accident_vehicle_ids = []

# 저속 발생 용
accident_low_sent = {}

# 과속 발생 용
accident_fast_sent = {}

# 차량 이름을 부여할 딕셔너리 (ID별로 차량 이름을 저장)
vehicle_names = {}

# 차량 이름을 부여할 배열 (ID별로 차량 이름을 저장)
vehicles_passed = []

def video_thread(socket_manager):
    global tracked_vehicles, accident_info, lane_lines, accident_sent, accident_vehicle_ids,white_points,old_lane_counts,old_tracked_vehicle, caution_time, accident_low_sent

    while True:
        success, img = cap.read()
        if not success:
            break  # 비디오 끝
        
        # YOLO 모델을 통해 객체 감지
        results = model(img, stream=True,verbose=False)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # 차량만 탐지
                if cls < len(classNames) and classNames[cls] == 'vehicle' and conf > 0.6:
                    currentarray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentarray))

        # IoU 기준 후처리 적용 (필요에 따라 활성화)
        detections = filter_detections_by_iou(detections, iou_thresh=0.5)

        # SORT 추적기 업데이트
        tracked_objects = tracker.update(detections)

        # 차선 감지 및 차선 업데이트
        if time.time() - start_time <= 2:
            lane_image, yellow_points, white_points, green_points = lane_detection(img)
            if green_points is None:
                print("Error: No green points found")
            else:
                green_points = filter_similar_lines(green_points)
                green_points = sorted(green_points, key=lambda points: (points[0][1] + points[1][1]) // 2)
                lane_lines = green_points
        else:
            lane_image = img.copy()
            green_points = lane_lines
        if white_points is None:
            white_points = []
        if green_points is None:
            green_points = [[[0, 311], [203, 477]]]
        # 선분 그리기
        if green_points is not None:
            for i, points in enumerate(green_points):
                cv2.line(lane_image, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
                mid_point = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
                cv2.putText(lane_image, f"Line {i + 1}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if white_points is not None:
            for i, points in enumerate(white_points):
                cv2.line(lane_image, tuple(points[0]), tuple(points[1]), (0, 0, 0), 2)

        # 추가 선분들로 구성된 모든 선분 (속도 계산용)
        all_lines = update_lane_lines_and_middle_lines(green_points)
        line_equations = [get_line_equation(points[0], points[1]) for points in all_lines]

        # # 선분 그리기 (기존 + 추가 선분)
        # for i, points in enumerate(all_lines):
        #         cv2.line(lane_image, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)

        # 사고 발생 여부 추적 (기존 선분만 사용)
        for obj in tracked_objects:
            x1, y1, x2, y2, id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vehicle_center = (cx, y2)

            id = int(id)
            current_time = time.time()

            tracked_vehicles.setdefault(id, {}).update({'lane': assign_lane(vehicle_center, white_points)})

            if tracked_vehicles[id]['lane']:
                lane = tracked_vehicles[id]['lane']
                if lane not in lane_counts:
                    lane_counts[lane] = 0
                lane_counts[lane] += 1
            
            calculate_speed_for_added_lines(vehicle_center, last_cross_info, line_equations, current_time, id, green_points)

            if id not in last_cross_info2:
                last_cross_info2[id] = (0, current_time)  # 초기값 설정
            last_line_index2, last_time2 = last_cross_info2[id]
            if last_line_index2+2 >= 5:
                last_line_index2 = 2
            
            tracked_vehicles.setdefault(id, {}).update({'last_line': last_line_index2+2})

            # 사고 발생 여부 추적
            if id in last_cross_info:
                last_line_index, last_time = last_cross_info[id]
            
                if current_time - last_time > 4:
                    if id not in accident_info:
                        accident_info[id] = {'last_time': current_time, 'last_line': last_line_index2, 'status': 'accident'}
                        tracked_vehicles.setdefault(id, {}).update({'speed': 0})

            if id in accident_info and accident_info[id]['status'] == 'accident':
                draw_bounding_box_smoothly(lane_image, x1, y1, x2, y2, (0, 0, 255), alpha=0.6)  # 빨간색으로 사고 차량
            else:
                draw_bounding_box_smoothly(lane_image, x1, y1, x2, y2, (255, 0, 0), alpha=0.6)  # 파란색으로 정상 차량

            # 차량 이름을 부여하거나 변경하는 부분
            vehicle_name_find = None
            for vehicle in vehicles_passed:
                if vehicle['line_number'] == last_line_index2 + 1:  # 선 번호는 1부터 시작하니 +1 해줌
                    vehicle_name_find = vehicle['vehicle_name']
                    break

            # 차량 이름을 업데이트하는 부분
            if vehicle_name_find:
                # 기존에 부여된 이름을 가진 차량이 있으면, 그 이름을 변경
                for vehicle in vehicles_passed:
                    if vehicle['vehicle_name'] == vehicle_name_find:
                        # tracked_vehicles[id] = {'vehicle_name': vehicle_name_find}
                        tracked_vehicles.setdefault(id, {}).update({'vehicle_name': vehicle_name_find})
                        # vehicle_names[id] = {vehicle_name_find}
                        vehicles_passed.remove(vehicle)

                tracked_vehicles.setdefault(id, {}).update({'vehicle_name': vehicle_name_find})
            else:
                if id not in tracked_vehicles:
                    tracked_vehicles.setdefault(id, {}).update({'vehicle_name': 'Unknown'})

            speed_text = f"ID: {id} "

            if 'speed' in tracked_vehicles[id]:
                speed = tracked_vehicles[id]['speed']
                speed_text += f"Speed: {speed:.2f} cm/s"
            else:
                speed_text += "Speed: N/A"
                tracked_vehicles.setdefault(id, {}).update({'last_line': 1})

            if 'lane' in tracked_vehicles[id]:
                lane = tracked_vehicles[id]['lane']
                speed_text += f" Lane: {lane}"
            else:
                speed_text += f" Lane: ??"

            
            # 차량 이름을 안전하게 가져오도록 수정
            vehicle_name_text = tracked_vehicles.get(id, {}).get('vehicle_name', 'Unknown')
            tracked_vehicle_name = tracked_vehicles.get(id, {}).get('vehicle_name', 'Unknown')
            tracked_vehicle_speed = tracked_vehicles.get(id, {}).get('speed', 0)
            tracked_vehicle_lane = tracked_vehicles.get(id, {}).get('lane', 0)
            tracked_vehicle_lastline = tracked_vehicles.get(id, {}).get('last_line', 0)

            total_count = sum(lane_counts.values())
            if total_count < 8:
            # 'Unknown' 또는 None인 차량 이름이 있을 경우 메시지를 보내지 않도록 처리
                if tracked_vehicles[id].get('speed') == 0:
                    if id not in accident_sent:
                        accident_sent[id] = False
                    if not accident_sent[id]:
                        socket_manager.send_msg(f'ACCIDENT@{tracked_vehicle_lane}@{tracked_vehicle_lastline}\n')
                        accident_sent[id] = True
                        accident_low_sent[id] = True
                elif tracked_vehicles[id].get('speed') and tracked_vehicles[id].get('speed') <= 10:
                    if not caution_time.get(id):  # id별로 caution_time이 없으면 설정
                        caution_time[id] = current_time
                    if current_time - caution_time[id] > 4:  # 4초가 지났을 때
                        if id not in accident_low_sent:
                            accident_low_sent[id] = False
                        if not accident_low_sent[id]:
                            socket_manager.send_msg(f'CAUTION@{tracked_vehicle_lane}@{tracked_vehicle_lastline}\n')
                            accident_low_sent[id] = True
                elif tracked_vehicles[id].get('speed') and tracked_vehicles[id].get('speed') >= 20:
                    if id not in accident_fast_sent:
                        accident_fast_sent[id] = False
                    if not accident_fast_sent[id]:
                        socket_manager.send_msg(f'DANGER@{tracked_vehicle_lane}@{tracked_vehicle_lastline}\n')
                        accident_fast_sent[id] = True 
                elif tracked_vehicles[id].get('speed') and tracked_vehicles[id].get('speed') > 10:
                    # 속도가 10 이상이면 알림을 초기화
                    if id in caution_time:
                        del caution_time[id]
                    if id in accident_low_sent:
                        accident_low_sent[id] = False


            if tracked_vehicles[id].get('vehicle_name') and tracked_vehicles[id].get('speed') and \
            tracked_vehicles[id].get('lane') and tracked_vehicles[id].get('last_line'):
                # 차량 정보가 달라졌을 경우에만 메시지 보내기
                if id not in old_tracked_vehicle:
                    old_tracked_vehicle[id] = {}

                if any(old_tracked_vehicle[id].get(key) != tracked_vehicles[id].get(key) for key in ['vehicle_name', 'speed', 'lane', 'last_line']):
                    for key in ['vehicle_name', 'speed', 'lane', 'last_line']:
                        old_tracked_vehicle[id][key] = tracked_vehicles[id][key]
                    socket_manager.send_msg(f'[JSH_QT]INFO@{tracked_vehicle_name}@{tracked_vehicle_speed}@{tracked_vehicle_lane}@{tracked_vehicle_lastline}\n')

                
            cv2.putText(lane_image, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(lane_image, vehicle_name_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ## 배열 지우기

        for id in list(tracked_vehicles.keys()):
            if id not in [int(obj[4]) for obj in tracked_objects]:
                del tracked_vehicles[id]
        
        # 사고 차량이 사라졌을 때 배열에서 지우는 부분
        for id in list(accident_info.keys()):
            # 화면에서 차량이 사라졌고, 사고 메시지를 아직 보내지 않았다면
            if id not in [int(obj[4]) for obj in tracked_objects]:
                # 사고 차량의 마지막 구간에 대한 메시지를 전송
                last_line_index = accident_info[id]['last_line']
                if id not in accident_sent:  # 사고 메시지가 이미 전송되지 않았다면
                    accident_sent[id] = True  # 사고 메시지를 전송했다고 기록
                    accident_low_sent[id] = False
                del accident_info[id]
        
        # 사고 차량이 모두 사라졌을 때 "OK" 보내기
        if not accident_info:  # 사고 차량이 모두 사라졌다면
            if not accident_sent.get('all_cleared', False):  # 이미 "OK"를 보냈으면 다시 보내지 않도록
                # socket_manager.send_msg("OK\n")
                accident_sent['all_cleared'] = True  # 한 번만 OK 메시지를 보냈다고 기록
        else:
            accident_sent['all_cleared'] = False  # 사고 차량이 남아있으면 다시 "OK"를 보낼 수 있도록

        cv2.imshow("Image", lane_image)
        
        # lane_counts 변경 시 출력 (이전 값과 달라지고 lane_counts가 비어 있지 않을 때)
        if old_lane_counts != lane_counts and lane_counts:
            old_lane_counts = lane_counts.copy()  # 딕셔너리 복사하여 값 변경 추적
            socket_manager.send_msg(f'[JSH_QT]COUNT@{lane_counts.get(1,0)}@{lane_counts.get(2,0)}@{lane_counts.get(-1,0)}\n')
        for i in lane_counts:
            lane_counts[i] = 0
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            lane_image, yellow_points, white_points, green_points = lane_detection(img)

            for i in range(len(white_points)):
                white_points[i] = sorted(white_points[i], key=lambda point: point[0])
            white_points = sorted(white_points, key=lambda points: (points[0][0], points[1][0]))

            green_points = filter_similar_lines(green_points)
            green_points = sorted(green_points, key=lambda points: (points[0][1] + points[1][1]) // 2)
            lane_lines = green_points
        if cv2.waitKey(1) & 0xFF == ord('f'):
            cv2.imwrite('image.jpg', lane_image)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def handle_message(message):
    global white_points
    """
    서버에서 차량 메시지를 받아 처리하는 함수.
    메시지 예시: [차량이름]PASS@2 -> 차량이 '차량이름'이고 Line 2를 지난 경우
    """
    print(message)
    # 메시지 예시: [차량이름]PASS@2
    if 'PASS@' in message:
        vehicle_name = message.split(']')[0][1:]  # 차량 이름 추출 (대괄호 안의 내용)
        line_number = int(message.split('@')[1])  # 라인 번호 추출
        
        # 차량 이름과 차선 번호를 배열에 저장
        vehicle_info = {'vehicle_name': vehicle_name, 'line_number': line_number}
        vehicles_passed.append(vehicle_info)  # vehicles_passed 배열에 저장

        # 예시로 출력
        print(f"차량 이름: {vehicle_name}, 차선 번호: {line_number}")

    elif 'GETLANE' in message:
        if white_points is not None:
            for i, points in enumerate(white_points):
                time.sleep(0.1)
                socket_manager.send_msg(f'[JSH_QT]LANE{i}@{points[0][0]}@{points[0][1]}@{points[1][0]}@{points[1][1]}\n')
    
    elif 'LANE' in message:
        # Extract the lane number (which should be between 'LANE' and '@')
        lane_index = int(message.split('LANE')[1].split('@')[0])  # Extract only the lane number
        
        # Extract the coordinates after the '@' symbol and convert to integers
        lane_points = list(map(int, message.split('@')[1:]))  # Convert all lane points to integers

        # 하나라도 -1이 포함되면 해당 lane 삭제
        if -1 in lane_points:
            if 0 <= lane_index < len(white_points):
                del white_points[lane_index]  # 리스트에서 해당 차선 삭제
            for i in range(len(white_points)):
                white_points[i] = sorted(white_points[i], key=lambda point: point[0])
            socket_manager.send_msg(f'[JSH_QT]LANE{lane_index}@-1@-1@-1@-1\n')
        else:
            socket_manager.send_msg(f'[JSH_QT]SORTED\n')
            # Check if the lane_index exists in the white_points list and update accordingly
            if len(white_points) <= lane_index:
                for i in range(len(white_points)):
                    white_points[i] = sorted(white_points[i], key=lambda point: point[0])
                white_points.append([(lane_points[0], lane_points[1]), (lane_points[2], lane_points[3])])
                white_points = sorted(white_points, key=lambda points: (int(points[0][0]), int(points[1][0])))
            else:
                for i in range(len(white_points)):
                    white_points[i] = sorted(white_points[i], key=lambda point: point[0])
                white_points[lane_index] = [(lane_points[0], lane_points[1]), (lane_points[2], lane_points[3])]

            white_points = sorted(white_points, key=lambda points: (int(points[0][0]), int(points[1][0])))

        for i, points in enumerate(white_points):
            time.sleep(0.1)
            socket_manager.send_msg(f'[JSH_QT]LANE{i}@{points[0][0]}@{points[0][1]}@{points[1][0]}@{points[1][1]}\n')

def connect_thread(socket_manager):
    """서버에 연결하는 스레드"""
    socket_manager.connect()

if __name__ == "__main__":
    # 소켓과 비디오 처리 각각 다른 스레드에서 실행
    socket_manager = SocketManager(SERVER_IP, SERVER_PORT, callback=handle_message)
    threading.Thread(target=connect_thread, args=(socket_manager,), daemon=True).start()
    video_thread(socket_manager)
