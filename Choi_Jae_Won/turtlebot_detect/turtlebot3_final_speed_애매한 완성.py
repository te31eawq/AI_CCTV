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

start_time = time.time()
lane_lines = None  # 초기에는 lane_lines를 None으로 설정하여 초기화

# 선분을 연장하여 화면 끝까지 그리기
def extend_line_to_screen(line, frame_width, frame_height):
    (x1, y1), (x2, y2) = line
    # 기울기와 y절편 구하기
    if x2 != x1:
        m, b = get_line_equation((x1, y1), (x2, y2))
        
        # y = mx + b에서 x에 대해 해를 구해 화면 끝까지 연장
        x_start = 0  # x1
        y_start = m * x_start + b if m != float('inf') else y1
        
        x_end = frame_width  # x2
        y_end = m * x_end + b if m != float('inf') else y2
        
        # 화면 크기에 맞게 범위 체크 (y가 화면 범위를 벗어나면 y값을 조정)
        if y_start < 0:
            y_start = 0
        elif y_start > frame_height:
            y_start = frame_height
            
        if y_end < 0:
            y_end = 0
        elif y_end > frame_height:
            y_end = frame_height
        
        return ((x_start, y_start), (x_end, y_end))
    else:
        # 수직선일 경우
        return ((x1, 0), (x1, frame_height))

def extend_line_to_intersection(line, white_points, frame_width, frame_height):
    """
    초록색 선분(line)이 white_points 배열의 첫 번째와 마지막 선분과 교차하는 지점까지만 그리도록 수정된 함수
    """
    if len(white_points) < 2:  # white_points가 2개 미만일 경우 교차점 계산 불가
        return extend_line_to_screen(line, frame_width, frame_height)  # 그냥 화면 끝까지 연장
    
    (x1, y1), (x2, y2) = line
    
    # 첫 번째와 마지막 선분만 사용
    first_white_line = white_points[0]
    last_white_line = white_points[-1]

    # 첫 번째와 마지막 선분과 교차점 계산
    intersection_first = get_intersection(line, first_white_line)
    intersection_last = get_intersection(line, last_white_line)
    
    if intersection_first and intersection_last:
        # 두 교차점을 계산한 후 선분을 그립니다
        x_int_first, y_int_first = intersection_first
        x_int_last, y_int_last = intersection_last
        
        # 교차점이 두 개 존재하면, 그 사이를 그리도록
        return ((x_int_first, y_int_first), (x_int_last, y_int_last))
    
    # 교차점이 없으면, 원래대로 화면 끝까지 연장
    return extend_line_to_screen(line, frame_width, frame_height)

def get_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # 기울기와 y절편 계산
    m1, b1 = get_line_equation((x1, y1), (x2, y2))
    m2, b2 = get_line_equation((x3, y3), (x4, y4))

    # 교차점 계산 (두 직선의 기울기가 같으면 교차점이 없으므로 무시)
    if m1 != m2:
        # 기울기 m1, m2가 다르면 교차점 계산
        if m1 != float('inf') and m2 != float('inf'):
            x_int = (b2 - b1) / (m1 - m2)
            y_int = m1 * x_int + b1
        elif m1 == float('inf'):  # 첫 번째 직선이 수직선
            x_int = x1
            y_int = m2 * x_int + b2
        elif m2 == float('inf'):  # 두 번째 직선이 수직선
            x_int = x3
            y_int = m1 * x_int + b1

        return (int(x_int), int(y_int))
    else:
        return None  # 교차점이 없으면 None

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

def filter_similar_lines(green_points, threshold=100):
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

# 사고 발생 여부 추적을 위한 변수
accident_sent = {}
accident_vehicle_ids = []

def video_thread(socket_manager):
    global tracked_vehicles, accident_info, lane_lines, accident_sent, accident_vehicle_ids
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

        # 차선 감지 및 차선 업데이트
        if time.time() - start_time <= 2:
            lane_image, yellow_points, white_points, green_points = lane_detection(img)
            green_points = filter_similar_lines(green_points)
            white_points = filter_similar_lines(white_points)
            green_points = sorted(green_points, key=lambda points: (points[0][1] + points[1][1]) // 2)
            white_points = sorted(white_points, key=lambda points: (points[0][1] + points[1][1]))
            lane_lines = green_points
        else:
            lane_image = img.copy()
            green_points = lane_lines

        # 교차점 찾기
        if len(white_points) > 1:
            first_white_line = white_points[0]
            last_white_line = white_points[-1]
            intersection = get_intersection(first_white_line, last_white_line)

            if intersection:
                x_int, y_int = intersection
                # 초록색 선분을 첫 번째와 마지막 선분 사이에 그리기
                green_points = [extend_line_to_intersection(line, white_points, frame_width, frame_height) for line in green_points]


        # 초록색 선분 그리기 수정
        for i, points in enumerate(green_points):
            # 교차점을 찾기 위한 선분 연장
            extended_line = extend_line_to_intersection(points, white_points, frame_width, frame_height)
            cv2.line(lane_image, (int(extended_line[0][0]), int(extended_line[0][1])), (int(extended_line[1][0]), int(extended_line[1][1])), (0, 255, 0), 2)
            mid_point = ((extended_line[0][0] + extended_line[1][0]) // 2, (extended_line[0][1] + extended_line[1][1]) // 2)
            cv2.putText(lane_image, f"Line {i + 1}", (int(mid_point[0]), int(mid_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for i, points in enumerate(white_points):
            cv2.line(lane_image, tuple(points[0]), tuple(points[1]), (0, 0, 0), 2)
            cv2.putText(lane_image, f"Line {i + 1}", (int(points[0][0]), int(points[0][1])+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        line_equations = [get_line_equation(points[0], points[1]) for points in green_points]

        # 사고 발생 여부 추적
        for obj in tracked_objects:
            x1, y1, x2, y2, id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vehicle_center = (cx, y2)

            id = int(id)

            accident_detected = False

            for i, (m, b) in enumerate(line_equations):
                if is_crossing_line(vehicle_center, m, b):
                    current_time = time.time()

                    if id in last_cross_info:
                        last_line_index , last_time = last_cross_info[id]
                        if last_line_index != i:
                            time_diff = current_time - last_time
                            if time_diff > 0.1:
                                distance = line_distance * abs(i - last_line_index)
                                speed = distance / time_diff
                                tracked_vehicles[id] = {'speed': speed}
                            last_cross_info[id] = (i, current_time)
                    else:
                        last_cross_info[id] = (i, current_time)

                # 사고 발생 여부 추적
                if id in last_cross_info:
                    last_line_index, last_time = last_cross_info[id]
                    current_time = time.time()

                    if current_time - last_time > 3:
                        if id not in accident_info:
                            accident_info[id] = {'last_time': current_time, 'last_line': last_line_index, 'status': 'accident'}
                            socket_manager.send_msg(f'Accident@{last_line_index + 1}\n')  # 사고 발생 지점 전송

            # 사고 차량 상태 표시
            if id in accident_info and accident_info[id]['status'] == 'accident':
                cv2.rectangle(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색으로 사고 차량 표시
            else:
                cv2.rectangle(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 정상 차량은 파란색

            speed_text = f"ID: {id} "
            if id in tracked_vehicles:
                speed = tracked_vehicles[id]['speed']
                speed_text += f"Speed: {speed:.2f} m/s"
            else:
                speed_text += "Speed: N/A"

            cv2.putText(lane_image, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 사고 차량이 사라졌을 때 배열에서 지우는 부분
        for id in list(accident_info.keys()):
            # 화면에서 차량이 사라졌고, 사고 메시지를 아직 보내지 않았다면
            if id not in [int(obj[4]) for obj in tracked_objects]:
                # 사고 차량의 마지막 구간에 대한 메시지를 전송
                last_line_index = accident_info[id]['last_line']
                if id not in accident_sent:  # 사고 메시지가 이미 전송되지 않았다면
                    socket_manager.send_msg(f'Accident@{last_line_index+1}\n')
                    accident_sent[id] = True  # 사고 메시지를 전송했다고 기록
                del accident_info[id]

        # 사고 차량이 모두 사라졌을 때 "OK" 보내기
        if not accident_info:  # 사고 차량이 모두 사라졌다면
            if not accident_sent.get('all_cleared', False):  # 이미 "OK"를 보냈으면 다시 보내지 않도록
                socket_manager.send_msg("OK\n")
                accident_sent['all_cleared'] = True  # 한 번만 OK 메시지를 보냈다고 기록
        else:
            accident_sent['all_cleared'] = False  # 사고 차량이 남아있으면 다시 "OK"를 보낼 수 있도록

        cv2.imshow("Image", lane_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            lane_image, yellow_points, white_points, green_points = lane_detection(img)
            green_points = filter_similar_lines(green_points)
            green_points = sorted(green_points, key=lambda points: (points[0][1] + points[1][1]) // 2)
            lane_lines = green_points
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 소켓과 비디오 처리 각각 다른 스레드에서 실행
    socket_manager = SocketManager(SERVER_IP, SERVER_PORT)
    threading.Thread(target=socket_manager.connect, daemon=True).start()
    video_thread(socket_manager)
