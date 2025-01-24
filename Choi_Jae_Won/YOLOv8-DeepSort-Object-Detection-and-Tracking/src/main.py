import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import Sort

# 비디오 캡쳐 및 첫 번째 프레임 가져오기
cap = cv2.VideoCapture("c:/Users/iot19/Documents/YOLOv8-DeepSort-Object-Detection-and-Tracking/assets/Videos/traffic.mp4")
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# YOLO 모델 로딩
model = YOLO(r'c:/Users/iot19/Documents/YOLOv8-DeepSort-Object-Detection-and-Tracking/src/best.pt')

# 클래스 이름 설정 (차량만)
classNames = ["vehicle"]

# 차선 정보 저장
lanes = []  # 여러 차선을 저장할 리스트

# 비디오 프레임 크기 가져오기
success, img = cap.read()
if not success:
    print("Error: Couldn't read the first frame of the video")
    exit()
frame_height, frame_width = img.shape[:2]

# SORT 추적기 초기화
tracker = Sort(max_age=200, min_hits=2, iou_threshold=0.3)

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

def calculate_line_equation(start, end):
    """
    두 점 (start, end)으로부터 직선 방정식의 기울기(m)와 절편(b)을 계산.
    수직선인 경우 None과 x 절편 반환.
    """
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:  # 수직선 처리
        return None, x1
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

# 두 점 사이의 맨해튼 거리 계산 함수
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def draw_lanes(event, x, y, flags, param):
    global lanes
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(lanes) % 2 == 0:  # 새로운 직선 시작
            lanes.append([(x, y)])  # 시작점 추가
        else:
            lanes[-1].append((x, y))  # 끝점 추가
            if len(lanes[-1]) == 2:  # 두 점이 완성되면 직선 방정식 계산
                m, b = calculate_line_equation(lanes[-1][0], lanes[-1][1])
                lanes[-1] = (m, b)  # 직선 방정식으로 저장
                print(f"Lane added: y = {m}x + {b}" if m is not None else f"Lane added: x = {b}")

# 마우스 콜백 설정
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_lanes)

def get_lane_number(lanes, cx, cy):
    """
    차량 중심 (cx, cy)이 직선과 직선 사이에 있으면 해당 차선 번호를 반환.
    - 직선이 하나일 경우 1차선과 2차선을 구분.
    - 직선이 추가로 그려질 경우 오른쪽으로 차선 번호를 확장.
    """
    if len(lanes) == 0:
        return None  # 직선이 없으면 차선 번호 없음

    if len(lanes) == 1:  # 직선이 하나만 있을 경우
        m, b = lanes[0]
        if m is None:  # 수직선인 경우
            x1 = b
            if cx < x1:
                return 1  # 직선 왼쪽: 1차선
            else:
                return 2  # 직선 오른쪽: 2차선
        else:  # 일반 직선인 경우
            y_on_line = m * cx + b
            if cy < y_on_line:
                return 1  # 직선 위쪽: 1차선
            else:
                return 2  # 직선 아래쪽: 2차선

    for i in range(len(lanes) - 1):
        line1 = lanes[i]
        line2 = lanes[i + 1]

        if line1[0] is None:  # 첫 번째 직선이 수직선인 경우
            x1 = line1[1]
            if line2[0] is None:  # 두 번째 직선도 수직선
                x2 = line2[1]
                if x1 < cx < x2 or x2 < cx < x1:
                    return i + 2
            else:  # 두 번째 직선이 일반 직선
                m2, b2 = line2
                y2 = m2 * cx + b2
                if x1 < cx and cy < y2:
                    return i + 2

        elif line2[0] is None:  # 두 번째 직선이 수직선인 경우
            x2 = line2[1]
            m1, b1 = line1
            y1 = m1 * cx + b1
            if cx < x2 and cy > y1:
                return i + 2

        else:  # 두 직선 모두 일반 직선
            m1, b1 = line1
            m2, b2 = line2
            y1 = m1 * cx + b1
            y2 = m2 * cx + b2
            if min(y1, y2) < cy < max(y1, y2):  # y 범위 안에 있는지 확인
                return i + 2

    return len(lanes) + 1  # 가장 오른쪽에 위치한 경우


# 비디오 처리 루프
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
    for obj in tracked_objects:
        x1, y1, x2, y2, id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 차량 추적 정보 저장 및 갱신
        vehicle_id = id_map.get(id, id_counter)
        if id not in id_map:
            id_map[id] = vehicle_id
            id_counter += 1

        if id in tracked_vehicles:
            prev_centroid = tracked_vehicles[id]['centroid']
            distance = manhattan_distance(prev_centroid, (cx, cy))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS 가져오기 (기본값: 30)
            time_interval = 1 / fps
            speed = (distance / time_interval) * scale * 3.6  # 픽셀 -> m/s -> km/h

            # 속도 계산
            if id not in frame_buffer:
                frame_buffer[id] = []
            frame_buffer[id].append(speed)
            if len(frame_buffer[id]) > buffer_size:
                frame_buffer[id].pop(0)

            avg_speed = np.mean(frame_buffer[id])
            tracked_vehicles[id]['speed'] = avg_speed

            # 상자 색상 설정
            box_color = (255, 0, 0)

            # 속도 및 ID 표시
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            cvzone.putTextRect(img, f'ID: {vehicle_id} Speed: {avg_speed:.2f} km/h', 
                               (x1, y1 - 10), scale=1, thickness=2, offset=1, colorT=(255, 255, 255))

            # 차량이 속한 차선 번호 표시
            lane_number = get_lane_number(lanes, cx, cy)
            if lane_number:
                cvzone.putTextRect(img, f'Lane: {lane_number}', 
                                   (x1, y2 + 20), scale=1, thickness=1, offset=1, colorT=(255, 255, 255))

            tracked_vehicles[id]['centroid'] = (cx, cy)
        else:
            tracked_vehicles[id] = {'centroid': (cx, cy), 'speed': 0}

    # 차선 표시
    for lane in lanes:
        if len(lane) == 2:  # 차선이 두 점으로 이루어져 있는지 확인
            cv2.line(img, lane[0], lane[1], (0, 255, 0), 2)

    # 평균 속도 표시
    all_speeds = [v['speed'] for v in tracked_vehicles.values() if 'speed' in v]
    avg_speed = np.mean(all_speeds) if all_speeds else last_avg_speed
    last_avg_speed = avg_speed
    cvzone.putTextRect(img, f'Average Speed: {avg_speed:.2f} km/h', 
                       (20, 30), scale=1, thickness=2, offset=1, colorT=(0, 255, 0))

    # 이미지를 640x480 크기로 리사이즈
    img_resized = cv2.resize(img, (640, 480))

    # 화면 출력
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키 종료
        break

# 종료
cv2.destroyAllWindows()
cap.release()